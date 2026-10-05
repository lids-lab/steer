use crate::common::{
    ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets, ArchivedSemType, ArchivedTableType,
    Offsets,
};
use half::bf16;
use itertools::izip;
use memmap2::{Mmap, MmapOptions};
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::PyObject;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use rand::seq::index;
use rkyv::rancor::Error;
use rkyv::vec::ArchivedVec;
use serde::Deserialize;
use std::alloc;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, Read};
use std::time::{Duration, Instant};

/// Per-item build budget; an item that exceeds it is dropped from the metric.
const TIMEOUT_PER_ITEM_SECS: f64 = 3600.0;

/// Allocate a zeroed Vec<T> via calloc (lazy zero pages). All-zero bits must be valid for T.
unsafe fn alloc_zeroed_vec<T>(len: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }
    let layout = alloc::Layout::array::<T>(len).unwrap();
    let ptr = unsafe { alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        alloc::handle_alloc_error(layout);
    }
    unsafe { Vec::from_raw_parts(ptr as *mut T, len, len) }
}

const MAX_F2P_NBRS: usize = 5;

// STEER: per-task sampler config (toggles, tier lists, keep probabilities), loaded once per Sampler.
#[derive(Clone, Debug, Default, Deserialize)]
struct TierProbs {
    #[serde(default)]
    very_strong: f32,
    #[serde(default)]
    strong: f32,
    #[serde(default)]
    moderate: f32,
    #[serde(default)]
    weak: f32,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct SamplerProbabilities {
    column_filter: TierProbs,
    p2f_edge: TierProbs,
    f2p_edge: TierProbs,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct SamplerToggles {
    #[serde(default)]
    column_filter: bool,
    #[serde(default)]
    p2f_edge_probabilistic_at_push: bool,
    #[serde(default)]
    f2p_edge_probabilistic_at_push: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct SamplerColumnFilterConfig {
    #[serde(default)]
    very_strong: Vec<i32>,
    #[serde(default)]
    strong: Vec<i32>,
    #[serde(default)]
    moderate: Vec<i32>,
    #[serde(default)]
    weak: Vec<i32>,
}

// Edges are "src_table_id->dst_table_id" strings.
#[derive(Clone, Debug, Default, Deserialize)]
struct SamplerEdgeTieredConfig {
    #[serde(default)]
    very_strong: Vec<String>,
    #[serde(default)]
    strong: Vec<String>,
    #[serde(default)]
    moderate: Vec<String>,
    #[serde(default)]
    weak: Vec<String>,
}

// Unknown keys are ignored (no deny_unknown_fields).
#[derive(Clone, Debug, Default, Deserialize)]
struct SamplerTaskConfig {
    #[serde(default)]
    toggles: SamplerToggles,
    keep_probabilities: SamplerProbabilities,
    #[serde(default)]
    column_filter: SamplerColumnFilterConfig,
    #[serde(default)]
    p2f_edge_probabilistic_at_push: SamplerEdgeTieredConfig,
    #[serde(default)]
    f2p_edge_probabilistic_at_push: SamplerEdgeTieredConfig,
}

#[derive(Clone, Debug, Default)]
struct ResolvedEdgeTieredConfig {
    very_strong: HashSet<(i32, i32)>,
    strong: HashSet<(i32, i32)>,
    moderate: HashSet<(i32, i32)>,
    weak: HashSet<(i32, i32)>,
}

#[derive(Clone, Debug, Default)]
struct ResolvedColumnFilterConfig {
    very_strong: HashSet<i32>,
    strong: HashSet<i32>,
    moderate: HashSet<i32>,
    weak: HashSet<i32>,
}

#[derive(Clone, Debug, Default)]
struct ResolvedSamplerTaskConfig {
    toggles: SamplerToggles,
    keep_probabilities: SamplerProbabilities,
    column_filter: ResolvedColumnFilterConfig,
    p2f_edge_probabilistic_at_push: ResolvedEdgeTieredConfig,
    f2p_edge_probabilistic_at_push: ResolvedEdgeTieredConfig,
}

/// Reads the sampler config at `config_path`; a missing path or file means all toggles off.
fn load_task_config(config_path: Option<&str>) -> SamplerTaskConfig {
    let Some(config_path) = config_path else {
        return SamplerTaskConfig::default();
    };
    let config_text = match fs::read_to_string(config_path) {
        Ok(s) => s,
        Err(_) => return SamplerTaskConfig::default(),
    };
    serde_json::from_str(&config_text).unwrap_or_else(|e| {
        panic!("failed to parse sampler config {config_path}: {e}")
    })
}

fn resolve_edge_tier(edges: Vec<String>) -> HashSet<(i32, i32)> {
    edges
        .into_iter()
        .map(|s| {
            let (src, dst) = s
                .split_once("->")
                .unwrap_or_else(|| panic!("invalid edge key '{s}': expected 'src_id->dst_id'"));
            let src_id = src
                .parse::<i32>()
                .unwrap_or_else(|e| panic!("invalid src id '{src}' in edge '{s}': {e}"));
            let dst_id = dst
                .parse::<i32>()
                .unwrap_or_else(|e| panic!("invalid dst id '{dst}' in edge '{s}': {e}"));
            (src_id, dst_id)
        })
        .collect()
}

fn resolve_task_config(config: SamplerTaskConfig) -> ResolvedSamplerTaskConfig {
    ResolvedSamplerTaskConfig {
        toggles: config.toggles,
        keep_probabilities: config.keep_probabilities,
        column_filter: ResolvedColumnFilterConfig {
            very_strong: config.column_filter.very_strong.into_iter().collect(),
            strong: config.column_filter.strong.into_iter().collect(),
            moderate: config.column_filter.moderate.into_iter().collect(),
            weak: config.column_filter.weak.into_iter().collect(),
        },
        p2f_edge_probabilistic_at_push: ResolvedEdgeTieredConfig {
            very_strong: resolve_edge_tier(config.p2f_edge_probabilistic_at_push.very_strong),
            strong: resolve_edge_tier(config.p2f_edge_probabilistic_at_push.strong),
            moderate: resolve_edge_tier(config.p2f_edge_probabilistic_at_push.moderate),
            weak: resolve_edge_tier(config.p2f_edge_probabilistic_at_push.weak),
        },
        f2p_edge_probabilistic_at_push: ResolvedEdgeTieredConfig {
            very_strong: resolve_edge_tier(config.f2p_edge_probabilistic_at_push.very_strong),
            strong: resolve_edge_tier(config.f2p_edge_probabilistic_at_push.strong),
            moderate: resolve_edge_tier(config.f2p_edge_probabilistic_at_push.moderate),
            weak: resolve_edge_tier(config.f2p_edge_probabilistic_at_push.weak),
        },
    }
}

/// Keep probability for a column's tier (untiered columns count as weak). Task-table cells bypass this.
fn column_filter_keep_probability(col_idx: i32, config: &ResolvedSamplerTaskConfig) -> f32 {
    if config.column_filter.very_strong.contains(&col_idx) {
        return config.keep_probabilities.column_filter.very_strong;
    }
    if config.column_filter.strong.contains(&col_idx) {
        return config.keep_probabilities.column_filter.strong;
    }
    if config.column_filter.moderate.contains(&col_idx) {
        return config.keep_probabilities.column_filter.moderate;
    }
    if config.column_filter.weak.contains(&col_idx) {
        return config.keep_probabilities.column_filter.weak;
    }
    // columns not in any tier are treated as weak
    config.keep_probabilities.column_filter.weak
}

/// Keep probability for a p2f edge's tier. Edges touching the task table bypass this.
fn p2f_edge_probabilistic_at_push_keep_probability(
    src_table_idx: i32,
    dst_table_idx: i32,
    config: &ResolvedSamplerTaskConfig,
) -> f32 {
    let edge = (src_table_idx, dst_table_idx);
    if config.p2f_edge_probabilistic_at_push.very_strong.contains(&edge) {
        return config.keep_probabilities.p2f_edge.very_strong;
    }
    if config.p2f_edge_probabilistic_at_push.strong.contains(&edge) {
        return config.keep_probabilities.p2f_edge.strong;
    }
    if config.p2f_edge_probabilistic_at_push.moderate.contains(&edge) {
        return config.keep_probabilities.p2f_edge.moderate;
    }
    if config.p2f_edge_probabilistic_at_push.weak.contains(&edge) {
        return config.keep_probabilities.p2f_edge.weak;
    }
    config.keep_probabilities.p2f_edge.weak
}

/// Keep probability for an f2p edge's tier. Edges touching the task table bypass this.
fn f2p_edge_probabilistic_at_push_keep_probability(
    src_table_idx: i32,
    dst_table_idx: i32,
    config: &ResolvedSamplerTaskConfig,
) -> f32 {
    let edge = (src_table_idx, dst_table_idx);
    if config.f2p_edge_probabilistic_at_push.very_strong.contains(&edge) {
        return config.keep_probabilities.f2p_edge.very_strong;
    }
    if config.f2p_edge_probabilistic_at_push.strong.contains(&edge) {
        return config.keep_probabilities.f2p_edge.strong;
    }
    if config.f2p_edge_probabilistic_at_push.moderate.contains(&edge) {
        return config.keep_probabilities.f2p_edge.moderate;
    }
    if config.f2p_edge_probabilistic_at_push.weak.contains(&edge) {
        return config.keep_probabilities.f2p_edge.weak;
    }
    config.keep_probabilities.f2p_edge.weak
}

/// Expected reasons an item cannot produce a sequence; eval drops such items.
#[derive(Debug)]
enum BuildError {
    MissingTargetCol,
    NanTargetValue,
}

#[inline]
fn check_deadline(deadline: Instant) {
    if Instant::now() >= deadline {
        panic!("timeout_per_item exceeded");
    }
}

/// Stride (power of two) for amortized deadline checks in hot inner loops.
const DEADLINE_CHECK_EVERY: usize = 1024;

struct Vecs {
    node_idxs: Vec<i32>,
    f2p_nbr_idxs: Vec<i32>,
    table_name_idxs: Vec<i32>,
    col_name_idxs: Vec<i32>,
    class_value_idxs: Vec<i32>,
    col_name_values: Vec<bf16>,
    sem_types: Vec<i32>,
    number_values: Vec<bf16>,
    text_values: Vec<bf16>,
    datetime_values: Vec<bf16>,
    boolean_values: Vec<bf16>,
    is_targets: Vec<bool>,
    is_task_nodes: Vec<bool>,
    is_padding: Vec<bool>,
    timestamps: Vec<i32>,
    // `false` for phantom slots (last-batch overshoot or a dropped item).
    batch_mask: Vec<bool>,
    seq_len: usize,
}

struct Slices<'a> {
    node_idxs: &'a mut [i32],
    f2p_nbr_idxs: &'a mut [i32],
    table_name_idxs: &'a mut [i32],
    col_name_idxs: &'a mut [i32],
    class_value_idxs: &'a mut [i32],
    col_name_values: &'a mut [bf16],
    sem_types: &'a mut [i32],
    number_values: &'a mut [bf16],
    text_values: &'a mut [bf16],
    datetime_values: &'a mut [bf16],
    boolean_values: &'a mut [bf16],
    is_targets: &'a mut [bool],
    is_task_nodes: &'a mut [bool],
    is_padding: &'a mut [bool],
    timestamps: &'a mut [i32],
}

impl Vecs {
    fn new(bs: usize, seq_len: usize, d_text: usize) -> Self {
        let l = bs * seq_len;
        Self {
            node_idxs: vec![-1; l],
            f2p_nbr_idxs: vec![-1; l * MAX_F2P_NBRS],
            table_name_idxs: vec![0; l],
            col_name_idxs: vec![0; l],
            class_value_idxs: vec![-1; l],
            col_name_values: unsafe { alloc_zeroed_vec(l * d_text) },
            sem_types: vec![0; l],
            number_values: unsafe { alloc_zeroed_vec(l) },
            text_values: unsafe { alloc_zeroed_vec(l * d_text) },
            datetime_values: unsafe { alloc_zeroed_vec(l) },
            boolean_values: unsafe { alloc_zeroed_vec(l) },
            is_targets: vec![false; l],
            is_task_nodes: vec![false; l],
            is_padding: vec![true; l],
            timestamps: vec![i32::MIN; l],
            batch_mask: vec![true; bs],
            seq_len,
        }
    }

    fn chunks_exact_mut(
        &mut self,
        seq_len: usize,
        d_text: usize,
    ) -> impl Iterator<Item = Slices<'_>> {
        izip!(
            self.node_idxs.chunks_exact_mut(seq_len),
            self.f2p_nbr_idxs.chunks_exact_mut(seq_len * MAX_F2P_NBRS),
            self.table_name_idxs.chunks_exact_mut(seq_len),
            self.col_name_idxs.chunks_exact_mut(seq_len),
            self.class_value_idxs.chunks_exact_mut(seq_len),
            self.col_name_values.chunks_exact_mut(seq_len * d_text),
            self.sem_types.chunks_exact_mut(seq_len),
            self.number_values.chunks_exact_mut(seq_len),
            self.text_values.chunks_exact_mut(seq_len * d_text),
            self.datetime_values.chunks_exact_mut(seq_len),
            self.boolean_values.chunks_exact_mut(seq_len),
            self.is_targets.chunks_exact_mut(seq_len),
            self.is_task_nodes.chunks_exact_mut(seq_len),
            self.is_padding.chunks_exact_mut(seq_len),
            self.timestamps.chunks_exact_mut(seq_len)
        )
        .map(
            |(
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                is_targets,
                is_task_nodes,
                is_padding,
                timestamps,
            )| Slices {
                node_idxs,
                f2p_nbr_idxs,
                table_name_idxs,
                col_name_idxs,
                class_value_idxs,
                col_name_values,
                sem_types,
                number_values,
                text_values,
                datetime_values,
                boolean_values,
                is_targets,
                is_task_nodes,
                is_padding,
                timestamps,
            },
        )
    }
    fn into_pyobject<'a>(self, py: Python<'a>) -> PyResult<Vec<PyObject>> {
        Ok(vec![
            ("node_idxs", PyArray1::from_vec(py, self.node_idxs))
                .into_py_any(py)
                .unwrap(),
            ("f2p_nbr_idxs", PyArray1::from_vec(py, self.f2p_nbr_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "table_name_idxs",
                PyArray1::from_vec(py, self.table_name_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            ("col_name_idxs", PyArray1::from_vec(py, self.col_name_idxs))
                .into_py_any(py)
                .unwrap(),
            (
                "class_value_idxs",
                PyArray1::from_vec(py, self.class_value_idxs),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "col_name_values",
                PyArray1::from_vec(py, self.col_name_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("sem_types", PyArray1::from_vec(py, self.sem_types))
                .into_py_any(py)
                .unwrap(),
            ("number_values", PyArray1::from_vec(py, self.number_values))
                .into_py_any(py)
                .unwrap(),
            ("text_values", PyArray1::from_vec(py, self.text_values))
                .into_py_any(py)
                .unwrap(),
            (
                "datetime_values",
                PyArray1::from_vec(py, self.datetime_values),
            )
                .into_py_any(py)
                .unwrap(),
            (
                "boolean_values",
                PyArray1::from_vec(py, self.boolean_values),
            )
                .into_py_any(py)
                .unwrap(),
            ("is_targets", PyArray1::from_vec(py, self.is_targets))
                .into_py_any(py)
                .unwrap(),
            ("is_task_nodes", PyArray1::from_vec(py, self.is_task_nodes))
                .into_py_any(py)
                .unwrap(),
            ("is_padding", PyArray1::from_vec(py, self.is_padding))
                .into_py_any(py)
                .unwrap(),
            ("timestamps", PyArray1::from_vec(py, self.timestamps))
                .into_py_any(py)
                .unwrap(),
            ("batch_mask", PyArray1::from_vec(py, self.batch_mask))
                .into_py_any(py)
                .unwrap(),
            ("seq_len", self.seq_len).into_py_any(py).unwrap(),
        ])
    }
}

struct Dataset {
    mmap: Mmap,
    text_mmap: Mmap,
    p2f_adj_mmap: Mmap,
    offsets: Vec<i64>,
}

struct Item {
    dataset_idx: i32,
    node_idx: i32,
    table_name: String,
}

#[pyclass]
pub struct Sampler {
    datasets: HashMap<String, Dataset>,
    items: Vec<Item>,
    local_ctx_size: usize, // max cells collected per BFS
    bfs_width: usize,      // max DB nodes expanded per BFS level
    d_text: usize,
    shuffle_seed: u64, // picks the evaluated items and their order
    context_seed: u64, // seeds the per-item context randomness
    target_columns: Vec<i32>,
    items_per_task: i64,
    dataset_tuples: Vec<(String, String, i32, i32)>, // (db_name, table_name, node_idx_offset, num_nodes)
    task_config: ResolvedSamplerTaskConfig,          // STEER config for this task
}

#[pymethods]
impl Sampler {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        dataset_tuples: Vec<(String, String, i32, i32)>,
        local_ctx_size: usize,
        bfs_width: usize,
        embedding_model: &str,
        pre_dir: String,
        d_text: usize,
        shuffle_seed: u64,
        context_seed: u64,
        target_columns: Vec<i32>,
        items_per_task: i64,
        sampler_config_path: Option<String>,
    ) -> Self {
        py.allow_threads(|| {
            assert!(!dataset_tuples.is_empty(), "dataset_tuples must be non-empty");

            // Pre-fault the data into RAM instead of cold-faulting it per item.
            let mut mmap_opts = MmapOptions::new();
            mmap_opts.populate();

            let mut datasets: HashMap<String, Dataset> = HashMap::new();
            for (db_name, _, _, _) in dataset_tuples.iter() {
                if !datasets.contains_key(db_name) {
                    datasets.insert(
                        db_name.clone(),
                        Self::load_dataset(db_name, &pre_dir, &mmap_opts, embedding_model),
                    );
                }
            }

            let task_config =
                resolve_task_config(load_task_config(sampler_config_path.as_deref()));
            let mut sampler = Self {
                datasets,
                items: Vec::new(),
                local_ctx_size,
                bfs_width,
                d_text,
                shuffle_seed: StdRng::seed_from_u64(shuffle_seed).random(),
                context_seed: StdRng::seed_from_u64(context_seed).random(),
                target_columns,
                items_per_task,
                dataset_tuples,
                task_config,
            };
            sampler.create_items();
            sampler
        })
    }

    #[getter]
    fn num_items(&self) -> usize {
        self.items.len()
    }

    fn batch_py(
        &self,
        py: Python<'_>,
        batch_idx: usize,
        bs: usize,
        ctx_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        self.batch(batch_idx, bs, ctx_size).into_pyobject(py)
    }
}

impl Sampler {
    fn load_dataset(
        db_name: &str,
        pre_dir: &str,
        mmap_opts: &MmapOptions,
        embedding_model: &str,
    ) -> Dataset {
        let pre_path = format!("{}/{}", pre_dir, db_name);

        let nodes_path = format!("{}/nodes.rkyv", pre_path);
        let file = fs::File::open(&nodes_path).unwrap();
        let mmap = unsafe { mmap_opts.map(&file).unwrap() };

        let text_path = format!("{}/text_emb_{}.bin", pre_path, embedding_model);
        let text_file = fs::File::open(&text_path).unwrap();
        let text_mmap = unsafe { mmap_opts.map(&text_file).unwrap() };

        let offsets_path = format!("{}/offsets.rkyv", pre_path);
        let file = fs::File::open(&offsets_path).unwrap();
        let mut bytes = Vec::new();
        BufReader::new(file).read_to_end(&mut bytes).unwrap();
        let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes).unwrap();
        let offsets = rkyv::deserialize::<Offsets, Error>(archived).unwrap();
        let offsets = offsets.offsets;

        let p2f_adj_path = format!("{}/p2f_adj.rkyv", pre_path);
        let p2f_adj_file = fs::File::open(&p2f_adj_path).unwrap();
        let p2f_adj_mmap = unsafe { mmap_opts.map(&p2f_adj_file).unwrap() };

        Dataset {
            mmap,
            text_mmap,
            p2f_adj_mmap,
            offsets,
        }
    }

    /// Build the item list: a per-task random subset, shuffled.
    fn create_items(&mut self) {
        self.items.clear();

        for (i, &(_, ref table_name, node_idx_offset, num_nodes)) in
            self.dataset_tuples.iter().enumerate()
        {
            let num_to_sample = if self.items_per_task == -1 {
                num_nodes as usize
            } else {
                (num_nodes as usize).min(self.items_per_task as usize)
            };

            let rng_seed = self.shuffle_seed + (i as u64);
            let mut rng = StdRng::seed_from_u64(rng_seed);

            let sampled_indices = index::sample(&mut rng, num_nodes as usize, num_to_sample);

            for idx in sampled_indices.iter() {
                let node_idx = node_idx_offset + idx as i32;
                self.items.push(Item {
                    dataset_idx: i as i32,
                    node_idx,
                    table_name: table_name.clone(),
                });
            }
        }

        let mut rng = StdRng::seed_from_u64(self.shuffle_seed);
        self.items.shuffle(&mut rng);
    }

    fn batch(&self, batch_idx: usize, bs: usize, ctx_size: usize) -> Vecs {
        let offset = batch_idx * bs;
        let mut vecs = Vecs::new(bs, ctx_size, self.d_text);
        for (i, m) in vecs.batch_mask.iter_mut().enumerate() {
            *m = offset + i < self.items.len();
        }

        // An item that fails or times out becomes a phantom slot (batch_mask=false)
        // rather than being replaced, so the reported predictions stay faithful.
        let timeout = Duration::from_secs_f64(TIMEOUT_PER_ITEM_SECS);
        let timed_out: Vec<bool> = vecs
            .chunks_exact_mut(ctx_size, self.d_text)
            .enumerate()
            .map(|(i, mut slices)| {
                let j = offset + i;
                if j >= self.items.len() {
                    return false; // overshoot slot: stays padded
                }
                let item = &self.items[j];
                let deadline = Instant::now() + timeout;
                let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.seq_build(item, &mut slices, ctx_size, deadline)
                }));
                let failed = !matches!(caught, Ok(Ok(())));
                if failed {
                    slices.is_targets.fill(false);
                    slices.is_padding.fill(true);
                    let db_name = &self.dataset_tuples[item.dataset_idx as usize].0;
                    eprintln!(
                        "\n\x1b[31meval: dropping item (db={}, table={}, node_idx={}): build failed/timed out\x1b[0m",
                        db_name, item.table_name, item.node_idx
                    );
                }
                failed
            })
            .collect();
        for (i, &failed) in timed_out.iter().enumerate() {
            if failed {
                vecs.batch_mask[i] = false;
            }
        }
        vecs
    }

    /// Build one item's context: the target cell first, then a BFS around the target node.
    fn seq_build(
        &self,
        item: &Item,
        slices: &mut Slices,
        ctx_len: usize,
        deadline: Instant,
    ) -> Result<(), BuildError> {
        check_deadline(deadline);
        let db_name = &self.dataset_tuples[item.dataset_idx as usize].0;
        let dataset = &self.datasets[db_name];
        let target_column = self.target_columns[item.dataset_idx as usize];

        let target_node_idx = item.node_idx;
        let target_node = get_node(dataset, target_node_idx);

        let target_cell_i = match target_node
            .col_name_idxs
            .iter()
            .position(|&col_idx| i32::from(col_idx) == target_column)
        {
            Some(i) => i,
            None => return Err(BuildError::MissingTargetCol),
        };

        let target_value_is_nan = {
            let sem = &target_node.sem_types[target_cell_i];
            match sem {
                ArchivedSemType::Number => {
                    f32::from(target_node.number_values[target_cell_i]).is_nan()
                }
                ArchivedSemType::DateTime => {
                    f32::from(target_node.datetime_values[target_cell_i]).is_nan()
                }
                ArchivedSemType::Boolean => {
                    f32::from(target_node.boolean_values[target_cell_i]).is_nan()
                }
                ArchivedSemType::Text => false,
            }
        };
        if target_value_is_nan {
            return Err(BuildError::NanTargetValue);
        }

        assert!(
            self.local_ctx_size <= ctx_len,
            "local_ctx_size {} is > ctx_len={}",
            self.local_ctx_size,
            ctx_len
        );

        let step_seed: u64 = StdRng::seed_from_u64(self.context_seed).random();

        let mut visited_in_ctx: HashSet<i32> = HashSet::with_capacity(ctx_len);
        let mut visited_at_depth: HashMap<i32, usize> = HashMap::with_capacity(ctx_len);
        // (node_idx, cell_i) in context order
        let mut cells_to_add: Vec<(i32, usize)> = Vec::with_capacity(ctx_len);
        let mut bfs_rng = StdRng::seed_from_u64(
            step_seed
                .wrapping_add(target_node_idx as u64)
                .wrapping_add(0xB0B0_B0B0_B0B0_B0B0),
        );

        // The target cell always goes first.
        cells_to_add.push((target_node_idx, target_cell_i));

        let bfs_nodes = self.bfs_collect_nodes(
            dataset,
            target_node_idx,
            &mut bfs_rng,
            self.local_ctx_size,
            self.bfs_width,
            &mut visited_at_depth,
            deadline,
        );

        'fill_ctx: for (bfs_node_idx, _depth) in bfs_nodes {
            check_deadline(deadline);
            if visited_in_ctx.contains(&bfs_node_idx) {
                continue;
            }
            visited_in_ctx.insert(bfs_node_idx);

            let node = get_node(dataset, bfs_node_idx);
            for cell_i in 0..node.col_name_idxs.len() {
                if cell_i & (DEADLINE_CHECK_EVERY - 1) == 0 {
                    check_deadline(deadline);
                }
                let col_idx: i32 = node.col_name_idxs[cell_i].into();

                // The target cell is already added.
                if bfs_node_idx == target_node_idx && col_idx == target_column {
                    continue;
                }

                // STEER: keep this cell with its column's tier probability (task-table cells always kept).
                if self.task_config.toggles.column_filter && !node.is_task_node {
                    let keep_prob = column_filter_keep_probability(col_idx, &self.task_config);
                    if keep_prob <= 0.0 || bfs_rng.random::<f32>() > keep_prob {
                        continue;
                    }
                }

                cells_to_add.push((bfs_node_idx, cell_i));

                if cells_to_add.len() == ctx_len {
                    break 'fill_ctx;
                }
            }
        }

        let mut seq_i = 0;
        let mut last_node_idx: i32 = -1;
        let mut cached_node: Option<&ArchivedNode> = None;
        for &(node_idx, cell_i) in cells_to_add.iter() {
            check_deadline(deadline);
            if seq_i >= ctx_len {
                break;
            }

            if node_idx != last_node_idx {
                cached_node = Some(get_node(dataset, node_idx));
                last_node_idx = node_idx;
            }

            self.add_single_cell(
                dataset,
                cached_node.unwrap(),
                cell_i,
                target_node_idx,
                target_column,
                &mut seq_i,
                slices,
            );
        }
        Ok(())
    }

    /// Write one cell of `node` into the next sequence slot.
    #[allow(clippy::too_many_arguments)]
    fn add_single_cell(
        &self,
        dataset: &Dataset,
        node: &ArchivedNode,
        cell_i: usize,
        target_node_idx: i32,
        target_column: i32,
        seq_i: &mut usize,
        slices: &mut Slices,
    ) {
        slices.node_idxs[*seq_i] = node.node_idx.into();

        assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS);
        for (j, f2p_nbr_idx) in node.f2p_nbr_idxs.iter().enumerate() {
            slices.f2p_nbr_idxs[*seq_i * MAX_F2P_NBRS + j] = f2p_nbr_idx.into();
        }

        slices.table_name_idxs[*seq_i] = node.table_name_idx.into();
        slices.col_name_idxs[*seq_i] = node.col_name_idxs[cell_i].into();
        slices.class_value_idxs[*seq_i] = node.class_value_idx[cell_i].into();
        slices.col_name_values[*seq_i * self.d_text..(*seq_i + 1) * self.d_text]
            .copy_from_slice(get_text_emb(dataset, slices.col_name_idxs[*seq_i], self.d_text));

        slices.sem_types[*seq_i] = node.sem_types[cell_i].clone() as i32;
        slices.number_values[*seq_i] = bf16::from_f32(node.number_values[cell_i].into());

        let text_idx: i32 = node.text_values[cell_i].into();
        slices.text_values[*seq_i * self.d_text..(*seq_i + 1) * self.d_text]
            .copy_from_slice(get_text_emb(dataset, text_idx, self.d_text));

        slices.datetime_values[*seq_i] = bf16::from_f32(node.datetime_values[cell_i].into());
        slices.boolean_values[*seq_i] = bf16::from_f32(node.boolean_values[cell_i].into());

        // A NaN value means missing data; such a cell is never the target.
        let value_is_nan = {
            let sem = &node.sem_types[cell_i];
            match sem {
                ArchivedSemType::Number => f32::from(node.number_values[cell_i]).is_nan(),
                ArchivedSemType::DateTime => f32::from(node.datetime_values[cell_i]).is_nan(),
                ArchivedSemType::Boolean => f32::from(node.boolean_values[cell_i]).is_nan(),
                ArchivedSemType::Text => false, // text uses index, not float
            }
        };

        slices.is_targets[*seq_i] = !value_is_nan
            && node.node_idx == target_node_idx
            && node.col_name_idxs[cell_i] == target_column;

        slices.is_task_nodes[*seq_i] =
            node.is_task_node || (node.col_name_idxs[cell_i] == target_column);
        slices.is_padding[*seq_i] = false;
        slices.timestamps[*seq_i] = match node.timestamp.as_ref() {
            Some(ts) => (*ts).into(),
            None => i32::MIN,
        };

        *seq_i += 1;
    }

    /// BFS from `start_idx`, collecting nodes until the `local_ctx_size` cell budget is reached.
    #[allow(clippy::too_many_arguments)]
    fn bfs_collect_nodes(
        &self,
        dataset: &Dataset,
        start_idx: i32,
        rng: &mut StdRng,
        local_ctx_size: usize,
        bfs_width: usize,
        visited_at_depth: &mut HashMap<i32, usize>,
        deadline: Instant,
    ) -> Vec<(i32, usize)> {
        let mut result: Vec<(i32, usize)> = Vec::with_capacity(128);

        let start_node = get_node(dataset, start_idx);
        let mut num_cells: f64 = 0.0;

        // Two frontier data structures:
        // f2p_ftr: stack of (depth, node_idx) for f2p edges
        // p2f_ftr: vector of vectors, one per depth level, for p2f edges
        let mut f2p_ftr: Vec<(usize, i32)> = Vec::with_capacity(64);
        let mut p2f_ftr: Vec<Vec<i32>> = vec![vec![start_idx]];
        // (child_node_idx, child_table_idx) -- the table is needed at push
        // time by the p2f_edge_probabilistic_at_push tier lookup below.
        let mut db_p2f_ftr: Vec<(i32, i32)> = Vec::with_capacity(bfs_width);

        loop {
            check_deadline(deadline);
            // Select node
            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                match p2f_ftr.iter().position(|v| !v.is_empty()) {
                    None => return result,
                    Some(depth) => {
                        let r = rng.random_range(0..p2f_ftr[depth].len());
                        let l = p2f_ftr[depth].len();
                        p2f_ftr[depth].swap(r, l - 1);
                        let node_idx = p2f_ftr[depth].pop().unwrap();
                        (depth, node_idx)
                    }
                }
            };

            // Check if node was visited at a depth <= current depth
            if let Some(&prev_depth) = visited_at_depth.get(&node_idx)
                && prev_depth <= depth
            {
                continue;
            }

            let node = get_node(dataset, node_idx);

            // STEER: with column_filter on, charge the expected surviving cell count (sum of keep probabilities).
            let node_cell_estimate: f64 = if self.task_config.toggles.column_filter && !node.is_task_node {
                node.col_name_idxs
                    .iter()
                    .map(|&ci| {
                        let col_idx: i32 = ci.into();
                        column_filter_keep_probability(col_idx, &self.task_config) as f64
                    })
                    .sum()
            } else {
                node.col_name_idxs.len() as f64
            };
            num_cells += node_cell_estimate;
            if num_cells >= local_ctx_size as f64 {
                return result;
            }

            // Record the depth at which this node was visited
            visited_at_depth.insert(node_idx, depth);

            result.push((node_idx, depth));

            for edge in node.f2p_edges.iter() {
                // STEER: keep this f2p edge with its tier's probability (task-table edges always kept).
                if self.task_config.toggles.f2p_edge_probabilistic_at_push
                    && !node.is_task_node
                    && edge.table_name_idx != start_node.table_name_idx
                {
                    let keep_prob = f2p_edge_probabilistic_at_push_keep_probability(
                        node.table_name_idx.into(),
                        edge.table_name_idx.into(),
                        &self.task_config,
                    );
                    if keep_prob <= 0.0 || rng.random::<f32>() > keep_prob {
                        continue;
                    }
                }
                f2p_ftr.push((depth + 1, edge.node_idx.into()));
            }

            // Get p2f edges and process them
            let p2f_edges = get_p2f_edges(dataset, node_idx);

            // Reuse pre-allocated storage for db edges to be subsampled
            db_p2f_ftr.clear();

            // The edges are sorted by timestamp, so we can binary search to find valid ones
            let valid_edges = p2f_edges.as_slice().partition_point(|edge| {
                edge.timestamp.is_none()
                    || (start_node.timestamp.is_some() && edge.timestamp <= start_node.timestamp)
            });

            // Filter valid edges by table constraints
            let p2f_edges = &p2f_edges.as_slice()[..valid_edges];

            for (i, edge) in p2f_edges.iter().enumerate() {
                if i & (DEADLINE_CHECK_EVERY - 1) == 0 {
                    check_deadline(deadline);
                }
                // include edges to task table only if seed node belongs to the task table
                if edge.table_name_idx != start_node.table_name_idx
                    && edge.table_type != ArchivedTableType::Db
                {
                    continue;
                }

                if edge.table_type == ArchivedTableType::Db {
                    db_p2f_ftr.push((edge.node_idx.into(), edge.table_name_idx.into()));
                    continue;
                }

                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(edge.node_idx.into());
            }

            // Subsample DB edges based on bfs_width
            let idxs = if db_p2f_ftr.len() > bfs_width {
                index::sample(rng, db_p2f_ftr.len(), bfs_width).into_vec()
            } else {
                (0..db_p2f_ftr.len()).collect::<Vec<_>>()
            };

            for idx in idxs.iter() {
                let (child_node_idx, child_table_idx) = db_p2f_ftr[*idx];
                // STEER: keep this p2f edge with its tier's probability (task-table sources always kept).
                if self.task_config.toggles.p2f_edge_probabilistic_at_push && !node.is_task_node {
                    let keep_prob = p2f_edge_probabilistic_at_push_keep_probability(
                        node.table_name_idx.into(),
                        child_table_idx,
                        &self.task_config,
                    );
                    if keep_prob <= 0.0 || rng.random::<f32>() > keep_prob {
                        continue;
                    }
                }
                if depth + 1 >= p2f_ftr.len() {
                    for _i in p2f_ftr.len()..=depth + 1 {
                        p2f_ftr.push(vec![]);
                    }
                }
                p2f_ftr[depth + 1].push(child_node_idx);
            }
        }
    }
}

fn get_node(dataset: &Dataset, idx: i32) -> &ArchivedNode {
    let l = dataset.offsets[idx as usize] as usize;
    let r = dataset.offsets[(idx + 1) as usize] as usize;
    let bytes = &dataset.mmap[l..r];
    unsafe { rkyv::access_unchecked::<ArchivedNode>(bytes) }
}

fn get_p2f_edges(dataset: &Dataset, idx: i32) -> &ArchivedVec<ArchivedEdge> {
    let bytes = &dataset.p2f_adj_mmap[..];
    let p2f_adj = unsafe { rkyv::access_unchecked::<ArchivedAdj>(bytes) };
    &p2f_adj.adj[idx as usize]
}

fn get_text_emb(dataset: &Dataset, idx: i32, d_text: usize) -> &[bf16] {
    let (pref, text_emb, suf) = unsafe { dataset.text_mmap.align_to::<bf16>() };
    assert!(pref.is_empty() && suf.is_empty());
    &text_emb[(idx as usize) * d_text..(idx as usize + 1) * d_text]
}
