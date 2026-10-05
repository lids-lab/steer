use crate::common::{
    ArchivedAdj, ArchivedEdge, ArchivedNode, ArchivedOffsets, ArchivedTableType, Offsets,
};
use clap::Parser;
use half::bf16;
use itertools::izip;
use memmap2::Mmap;
use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::PyObject;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::{pyclass, pymethods};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::seq::index;
use rkyv::rancor::Error;
use rkyv::vec::ArchivedVec;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::str;
use std::time::Instant;

const MAX_F2P_NBRS: usize = 5;

// keep-probabilities are now config-driven (SamplerProbabilities, below)
// instead of compile-time constants -- these are only the *defaults* used
// when a task config doesn't specify keep_probabilities at all (old configs,
// or STEER's baseline runs), so behavior is unchanged for anyone not opting
// into the new field.
#[derive(Clone, Debug, Deserialize)]
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

#[derive(Clone, Debug, Deserialize)]
struct SamplerProbabilities {
    #[serde(default = "default_column_filter_probs")]
    column_filter: TierProbs,
    #[serde(default = "default_p2f_edge_probs")]
    p2f_edge: TierProbs,
    #[serde(default = "default_f2p_edge_probs")]
    f2p_edge: TierProbs,
}

fn default_column_filter_probs() -> TierProbs {
    TierProbs { very_strong: 1.0, strong: 1.0, moderate: 0.2, weak: 0.0 }
}
fn default_p2f_edge_probs() -> TierProbs {
    TierProbs { very_strong: 1.0, strong: 0.6, moderate: 0.0, weak: 0.0 }
}
fn default_f2p_edge_probs() -> TierProbs {
    TierProbs { very_strong: 1.0, strong: 1.0, moderate: 0.0, weak: 0.0 }
}

impl Default for SamplerProbabilities {
    fn default() -> Self {
        Self {
            column_filter: default_column_filter_probs(),
            p2f_edge: default_p2f_edge_probs(),
            f2p_edge: default_f2p_edge_probs(),
        }
    }
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

// Edges stored as "src_table_id->dst_table_id" strings after conversion.
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

#[derive(Clone, Debug, Default, Deserialize)]
struct SamplerTaskConfig {
    #[serde(default)]
    toggles: SamplerToggles,
    #[serde(default)]
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
    masks: Vec<bool>,
    is_targets: Vec<bool>,
    is_task_nodes: Vec<bool>,
    is_padding: Vec<bool>,
    true_batch_size: usize,
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
    masks: &'a mut [bool],
    is_targets: &'a mut [bool],
    is_task_nodes: &'a mut [bool],
    is_padding: &'a mut [bool],
}
// Vecs::new(self.batch_size, self.seq_len, true_batch_size, self.d_text);
impl Vecs {
    fn new(batch_size: usize, seq_len: usize, true_batch_size: usize, d_text: usize) -> Self {
        let l = batch_size * seq_len;
        Self {
            node_idxs: vec![-1; l],
            f2p_nbr_idxs: vec![-1; l * MAX_F2P_NBRS],
            table_name_idxs: vec![0; l],
            col_name_idxs: vec![0; l],
            class_value_idxs: vec![-1; l],
            col_name_values: vec![bf16::ZERO; l * d_text],
            sem_types: vec![0; l],
            number_values: vec![bf16::ZERO; l],
            text_values: vec![bf16::ZERO; l * d_text],
            datetime_values: vec![bf16::ZERO; l],
            boolean_values: vec![bf16::ZERO; l],
            masks: vec![false; l],
            is_targets: vec![false; l],
            is_task_nodes: vec![false; l],
            is_padding: vec![true; l],
            true_batch_size,
        }
    }

    fn chunks_exact_mut(&mut self, seq_len: usize, d_text: usize) -> impl Iterator<Item = Slices> {
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
            self.masks.chunks_exact_mut(seq_len),
            self.is_targets.chunks_exact_mut(seq_len),
            self.is_task_nodes.chunks_exact_mut(seq_len),
            self.is_padding.chunks_exact_mut(seq_len),
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
                masks,
                is_targets,
                is_task_nodes,
                is_padding,
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
                masks,
                is_targets,
                is_task_nodes,
                is_padding,
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
            ("masks", PyArray1::from_vec(py, self.masks))
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
            ("true_batch_size", self.true_batch_size)
                .into_py_any(py)
                .unwrap(),
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
}

#[pyclass]
pub struct Sampler {
    batch_size: usize,
    seq_len: usize,
    rank: usize,
    world_size: usize,
    datasets: Vec<Dataset>,
    items: Vec<Item>,
    max_bfs_width: usize,
    epoch: u64,
    d_text: usize,
    seed: u64,
    target_columns: Vec<i32>,
    columns_to_drop: Vec<Vec<i32>>,
    task_configs: Vec<ResolvedSamplerTaskConfig>,
}

#[pymethods]
impl Sampler {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset_tuples: Vec<(String, i32, i32)>,
        batch_size: usize,
        seq_len: usize,
        rank: usize,
        world_size: usize,
        max_bfs_width: usize,
        embedding_model: &str,
        d_text: usize,
        seed: u64,
        target_columns: Vec<i32>,
        columns_to_drop: Vec<Vec<i32>>,
        sampler_config_paths: Vec<Option<String>>,
    ) -> Self {
        let mut datasets = Vec::new();
        let mut items = Vec::new();
        let mut task_configs = Vec::new();
        for (i, (db_name, node_idx_offset, num_nodes)) in dataset_tuples.into_iter().enumerate() {
            // Preprocessed data lives alongside this crate, under models/rt/data/pre/<dataset>
            // (STEER's own self-contained layout -- not relational-transformer's).
            let pre_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("data")
                .join("pre")
                .join(&db_name)
                .to_string_lossy()
                .into_owned();
            let nodes_path = format!("{}/nodes.rkyv", pre_path);
            let file = fs::File::open(&nodes_path).unwrap();
            let mmap = unsafe { Mmap::map(&file).unwrap() };

            let text_path = format!("{}/text_emb_{}.bin", pre_path, embedding_model);
            let text_file = fs::File::open(&text_path).unwrap();
            let text_mmap = unsafe { Mmap::map(&text_file).unwrap() };

            let offsets_path = format!("{}/offsets.rkyv", pre_path);
            let file = fs::File::open(&offsets_path).unwrap();
            let mut bytes = Vec::new();
            BufReader::new(file).read_to_end(&mut bytes).unwrap();
            let archived = rkyv::access::<ArchivedOffsets, Error>(&bytes).unwrap();
            let offsets = rkyv::deserialize::<Offsets, Error>(archived).unwrap();
            let offsets = offsets.offsets;

            let p2f_adj_path = format!("{}/p2f_adj.rkyv", pre_path);
            let p2f_adj_file = fs::File::open(&p2f_adj_path).unwrap();
            let p2f_adj_mmap = unsafe { Mmap::map(&p2f_adj_file).unwrap() };
            let target = target_columns[i];
            let task_config = resolve_task_config(load_task_config(sampler_config_paths[i].as_deref()));

            datasets.push(Dataset {
                mmap,
                text_mmap,
                p2f_adj_mmap,
                offsets,
            });
            task_configs.push(task_config);

            for j in node_idx_offset..node_idx_offset + num_nodes {
                let node = get_node(&datasets[i], j);
                // skip the node if target column was removed during preprocessing
                if node.col_name_idxs.iter().any(|&c| c == target) {
                    items.push(Item {
                        dataset_idx: i as i32,
                        node_idx: j,
                    });
                }
            }
        }

        let epoch = 0;
        Self {
            batch_size,
            seq_len,
            rank,
            world_size,
            datasets,
            items,
            max_bfs_width,
            epoch,
            d_text,
            seed,
            target_columns,
            columns_to_drop,
            task_configs,
        }
    }

    fn len_py(&self) -> PyResult<usize> {
        Ok(self.len())
    }

    fn batch_py<'a>(&self, py: Python<'a>, batch_idx: usize) -> PyResult<Vec<PyObject>> {
        self.batch(batch_idx).into_pyobject(py)
    }

    fn shuffle_py(&mut self, epoch: u64) {
        self.epoch = epoch;
        let mut rng = StdRng::seed_from_u64(epoch.wrapping_add(self.seed));
        self.items.shuffle(&mut rng);
    }
}

// STEER always knows exactly which config file a given run needs (it's
// resolved by the Python side from its own task registry), so this takes
// the path directly -- no SAMPLER_CONFIG_DIR env var, no per-(db_name,
// target_col_idx) filename lookup table.
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

fn resolve_task_config(config: SamplerTaskConfig) -> ResolvedSamplerTaskConfig {
    let column_filter = ResolvedColumnFilterConfig {
        very_strong: config.column_filter.very_strong.into_iter().collect(),
        strong: config.column_filter.strong.into_iter().collect(),
        moderate: config.column_filter.moderate.into_iter().collect(),
        weak: config.column_filter.weak.into_iter().collect(),
    };

    fn resolve_edge_tier(edges: Vec<String>) -> HashSet<(i32, i32)> {
        edges
            .into_iter()
            .map(|s| {
                let (src, dst) = s.split_once("->").unwrap_or_else(|| {
                    panic!("invalid edge key '{s}': expected 'src_id->dst_id'")
                });
                let src_id = src.parse::<i32>().unwrap_or_else(|e| {
                    panic!("invalid src id '{src}' in edge '{s}': {e}")
                });
                let dst_id = dst.parse::<i32>().unwrap_or_else(|e| {
                    panic!("invalid dst id '{dst}' in edge '{s}': {e}")
                });
                (src_id, dst_id)
            })
            .collect()
    }

    let p2f_edge_probabilistic_at_push = ResolvedEdgeTieredConfig {
        very_strong: resolve_edge_tier(config.p2f_edge_probabilistic_at_push.very_strong),
        strong: resolve_edge_tier(config.p2f_edge_probabilistic_at_push.strong),
        moderate: resolve_edge_tier(config.p2f_edge_probabilistic_at_push.moderate),
        weak: resolve_edge_tier(config.p2f_edge_probabilistic_at_push.weak),
    };
    let f2p_edge_probabilistic_at_push = ResolvedEdgeTieredConfig {
        very_strong: resolve_edge_tier(config.f2p_edge_probabilistic_at_push.very_strong),
        strong: resolve_edge_tier(config.f2p_edge_probabilistic_at_push.strong),
        moderate: resolve_edge_tier(config.f2p_edge_probabilistic_at_push.moderate),
        weak: resolve_edge_tier(config.f2p_edge_probabilistic_at_push.weak),
    };

    ResolvedSamplerTaskConfig {
        toggles: config.toggles,
        keep_probabilities: config.keep_probabilities,
        column_filter,
        p2f_edge_probabilistic_at_push,
        f2p_edge_probabilistic_at_push,
    }
}

fn column_filter_keep_probability(
    col_idx: i32,
    is_target: bool,
    is_task_table: bool,
    config: &ResolvedSamplerTaskConfig,
) -> f32 {
    if is_target || is_task_table {
        return 1.0;
    }
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

fn p2f_edge_probabilistic_at_push_keep_probability(
    src_table_idx: i32,
    dst_table_idx: i32,
    is_task_table: bool,
    config: &ResolvedSamplerTaskConfig,
) -> f32 {
    if is_task_table {
        return 1.0;
    }
    if !config.toggles.p2f_edge_probabilistic_at_push {
        return 1.0;
    }
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

fn f2p_edge_probabilistic_at_push_keep_probability(
    src_table_idx: i32,
    dst_table_idx: i32,
    is_task_table: bool,
    config: &ResolvedSamplerTaskConfig,
) -> f32 {
    if is_task_table {
        return 1.0;
    }
    if !config.toggles.f2p_edge_probabilistic_at_push {
        return 1.0;
    }
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

impl Sampler {
    fn len(&self) -> usize {
        self.items.len().div_ceil(self.batch_size * self.world_size)
    }

    fn batch(&self, batch_idx: usize) -> Vecs {
        let true_batch_size = self.batch_size.min(
            self.items.len()
                - self.rank * self.batch_size
                - batch_idx * self.batch_size * self.world_size,
        );

        let mut vecs = Vecs::new(self.batch_size, self.seq_len, true_batch_size, self.d_text);
        vecs.chunks_exact_mut(self.seq_len, self.d_text)
            .enumerate()
            .for_each(|(i, mut slices)| {
                let j =
                    batch_idx * self.batch_size * self.world_size + self.rank * self.batch_size + i;
                // when self.batch_size > true_batch_size, this will wrap around
                let j = j % self.items.len();
                let item = &self.items[j];
                let dataset_idx = item.dataset_idx as usize;

                self.seq(item, &mut slices, &self.task_configs[dataset_idx]);
            });
        vecs
    }

    fn seq(&self, item: &Item, slices: &mut Slices<'_>, task_config: &ResolvedSamplerTaskConfig) -> usize {
        let dataset = &self.datasets[item.dataset_idx as usize];
        let target_column = self.target_columns[item.dataset_idx as usize];
        let columns_to_drop = &self.columns_to_drop[item.dataset_idx as usize];
        let seed_node_idx = item.node_idx;

        let mut visited = vec![false; dataset.offsets.len() - 1];

        let mut f2p_ftr = vec![(0, seed_node_idx)];
        let seed_node = get_node(dataset, seed_node_idx);
        let mut p2f_ftr = Vec::<Vec<_>>::new();
        let mut seq_i = 0;
        let base_seed = self.epoch
            .wrapping_add(seed_node_idx as u64)
            .wrapping_add(self.seed);
        let mut rng = StdRng::seed_from_u64(base_seed);
        let mut coin_rng_p2f_edge_push = StdRng::seed_from_u64(base_seed.wrapping_add(5));
        let mut coin_rng_f2p_edge_push = StdRng::seed_from_u64(base_seed.wrapping_add(6));
        let mut coin_rng_column_filter = StdRng::seed_from_u64(base_seed.wrapping_add(7));
        loop {
            // select node
            let (depth, node_idx) = if !f2p_ftr.is_empty() {
                f2p_ftr.pop().unwrap()
            } else {
                let mut depth_choices = Vec::new();
                for (i, node) in p2f_ftr.iter().enumerate() {
                    if !node.is_empty() {
                        depth_choices.push(i);
                    }
                }
                if depth_choices.is_empty() {
                    return 0;
                } else {
                    let depth = depth_choices[0];
                    let r = rng.random_range(0..p2f_ftr[depth].len());
                    let l = p2f_ftr[depth].len();
                    p2f_ftr[depth].swap(r, l - 1);
                    let node_idx = p2f_ftr[depth].pop().unwrap();
                    (depth, node_idx)
                }
            };

            if visited[node_idx as usize] {
                continue;
            }
            let node = get_node(dataset, node_idx);
            visited[node_idx as usize] = true;

            for edge in node.f2p_edges.iter() {
                let parent_node_idx: i32 = edge.node_idx.into();
                let parent_node = get_node(dataset, parent_node_idx);
                let table_idx: i32 = parent_node.table_name_idx.into();
                let is_task_table = parent_node.table_name_idx == seed_node.table_name_idx;

                if task_config.toggles.f2p_edge_probabilistic_at_push {
                    let src_table_idx: i32 = node.table_name_idx.into();
                    let src_is_task_table = node.table_name_idx == seed_node.table_name_idx;
                    if !src_is_task_table {
                        let keep_prob = f2p_edge_probabilistic_at_push_keep_probability(
                            src_table_idx,
                            table_idx,
                            is_task_table,
                            task_config,
                        );
                        if keep_prob <= 0.0 || coin_rng_f2p_edge_push.random::<f32>() > keep_prob {
                            continue;
                        }
                    }
                }

                f2p_ftr.push((depth + 1, parent_node_idx));
            }

            let p2f_edges = get_p2f_edges(dataset, node_idx);

            // temporary storage for db edges to be subsampled
            let mut db_p2f_ftr: Vec<(i32, i32)> = Vec::new();

            for edge in p2f_edges.iter() {
                // include edges to task table only if seed node belongs to the task table
                if edge.table_name_idx != seed_node.table_name_idx && edge.table_type != ArchivedTableType::Db {
                    continue;
                }

                // temporal constraint
                if edge.timestamp.is_some() && seed_node.timestamp.is_some() && edge.timestamp > seed_node.timestamp {
                    continue;
                }

                let table_idx: i32 = edge.table_name_idx.into();
                let is_task_table = edge.table_type != ArchivedTableType::Db;

                if edge.table_type == ArchivedTableType::Db {
                    let child_node_idx: i32 = edge.node_idx.into();
                    db_p2f_ftr.push((child_node_idx, table_idx));
                    continue;
                }

                let child_node_idx: i32 = edge.node_idx.into();
                if task_config.toggles.p2f_edge_probabilistic_at_push {
                    let src_table_idx: i32 = node.table_name_idx.into();
                    let keep_prob = p2f_edge_probabilistic_at_push_keep_probability(
                        src_table_idx,
                        table_idx,
                        is_task_table,
                        task_config,
                    );
                    if keep_prob <= 0.0 || coin_rng_p2f_edge_push.random::<f32>() > keep_prob {
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

            // println!(
            //     "[STEP {} EXPAND] node_idx={} p2f_edges_total={} db_p2f_candidates={} f2p_len_now={}",
            //     step, node_idx, p2f_edges.len(), db_p2f_ftr.len(), f2p_ftr.len()
            // );

            let sampled_db_idxs = if db_p2f_ftr.len() > self.max_bfs_width {
                index::sample(&mut rng, db_p2f_ftr.len(), self.max_bfs_width).into_vec()
            } else {
                (0..db_p2f_ftr.len()).collect::<Vec<_>>()
            };

            // println!(
            //     "[STEP {} DB SAMPLE] kept_db_edges={} max_bfs_width={}",
            //     step, sampled_db_idxs.len(), self.max_bfs_width
            // );

            for idx in sampled_db_idxs {
                let child_node_idx = db_p2f_ftr[idx].0;
                let child_table_idx = db_p2f_ftr[idx].1;
                if task_config.toggles.p2f_edge_probabilistic_at_push {
                    let src_table_idx: i32 = node.table_name_idx.into();
                    let keep_prob = p2f_edge_probabilistic_at_push_keep_probability(
                        src_table_idx,
                        child_table_idx,
                        false,
                        task_config,
                    );
                    if keep_prob <= 0.0 || coin_rng_p2f_edge_push.random::<f32>() > keep_prob {
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

            let num_cells = node.col_name_idxs.len();
            for cell_i in 0..num_cells {
                let col_idx: i32 = node.col_name_idxs[cell_i].into();
                let is_target =
                    seed_node_idx == node.node_idx && node.col_name_idxs[cell_i] == target_column;
                let is_task_table = node.table_name_idx == seed_node.table_name_idx;

                if (node.node_idx == seed_node_idx && columns_to_drop.contains(&col_idx))
                    || (node.timestamp == seed_node.timestamp && columns_to_drop.contains(&col_idx))
                {
                    continue; // do not add this cell to the sequence
                }

                if task_config.toggles.column_filter {
                    let keep_prob = column_filter_keep_probability(col_idx, is_target, is_task_table, task_config);
                    if keep_prob <= 0.0 || coin_rng_column_filter.random::<f32>() > keep_prob {
                        continue;
                    }
                }

                slices.node_idxs[seq_i] = node.node_idx.into();

                assert!(node.f2p_nbr_idxs.len() <= MAX_F2P_NBRS);
                for (j, f2p_nbr_idx) in node.f2p_nbr_idxs.iter().enumerate() {
                    slices.f2p_nbr_idxs[seq_i * MAX_F2P_NBRS + j] = f2p_nbr_idx.into();
                }

                slices.table_name_idxs[seq_i] = node.table_name_idx.into();
                slices.col_name_idxs[seq_i] = node.col_name_idxs[cell_i].into();
                slices.class_value_idxs[seq_i] = node.class_value_idx[cell_i].into();
                slices.col_name_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]
                    .copy_from_slice(get_text_emb(
                        dataset,
                        slices.col_name_idxs[seq_i],
                        self.d_text,
                    ));

                let s = node.sem_types[cell_i].clone() as i32;
                slices.sem_types[seq_i] = s;

                slices.number_values[seq_i] = bf16::from_f32(node.number_values[cell_i].into());

                let text_idx: i32 = node.text_values[cell_i].into();
                slices.text_values[seq_i * self.d_text..(seq_i + 1) * self.d_text]
                    .copy_from_slice(get_text_emb(dataset, text_idx, self.d_text));

                slices.datetime_values[seq_i] = bf16::from_f32(node.datetime_values[cell_i].into());

                slices.boolean_values[seq_i] = bf16::from_f32(node.boolean_values[cell_i].into());

                slices.is_targets[seq_i] = is_target;

                slices.masks[seq_i] = is_target;

                slices.is_task_nodes[seq_i] =
                    node.is_task_node || (node.col_name_idxs[cell_i] == target_column);
                slices.is_padding[seq_i] = false;

                seq_i += 1;
                if seq_i >= self.seq_len {
                    break;
                }
            }
            if seq_i >= self.seq_len {
                break;
            }
        }
        seq_i
    }
}

fn get_node(dataset: &Dataset, idx: i32) -> &ArchivedNode {
    let l = dataset.offsets[idx as usize] as usize;
    let r = dataset.offsets[(idx + 1) as usize] as usize;
    let bytes = &dataset.mmap[l..r];
    // rkyv::access::<ArchivedNode, Error>(bytes).unwrap()
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

#[derive(Parser)]
pub struct Cli {
    #[arg(default_value = "rel-f1")]
    db_name: String,
    #[arg(default_value = "128")]
    batch_size: usize,
    #[arg(default_value = "1024")]
    seq_len: usize,
    #[arg(default_value = "1000")]
    num_trials: usize,
}

pub fn main(cli: Cli) {
    let tic = Instant::now();
    let sampler = Sampler::new(
        vec![(cli.db_name, 0, 10)], // dataset_tuples
        cli.batch_size,             // batch_size
        cli.seq_len,                // seq_len
        0,                          // rank
        1,                          // world_size
        256,                        // max_bfs_width
        "all-MiniLM-L12-v2",        // embedding_model
        384,                        // d_text
        0,                          // seed
        vec![-1; 1],                // target_columns
        vec![Vec::<i32>::new()],    // columns_to_drop
        vec![None],                 // sampler_config_paths
    );
    println!("Sampler loaded in {:?}", tic.elapsed());

    let mut sum = 0;
    let mut sum_sq = 0;
    let mut rng = rand::rng();
    for _ in 0..cli.num_trials {
        let tic = Instant::now();
        let batch_idx = rng.random_range(0..sampler.len());
        let _batch = sampler.batch(batch_idx);
        let elapsed = tic.elapsed().as_millis();
        sum += elapsed;
        sum_sq += elapsed * elapsed;
    }
    let mean = sum as f64 / cli.num_trials as f64;
    let std = (sum_sq as f64 / cli.num_trials as f64 - mean * mean).sqrt();
    println!("Mean: {} ms,\tStd: {} ms", mean, std);
}
