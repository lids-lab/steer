"""LLM config generation: one call per approach (column_filter, p2f_edge,
f2p_edge), each following the same step-by-step scaffold (inventory -> task
understanding -> abstain check -> explicit enumeration -> tiering rubric ->
sanity check). Provider-agnostic: any object with a `generate(prompt)` method
works (see providers/).
"""

from __future__ import annotations

import json
from typing import Any

from .providers.base import Provider
from .schema import APPROACHES as _CONFIG_APPROACHES  # column_filter, p2f_edge, f2p_edge
from .schema import TieredConfig

# The approach names as used in the LLM prompt/output (edge approaches carry
# the sampler's own toggle-key suffix in their prompt identity, matching
# what generate_policy_config.py has emitted since run 016).
_PROMPT_APPROACH_KEY = {
    "column_filter": "column_filter",
    "p2f_edge": "p2f_edge_probabilistic_at_push",
    "f2p_edge": "f2p_edge_probabilistic_at_push",
}

_TIERS = ("very_strong", "strong", "moderate", "weak")

# Columns present in every model's actual preprocessed runtime data (RT,
# rt-j, and Griffin all retain them) that the currently-installed live
# relbench no longer exposes via get_db() -- confirmed against
# stanford-star/relbench-v2-extra commit 2d0c6d19 ("rel-ratebeer: drop
# NaN/leakage/artifact columns (#373)"). Several of these (the per-user/
# per-beer/per-place lifetime aggregates) are flagged by that issue as
# temporally leaky -- computed without a time cutoff. They are restored
# here rather than stripped: the runtime checkpoints were trained on data
# that always included them, so hiding them from the LLM doesn't remove
# the leakage the model already relies on, it only stops the sampler from
# being able to choose to keep them. Restoring them lets the LLM reason
# over the same schema the model actually sees at inference time.
_SCHEMA_ADDITIONS: dict[str, dict[str, list[str]]] = {
    "rel-ratebeer": {
        "beer_ratings": ["updated_at"],
        "beers": [
            "avg_rating", "last_9m_avg", "last_9m_count", "overall_percentile",
            "rating_count", "rating_std_dev", "real_avg_rating", "straight_avg_rating",
            "straight_rating_count", "style_percentile", "updated_at", "view_count",
            "year4_avg", "year4_count", "year4_overall", "year4_style",
        ],
        "brewers": ["msrepl_tran_version", "score", "updated_at", "view_count"],
        "place_ratings": ["updated_at"],
        "places": [
            "avg_rating", "bay_mean", "percentile", "rating_count", "rating_text",
            "score", "updated_at", "valid_rating_count", "weighted_avg",
        ],
        "states": ["msrepl_tran_version"],
        "users": [
            "avg_beer_rating", "avg_place_rating", "beer_first_rating", "beer_last_rating",
            "beer_rating_count", "favorite_count", "max_beer_rating", "max_place_rating",
            "min_beer_rating", "min_place_rating", "place_first_rating", "place_last_rating",
            "place_rating_count", "total_activity_count", "updated_at",
        ],
    },
}

_SECTION_TEMPLATES: dict[str, Any] = {
    "column_filter": {
        "very_strong": ["table_a.column_x"],
        "strong": ["table_b.column_y"],
        "moderate": ["table_c.column_z"],
        "weak": ["table_c.column_w"],
    },
    "p2f_edge_probabilistic_at_push": {
        "very_strong": ["source_a->destination_a:fk_col_x"],
        "strong": ["source_b->destination_b:fk_col_y"],
        "moderate": ["source_c->destination_c:fk_col_z"],
        "weak": ["source_d->destination_d:fk_col_w"],
    },
    "f2p_edge_probabilistic_at_push": {
        "very_strong": ["source_a->destination_a:fk_col_x"],
        "strong": ["source_b->destination_b:fk_col_y"],
        "moderate": ["source_c->destination_c:fk_col_z"],
        "weak": ["source_d->destination_d:fk_col_w"],
    },
}


def build_task_input(dataset: str, task_table: str) -> dict[str, Any]:
    """Schema description an LLM call needs: task_name, dataset,
    task_table_name, task_table_columns, database_definition. Built directly
    from relbench -- no dependency on any preprocessed cache."""
    from relbench.load import load_dataset

    db_dataset = load_dataset(dataset)
    db = db_dataset.get_db(upto_test_timestamp=False)
    task = db_dataset.load_task(task_table)
    task_table_columns = list(task.get_table("train").df.columns)

    additions = _SCHEMA_ADDITIONS.get(dataset, {})

    database_definition = []
    for table_name, table in sorted(db.table_dict.items()):
        fkey_map = getattr(table, "fkey_col_to_pkey_table", {}) or {}
        columns = list(table.df.columns)
        for extra_col in additions.get(table_name, []):
            if extra_col not in columns:
                columns.append(extra_col)
        database_definition.append({
            "name": table_name,
            "columns": columns,
            "primary_key": getattr(table, "pkey_col", None),
            "foreign_keys": [
                {"column": col, "references_table": ref_table}
                for col, ref_table in sorted(fkey_map.items())
            ],
        })

    return {
        "task_name": f"{dataset.replace('-', '_')}_{task_table.replace('-', '_')}",
        "dataset": dataset,
        "task_table_name": task_table,
        "task_table_columns": task_table_columns,
        "database_definition": database_definition,
    }


def _render_database_definition(database_definition: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for table in database_definition:
        lines.append(f"Table: {table['name']}")
        lines.append(f"Columns: {', '.join(table['columns'])}")
        lines.append(f"Primary key: {table['primary_key']}")
        fks = table.get("foreign_keys", [])
        if fks:
            lines.append("Foreign keys:")
            for fk in fks:
                lines.append(f"- {fk['column']} -> {fk['references_table']}")
        else:
            lines.append("Foreign keys: none")
        lines.append("")
    return "\n".join(lines).strip()


def _build_prompt(task_input: dict[str, Any], prompt_approach: str) -> str:
    database_block = _render_database_definition(task_input["database_definition"])
    section_format = json.dumps(_SECTION_TEMPLATES[prompt_approach], indent=2)
    task_context = f"""Task name: {task_input["task_name"]}
Dataset: {task_input["dataset"]}
Task table: {task_input["task_table_name"]}
Task table columns: {", ".join(task_input["task_table_columns"])}

Database definition:
{database_block}"""

    if prompt_approach == "column_filter":
        return f"""You are an expert in relational databases and machine learning on tabular data.

Your job is to rank columns within each table into tiers for a context filtering policy applied during sequence writing in a BFS-like relational sampler.
In a relational database sampler, once a row is selected for inclusion in the context, its columns are written as tokens into the input sequence. Not all columns are equally useful for prediction — some carry direct signal toward the target while others add noise or waste sequence budget. Your goal is to rank each column into a tier based on how useful its values are for predicting the target. Columns in lower tiers will be written with lower probability or excluded entirely, preserving sequence budget for higher-signal content.
Your ranking should be based purely on semantic reasoning: what each column represents, what kind of values it holds, and how directly those values relate to the prediction target and the entity being predicted for.
Beyond a column's direct predictive value, you must also consider an additional factor that takes priority over predictive value for certain columns: the sampler that consumes this schema builds its input by traversing and sampling rows from the database, and the model may rely on certain columns to recognize that multiple sampled rows describe the same real-world entity across time. A column that serves as a stable, largely unique identifier for the entity being predicted for — such as a name, reference code, or similar per-entity label — has significant value for linking that entity's own records together, and this entity-linking value must be treated as at least as important as direct predictive value, even when the column's raw value has no causal or predictive relationship to the target. Do not tier such a column as weak based on predictive value alone.
Follow these steps in order:
Step 1 - Understand the database structure:

Read through all the tables and their columns. For each table, form a clear understanding of what real-world entity or concept that table represents and what kind of information each column holds. Identify primary keys and foreign keys and understand which columns are identifiers versus content columns.
Step 2 - Understand the task:

Identify the entity being predicted for, the target value being predicted, and what the prediction task is asking you to estimate or classify.
Step 3 - Assess semantic interpretability:

Before proceeding to ranking, assess whether the schema contains enough semantic information to reason about column relevance. If the majority of table names and column names are opaque — for example generic codes, hashed identifiers, numeric labels, or names that carry no meaningful description of what the data represents — then semantic reasoning is not possible and you must abstain. If the schema is interpretable enough to reason about, proceed to step 4.
Step 4 - Rank feature columns by predictive signal and entity-linking value:

Go through each feature column in each table and reason about how useful its values are for predicting the target and characterizing the entity being predicted for, as well as whether it carries entity-linking value as described above. Only feature columns are ranked — primary key and foreign key columns, as identified in the database definition, are excluded from ranking entirely as they are used only for graph traversal and carry no content signal. Rank all feature columns into the following tiers: very_strong, strong, moderate, weak.
Use the following criteria:

very_strong: this column directly records an outcome, measurement, status, or label that is immediately and causally relevant to the prediction target. Removing it would significantly hurt prediction quality.
strong: this column provides important contextual information closely related to the target even if not directly causal. It characterizes the prediction entity or its environment in a meaningful way.
moderate: this column has some plausible relationship to the target but the connection requires more than one step of reasoning or is indirect. It may provide weak signal in some cases.
weak: this column provides little to no predictive signal toward the target given the specific prediction task.

Entity-linking override: if a feature column is a stable, largely unique identifier for the entity being predicted for (a name, ID-like code, or similar per-entity label, on that entity's own table), rank it at least moderate — strong or very_strong if it is likely the primary way the model would recognize that specific entity — regardless of whether its raw value has direct predictive value. This entity-linking criterion takes priority over the predictive-value criteria above for such columns.

Step 5 - Ensure there is at least one non-weak column in your output:

That means that if you sum up the number of columns in very_strong, strong, and moderate tiers they should be greater than or equal to 1. If the sum is 0, go back to step 4 and select the single most relevant feature column across all tables and add it to the very_strong tier.
Hard constraints that always apply regardless of the steps above:

Only feature columns appear in your output. Do not include primary key or foreign key columns in any tier.
Every feature column from every table must appear in exactly one tier. Do not omit any feature column and do not place the same column in multiple tiers.
Use the exact table names and column names from the database definition. Do not invent columns.
Each entry in your output must follow the format table_name.column_name to avoid ambiguity across tables.

If you determined in step 3 that the schema is not semantically interpretable, return only this exact JSON and nothing else:
{{"abstain": true, "reasoning": "..."}}
Otherwise return only a valid JSON object in this exact format, with no explanation, no markdown, and nothing else:
{{
  "config": {section_format},
  "reasoning": {{
    "step_1": "...",
    "step_2": "...",
    "step_3": "...",
    "step_4": "...",
    "step_5": "..."
  }}
}}

{task_context}
"""
    if prompt_approach == "p2f_edge_probabilistic_at_push":
        return f"""You are an expert in relational databases and machine learning on tabular data.

Your job is to rank traversal edges into tiers for a probabilistic BFS-like primary-to-foreign-key traversal policy based on semantic reasoning about the database schema.
In a relational database, a primary-to-foreign-key relationship means that a table with a primary key is referenced by other tables via foreign keys. Those referencing tables are the child tables. When the sampler is at a row in any table, it can expand outward into child tables by following those foreign key references. Each such expansion is a traversal edge — a directed connection from the current parent table to a specific child table via a foreign key. Your goal is to rank each traversal edge into tiers based on how useful it is to follow that specific edge during sampling. The coin toss happens at the edge level when a node is being pushed into the frontier, so rejected edges are never enqueued at all.
Your ranking should be based purely on semantic reasoning: what each table represents, what its columns describe, how directly that information relates to the prediction target and the entity being predicted for, and crucially, what the source table context is when the edge is crossed.
Follow these steps in order:
Step 1 - Understand the database structure:

Read through all the tables and their columns. For each table, form a clear understanding of what real-world entity or concept that table represents and what kind of information each column holds. Identify all primary keys and foreign keys and understand how the tables are connected to each other.
Step 2 - Understand the task:

Identify the entity being predicted for, the target value being predicted, and what the prediction task is asking you to estimate or classify.
Step 3 - Assess semantic interpretability:

Before proceeding to ranking, assess whether the schema contains enough semantic information to reason about edge relevance. If the majority of table names and column names are opaque — for example generic codes, hashed identifiers, numeric labels, or names that carry no meaningful description of what the data represents — then semantic reasoning is not possible and you must abstain. If the schema is interpretable enough to reason about, proceed to step 4.
Step 4 - Build the edge list:

Go through each table in the database and identify which tables have at least one foreign key defined. For each foreign key column in each such table, identify its parent table — the table whose primary key is referenced by that foreign key column. Write out all traversal edges as source->destination:fk_col triples, where source is the parent table (where the sampler currently is), destination is the child table (where the sampler would traverse to), and fk_col is the exact foreign key column name defined on the destination table. Note that the arrow in source->destination represents the direction the sampler travels, not the direction of the foreign key. The foreign key is always defined on the destination table pointing back to the source. Each foreign key column in the database produces exactly one edge. If a destination table has multiple foreign keys pointing to the same source, each FK column produces a separate edge with a distinct fk_col suffix. Tables that have no foreign keys produce no edges as destinations. List every edge explicitly in source->destination:fk_col format before moving on.
Step 5 - Rank edges by semantic relevance:

Go through each edge in your edge list and reason about how useful it is to follow that specific edge during sampling. The key question for each edge is: given that the sampler is currently at a row in the source table, does crossing this edge to the destination table provide meaningful signal toward the prediction target and the entity being predicted for? Also consider what subset of the destination table's rows you would actually reach by following this edge from this specific source. A more targeted subset that directly characterizes the prediction entity is more valuable than a broad or noisy one.
Two edges that share the same destination table can have very different values if their source tables provide different context. Reason about each edge independently based on its specific source-destination combination.
Rank all edges into the following tiers: very_strong, strong, moderate, weak.
Use the following criteria:

very_strong: following this edge from this specific source will almost always lead to rows that directly describe or characterize the prediction entity or its outcomes in a way that is immediately relevant to the target.
strong: following this edge provides important contextual signal closely related to the target, even if not directly causal, given the source context.
moderate: following this edge has some plausible relationship to the target but the connection requires more than one step of reasoning or is indirect given the source context.
weak: following this edge leads to rows that describe entities or events with no plausible direct or indirect connection to the prediction target given where the sampler currently is.

Step 6 - Ensure there is at least one edge in your output:

That means that if you sum up the number of edges in all the tiers they should be greater than or equal to 1. If the sum is 0, go back to step 5 and select the single most relevant edge and add it to the very_strong tier.
Hard constraints that always apply regardless of the steps above:

Only use source->destination:fk_col triples that appear in your edge list from step 4. Do not invent edges.
Use the exact table names and column names from the database definition.
Each edge must appear in exactly one tier. Do not place the same edge in multiple tiers.

If you determined in step 3 that the schema is not semantically interpretable, return only this exact JSON and nothing else:
{{"abstain": true, "reasoning": "..."}}
Otherwise return only a valid JSON object in this exact format, with no explanation, no markdown, and nothing else:
{{
  "config": {section_format},
  "reasoning": {{
    "step_1": "...",
    "step_2": "...",
    "step_3": "...",
    "step_4": "...",
    "step_5": "...",
    "step_6": "..."
  }}
}}

{task_context}
"""
    if prompt_approach == "f2p_edge_probabilistic_at_push":
        return f"""You are an expert in relational databases and machine learning on tabular data.

Your job is to rank traversal edges into tiers for a probabilistic BFS-like foreign-to-primary-key traversal policy based on semantic reasoning about the database schema.
In a relational database, a foreign-to-primary-key relationship means that a table holds a foreign key that references a primary key in another table. That referenced table is the parent table. When the sampler is at a row in any table, it can expand outward into parent tables by following those foreign key references. Each such expansion is a traversal edge — a directed connection from the current child table to a specific parent table via a foreign key column. Your goal is to rank each traversal edge into tiers based on how useful it is to follow that specific edge during sampling. The coin toss happens at the edge level when a node is being pushed into the frontier, so rejected edges are never enqueued at all.
Your ranking should be based purely on semantic reasoning: what each table represents, what its columns describe, how directly that information relates to the prediction target and the entity being predicted for, and crucially, what the source table context is when the edge is crossed.
Follow these steps in order:
Step 1 - Understand the database structure:

Read through all the tables and their columns. For each table, form a clear understanding of what real-world entity or concept that table represents and what kind of information each column holds. Identify all primary keys and foreign keys and understand how the tables are connected to each other.
Step 2 - Understand the task:

Identify the entity being predicted for, the target value being predicted, and what the prediction task is asking you to estimate or classify.
Step 3 - Assess semantic interpretability:

Before proceeding to ranking, assess whether the schema contains enough semantic information to reason about edge relevance. If the majority of table names and column names are opaque — for example generic codes, hashed identifiers, numeric labels, or names that carry no meaningful description of what the data represents — then semantic reasoning is not possible and you must abstain. If the schema is interpretable enough to reason about, proceed to step 4.
Step 4 - Build the edge list:

Go through each table in the database that has at least one foreign key defined. For each foreign key column in each such table, identify the parent table it points to. Write out all traversal edges as source->destination:fk_col triples, where source is the child table (where the sampler currently is), destination is the parent table (where the sampler would traverse to), and fk_col is the exact foreign key column name defined on the source table. Note that the arrow in source->destination represents the direction the sampler travels. In f2p traversal the foreign key is defined on the source table and points to the destination table. Each foreign key column in the database produces exactly one edge. If a source table has multiple foreign keys pointing to the same destination, each FK column produces a separate edge with a distinct fk_col suffix. Note that some tables may have a foreign key that references their own primary key. These self-referential edges are valid and should be included in your edge list and ranked like any other edge. List every edge explicitly in source->destination:fk_col format before moving on.
Step 5 - Rank edges by semantic relevance:

Go through each edge in your edge list and reason about how useful it is to follow that specific edge during sampling. The key question for each edge is: given that the sampler is currently at a row in the source table, does crossing this edge to the destination table provide meaningful signal toward the prediction target and the entity being predicted for? Also consider what specific parent entity you would reach by following this edge from this specific source. A parent row that directly identifies or characterizes the prediction entity is more valuable than one that only provides generic or loosely related context.
Two edges that share the same destination table can have very different values if their source tables provide different context. Reason about each edge independently based on its specific source-destination combination.
Rank all edges into the following tiers: very_strong, strong, moderate, weak.
Use the following criteria:

very_strong: following this edge from this specific source will almost always lead to rows that directly describe or characterize the prediction entity or its outcomes in a way that is immediately relevant to the target.
strong: following this edge provides important contextual signal closely related to the target, even if not directly causal, given the source context.
moderate: following this edge has some plausible relationship to the target but the connection requires more than one step of reasoning or is indirect given the source context.
weak: following this edge leads to rows that describe entities or events with no plausible direct or indirect connection to the prediction target given where the sampler currently is.

Step 6 - Ensure there is at least one edge in your output:

That means that if you sum up the number of edges in all the tiers they should be greater than or equal to 1. If the sum is 0, go back to step 5 and select the single most relevant edge and add it to the very_strong tier.
Hard constraints that always apply regardless of the steps above:

Only use source->destination:fk_col triples that appear in your edge list from step 4. Do not invent edges.
Use the exact table names and column names from the database definition.
Each edge must appear in exactly one tier. Do not place the same edge in multiple tiers.

If you determined in step 3 that the schema is not semantically interpretable, return only this exact JSON and nothing else:
{{"abstain": true, "reasoning": "..."}}
Otherwise return only a valid JSON object in this exact format, with no explanation, no markdown, and nothing else:
{{
  "config": {section_format},
  "reasoning": {{
    "step_1": "...",
    "step_2": "...",
    "step_3": "...",
    "step_4": "...",
    "step_5": "...",
    "step_6": "..."
  }}
}}

{task_context}
"""
    raise ValueError(f"Unknown approach '{prompt_approach}'")


def _allowed_tables(task_input: dict[str, Any]) -> dict[str, set[str]]:
    tables = {table["name"]: set(table["columns"]) for table in task_input["database_definition"]}
    tables[task_input["task_table_name"]] = set(task_input["task_table_columns"])
    return tables


def _validate_column_section(section: Any, task_input: dict[str, Any]) -> dict[str, list[str]]:
    if not isinstance(section, dict):
        raise ValueError("column_filter output must be a JSON object")
    unknown = set(section) - set(_TIERS)
    if unknown:
        raise ValueError(f"Unknown tier key(s) in column_filter: {sorted(unknown)}")

    tables = _allowed_tables(task_input)
    validated: dict[str, list[str]] = {t: [] for t in _TIERS}
    seen: dict[str, str] = {}
    for tier in _TIERS:
        for entry in section.get(tier, []):
            if not isinstance(entry, str) or "." not in entry:
                raise ValueError(f"Invalid entry '{entry}' in column_filter[{tier}]: must be 'table_name.column_name'")
            table_name, column_name = entry.split(".", 1)
            if table_name not in tables:
                raise ValueError(f"Unknown table '{table_name}' in column_filter entry '{entry}'")
            if column_name not in tables[table_name]:
                raise ValueError(f"Unknown column '{column_name}' in table '{table_name}' for entry '{entry}'")
            if entry in seen:
                continue  # keep the first (highest) tier seen
            seen[entry] = tier
            validated[tier].append(entry)
    return validated


def _validate_edge_section(section: Any, prompt_approach: str, tables: dict[str, set[str]]) -> dict[str, list[str]]:
    if not isinstance(section, dict):
        raise ValueError(f"{prompt_approach} output must be a JSON object")
    if "reasoning" in section:
        section = {k: v for k, v in section.items() if k != "reasoning"}
    unknown = set(section) - set(_TIERS)
    if unknown:
        raise ValueError(f"Unknown tier key(s) in {prompt_approach}: {sorted(unknown)}")

    is_f2p = "f2p" in prompt_approach
    validated: dict[str, list[str]] = {t: [] for t in _TIERS}
    seen: dict[str, str] = {}
    for tier in _TIERS:
        for edge in section.get(tier, []):
            if not isinstance(edge, str) or "->" not in edge or ":" not in edge.split("->", 1)[1]:
                raise ValueError(f"Invalid edge '{edge}' in {prompt_approach}[{tier}]: must be 'source->destination:fk_col'")
            src, rest = edge.split("->", 1)
            dst, fk_col = rest.split(":", 1)
            if src not in tables:
                raise ValueError(f"Unknown source table '{src}' in edge '{edge}'")
            if dst not in tables:
                raise ValueError(f"Unknown destination table '{dst}' in edge '{edge}'")
            fk_table = src if is_f2p else dst
            if fk_col not in tables[fk_table]:
                print(f"Warning: FK column '{fk_col}' not found in table '{fk_table}' for edge '{edge}'; accepting anyway.")
            if edge in seen:
                continue
            seen[edge] = tier
            validated[tier].append(edge)
    return validated


def _strip_markdown_fence(text: str) -> str:
    """Defensive only -- the prompt already asks for no markdown, and every
    reasoning_effort='high' generation run to date has complied. Without high
    reasoning effort, instruction-following on output format got noticeably
    less reliable (some responses come back fenced), so this protects that
    case without changing behavior when the model already returns bare JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _extract_section(payload: Any) -> Any:
    """Returns None if the model abstained (caller should keep the section empty)."""
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object")
    if payload.get("abstain") is True:
        print(f"  -> LLM abstained: {payload.get('reasoning', '')}")
        return None
    return payload.get("config", payload)


def generate_config(
    task_input: dict[str, Any],
    provider: Provider,
    *,
    reasoning_effort: str | None = None,
    approaches: tuple[str, ...] = _CONFIG_APPROACHES,
    existing_config: TieredConfig | None = None,
) -> TieredConfig:
    """One LLM call per approach in `approaches`, validated, merged into a TieredConfig.

    `approaches` narrows which sections actually get a fresh LLM call -- e.g.
    `("column_filter",)` to update only that section. Any approach left out
    is instead carried over unchanged from `existing_config` (required in
    that case). Not meant as a general per-approach regeneration mode --
    used specifically when re-running the LLM for one approach only, without
    disturbing already-tuned edge tiers for the others.
    """
    tables = _allowed_tables(task_input)
    sections: dict[str, dict[str, list[str]]] = {}

    for approach in _CONFIG_APPROACHES:
        if approach not in approaches:
            if existing_config is None:
                raise ValueError(f"approach '{approach}' excluded from `approaches` but no existing_config given to carry it over from")
            sections[approach] = getattr(existing_config, approach)
            continue
        prompt_approach = _PROMPT_APPROACH_KEY[approach]
        prompt = _build_prompt(task_input, prompt_approach)
        output_text = provider.generate(prompt, reasoning_effort=reasoning_effort)
        payload = json.loads(_strip_markdown_fence(output_text))
        section = _extract_section(payload)

        if section is None:
            sections[approach] = {t: [] for t in _TIERS}
        elif approach == "column_filter":
            sections[approach] = _validate_column_section(section, task_input)
        else:
            sections[approach] = _validate_edge_section(section, prompt_approach, tables)

    return TieredConfig(
        column_filter=sections["column_filter"],
        p2f_edge=sections["p2f_edge"],
        f2p_edge=sections["f2p_edge"],
    )
