"""Core data shapes: tiered configs, sampling profiles, toggle modes.

Three approaches only: column_filter, p2f_edge, f2p_edge. Four tiers:
very_strong, strong, moderate, weak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

APPROACHES = ("column_filter", "p2f_edge", "f2p_edge")
TIERS = ("very_strong", "strong", "moderate", "weak")

# The toggle keys as the samplers actually spell them.
TOGGLE_KEYS = (
    "column_filter",
    "p2f_edge_probabilistic_at_push",
    "f2p_edge_probabilistic_at_push",
)

# approach name (as used in TieredConfig/Profile) -> toggle key
APPROACH_TOGGLE_KEY = {
    "column_filter": "column_filter",
    "p2f_edge": "p2f_edge_probabilistic_at_push",
    "f2p_edge": "f2p_edge_probabilistic_at_push",
}


def _empty_tiers() -> dict[str, list[str]]:
    return {tier: [] for tier in TIERS}


@dataclass
class TieredConfig:
    """Names-level config: which columns/edges are in which tier, per approach."""

    column_filter: dict[str, list[str]] = field(default_factory=_empty_tiers)
    p2f_edge: dict[str, list[str]] = field(default_factory=_empty_tiers)
    f2p_edge: dict[str, list[str]] = field(default_factory=_empty_tiers)

    def to_dict(self) -> dict:
        return {
            "column_filter": self.column_filter,
            "p2f_edge": self.p2f_edge,
            "f2p_edge": self.f2p_edge,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TieredConfig":
        return cls(
            column_filter={tier: list(d.get("column_filter", {}).get(tier, [])) for tier in TIERS},
            p2f_edge={tier: list(d.get("p2f_edge", {}).get(tier, [])) for tier in TIERS},
            f2p_edge={tier: list(d.get("f2p_edge", {}).get(tier, [])) for tier in TIERS},
        )


def _default_tier_probs() -> dict[str, float]:
    return {"very_strong": 0.0, "strong": 0.0, "moderate": 0.0, "weak": 0.0}


@dataclass
class Profile:
    """keep_probabilities: per-approach, per-tier keep probability."""

    column_filter: dict[str, float] = field(default_factory=_default_tier_probs)
    p2f_edge: dict[str, float] = field(default_factory=_default_tier_probs)
    f2p_edge: dict[str, float] = field(default_factory=_default_tier_probs)

    def to_dict(self) -> dict:
        return {
            "column_filter": self.column_filter,
            "p2f_edge": self.p2f_edge,
            "f2p_edge": self.f2p_edge,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(
            column_filter=dict(d.get("column_filter", _default_tier_probs())),
            p2f_edge=dict(d.get("p2f_edge", _default_tier_probs())),
            f2p_edge=dict(d.get("f2p_edge", _default_tier_probs())),
        )


@dataclass(frozen=True)
class ToggleMode:
    """A '+'-joined toggle-mode string, parsed and validated up front."""

    active: frozenset[str]

    @classmethod
    def parse(cls, mode: str) -> "ToggleMode":
        if mode == "baseline":
            return cls(frozenset())
        keys = frozenset(m.strip() for m in mode.split("+") if m.strip())
        unknown = keys - set(TOGGLE_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown toggle_mode key(s) {sorted(unknown)} in '{mode}'. "
                f"Known keys: {TOGGLE_KEYS}"
            )
        return cls(keys)

    def is_active(self, approach: str) -> bool:
        return APPROACH_TOGGLE_KEY[approach] in self.active

    def toggles_dict(self) -> dict[str, bool]:
        return {key: key in self.active for key in TOGGLE_KEYS}
