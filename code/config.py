from dataclasses import dataclass


CONFIG_VERSION = "2026-11-06.v1"


@dataclass(frozen=True)
class SelectionThresholds:
    budget_fraction: float = 0.145
    margin_activity_scale: float = 0.10
    margin_low: float = 0.005
    margin_high: float = 0.045
    change_low: float = 0.012
    fallback_activity_threshold: float = 0.015
    novelty_weight: float = 0.8
    labeled_bonus: float = 0.08
    redundancy_penalty: float = 0.22


@dataclass(frozen=True)
class LocalizationThresholds:
    margin_low: float = 0.005
    margin_high: float = 0.045
    change_low: float = 0.012
    topk_delta: float = 0.025


@dataclass(frozen=True)
class RoutingThresholds:
    kitchen_margin_min: float = 0.03
    kitchen_fraction_min: float = 0.5


@dataclass(frozen=True)
class TemporalSelectionThresholds:
    budget_fraction: float = 0.145
    novelty_weight: float = 0.8
    redundancy_penalty: float = 0.22
    retained_pair_min_fraction: float = 0.5
    generic_score_temperature: float = 20.0


SELECTION = SelectionThresholds()
LOCALIZATION = LocalizationThresholds()
ROUTING = RoutingThresholds()
TEMPORAL_SELECTION = TemporalSelectionThresholds()
