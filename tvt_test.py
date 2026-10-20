#!/usr/bin/env python3
"""TVT connectivity control: goal-conditioned DDQN baselines and the E2E
Transformer latent-goal + DDQN architecture.

Architectures (--architecture):
  goal-ddqn       rule/fixed/agentic planner -> Goal one-hot -> DDQN (baseline)
  goal-free-ddqn  base observation -> DDQN, no planner and no Goal (ablation)
  e2e-transformer context history -> Transformer -> latent z -> DDQN (proposed)

Timing policy (E2E_TIMING_POLICY): replay training runs after the action has
been sent, so it stays outside the inference latency measurement interval
(infer_start_wall_s .. action_sent_wall_s). The ns3-ai handshake is
lock-step, so this separates the measurement only: training can still
extend total simulation wall-clock progression and must never be described
as running in parallel with ns-3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tvt_config import (
    ACTION_DIM,
    ACTION_NAMES,
    ACTION_SEMANTICS,
    DUPLICATION_LEVELS,
    duplication_ratio,
    ARCHITECTURE_VERSION,
    ARCHITECTURES,
    CHECKPOINT_SCHEMA_VERSION,
    CONTEXT_FEATURE_DIM,
    DEFAULT_E2E_REWARD_VERSION,
    DEFAULT_GOAL_PLANNER,
    DEFAULT_GRAD_CLIP,
    DEFAULT_SERVICE_CONTEXT_MODE,
    DEFAULT_SERVICE_FLOW_SCENARIO,
    DEFAULT_TRANSFORMER,
    E2E_ARCHITECTURE,
    E2E_TIMING_POLICY,
    FIXED_GOAL_WEIGHTS,
    GOAL_DIM,
    GOAL_PLANNERS,
    GOAL_WEIGHT_NAMES,
    HISTORY_LEN,
    LOWER_INPUT_DIM,
    LOWER_REWARD_VERSION,
    NORMALIZATION,
    OBSERVATION_CONTRACT_VERSION,
    EPISODE_RNG_POLICY,
    EPISODE_RNG_STRIDE,
    FIRST_EPISODE_INDEX,
    SCENARIO_ID,
    episode_rng_run,
    PENDING_MAX_AGE_S,
    PENDING_MAX_ENTRIES,
    QOS_THRESHOLDS,
    RADIO_STATE_DIM,
    SEMANTIC_CONFIG_VERSION,
    SERVICE_CONTEXT_DIM,
    SERVICE_CONTEXT_MODES,
    UPPER_OBJECTIVE_VERSION,
    SERVICE_AGGREGATION_MODE,
    SERVICE_AGGREGATION_MODES,
    MACRO_HORIZON,
    MACRO_RETURN_CONVENTION,
    GOAL_INTERVAL_MS,
    DECISION_INTERVAL_MS,
    RESOURCE_CONSTRAINT_VERSION,
    DUPLICATION_BUDGET,
    FORMAL_GOAL_PLANNERS,
    RESOURCE_COST_VERSION,
    LEGACY_TASK_REWARD_VERSION,
    ServiceType,
    TransformerConfig,
    active_services_for_time,
    parse_service_flow_scenario,
    qos_satisfied_multi,
    serializable_configuration,
    service_for_time,
    ACTION_HOLD_ENABLED,
    ACTION_HOLD_STATUS,
    ACTION_TRANSITION_SEMANTICS,
    PHYSICAL_SWITCHING_COST_MODELED,
    MACRO_INTERVAL_BASIS,
    MACRO_TERMINAL_CONVENTION,
    RELIABILITY_ATTRIBUTION_VERSION,
    GOAL_EXPLORATION_VERSION,
    EXPLORATION_PROTOCOL_VERSION,
    GOAL_EXPLORATION_MECHANISM,
    BETA_SELECTION_TYPE,
    BETA_COVERAGE_LEVELS,
    COVERAGE_PER_BETA,
    PLANNER_DOMINANT_EXPLORATION_POLICY_MAX_BETA,
    SCREENING_PROTOCOL_VERSION,
    beta_from_coverage,
    coverage_from_beta,
    exploration_radius_bracket,
    TWO_TIMESCALE_VERSION,
    SYSTEM_AGGREGATION_MODE,
    RUN_MODES,
    DEFAULT_RUN_MODE,
    UNRESOLVED_PROTOCOL_PARAMETERS,
    ProtocolNotFrozenError,
    InvocationBudgetError,
    validate_invocation_budget,
    SCREENING_SEEDS,
    protocol_freeze_report,
    require_frozen_protocol,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY_MODE,
    EPSILON_DECAY_MODES,
    EPSILON_DECAY_RULE,
    EPSILON_SCHEDULE_VERSION,
    FORMAL_EPISODE_LENGTH_S,
    FORMAL_TRAINING_HORIZON,
    C_SLICE,
    V_SLICE,
    N_SLICE,
    EPSILON_ANNEALING_STEPS,
    EPSILON_EXTENSION_POLICY,
    frozen_epsilon_annealing_steps,
    frozen_epsilon_decay,
    FORMAL_CRITIC_WARMUP_W,
    FORMAL_SERVICE_FLOW_SCENARIO,
    FORMAL_MACROS_PER_EPISODE,
    TRAINING_HORIZON_STATUS,
    HORIZON_SENSITIVITY_ARM,
    BINDING_TIE_LABEL,
    DUPLICATION_DUAL_CADENCE,
    DUPLICATION_PARTIAL_WINDOW_POLICY,
    DUPLICATION_VIOLATION_NORMALIZED,
    formal_action_count,
    epsilon_decay_steps,
    derived_epsilon_decay,
)
from tvt_e2e import (
    E2E_GRAD_CLIP,
    LOWER_HIDDEN_DIM,
    ConnectivityAgent,
    ContextHistory,
    MetricTracker,
    PendingPool,
    Transition,
    build_context_frame,
    build_radio_state,
    mobility_context_raw,
    compute_reward,
    MacroTransition,
    TickCohort,
    read_tick_cohort,
    cohort_upper_objective,
    duplication_rate,
    goal_statistics,
    attempted_copies_per_logical,
    successful_copies_per_logical,
    duplication_overhead,
    validate_goal,
)
from tvt_logging import ExperimentLogger


# Default name, preserved exactly so existing sequential invocations are
# bit-for-bit backward compatible. Overridable per run via --shm-segment so
# that concurrent isolated runs cannot attach to one another's segment.
# INFRASTRUCTURE ONLY: the segment NAME has no effect on the observation,
# action, reward, architecture, RNG streams, seed mapping, scenario, timing,
# learning, packet accounting, or the shared-memory struct layout.
SHM_SEG_DEFAULT = "TvtAiSeg"
SHM_SEG = SHM_SEG_DEFAULT


def sanitize_shm_segment(name: str) -> str:
    """Boost.Interprocess accepts a filesystem-like name; keep it portable.

    Only [A-Za-z0-9._-] survives; anything else becomes '_'. Raises rather
    than silently truncating, because two runs whose names collided after
    truncation would corrupt each other -- exactly what this exists to prevent.
    """
    import re as _re
    cleaned = _re.sub(r"[^A-Za-z0-9._-]", "_", str(name))
    if not cleaned:
        raise ValueError("--shm-segment must contain at least one usable char")
    if len(cleaned) > 200:
        raise ValueError(
            f"--shm-segment too long after sanitising ({len(cleaned)} > 200); "
            "shorten it rather than risking a post-truncation collision")
    return cleaned
SHM_ENV_NAME = "TvtAiEnv"
SHM_ACT_NAME = "TvtAiAct"
SHM_SYNC_NAME = "TvtAiSync"
SHM_SIZE = 65536

ALGORITHM = "ddqn"
HIDDEN_DIM = 64
LR = 1e-3
GAMMA = 0.99
# EPSILON_START / EPSILON_MIN now live in tvt_config (single source of truth).
# There is deliberately no EPSILON_DECAY constant: in the formal protocol the
# decay is DERIVED from the frozen training budget, so an inherited constant
# cannot silently outlive the budget it was calibrated for.
BATCH_SIZE = 64
MEMORY_CAPACITY = 20_000
TARGET_UPDATE = 200
DEFAULT_MODEL_NAME = "ddqn_v2x_agentic_v1.pt"
GOAL_FREE_MODEL_NAME = "ddqn_v2x_goalfree_v1.pt"
E2E_MODEL_NAME = "e2e_transformer_ddqn_v1.pt"   # retired two-action layout
GOAL_PLANNER_MODEL_NAME = "goal_planner_ddqn_v1.pt"

# Retired selectors, kept only so old logs remain readable.
ARCH_GOAL_DDQN = "goal-ddqn"
ARCH_GOAL_FREE = "goal-free-ddqn"
ARCH_E2E = "e2e-transformer"

# Fields that define the experiment condition and therefore the config hash.
# Only the Goal Planner differs between the four variants, so the planner name
# is the one structural field here.
_HASHED_GOAL_PLANNER = (
    "goal_planner",
    "service_flow_scenario",
    "service_context",
    "e2e_reward_version",
    "fixed_goal_weights",
    "history_len",
    "d_model",
    "nhead",
    "num_layers",
    "dim_ff",
    "dropout",
    "learning_rate",
    "target_update",
    "memory_capacity",
    "epsilon_decay",
    "epsilon_decay_mode",
    "training_horizon",
    "sim_time",
    "grad_clip",
    "train_every",
    # two-timescale + exploration + constraint: part of the experimental
    # condition, so two runs that differ in any of them cannot share a hash
    "planner_update_k",
    "critic_warmup_w",
    "beta",
    "duplication_budget",
    "duplication_window",
    "duplication_dual_eta",
    "service_aggregation",
    "run_mode",
)

# GOAL_DDQN_ONLY_OPTIONS / E2E_ONLY_OPTIONS removed: both tuples were unused,
# and every option they named (planner_mode, fixed_goal, planner_interval_ms,
# goal_min_dwell_ms, service_scenario, degradation_scenario) was retired with
# the discrete-Goal planner and no longer exists on the parser, so the list
# could not warn about anything.
TRANSFORMER_ONLY_OPTIONS = ("d_model", "nhead", "num_layers", "dim_ff", "dropout")
E2E_DEBUG_EVERY = 50


def load_binding():
    try:
        import ns3ai_tvt_py as binding
    except ImportError as exc:
        raise RuntimeError(
            "ns3ai_tvt_py module not found; build tvt_ns3 first"
        ) from exc
    if int(binding.CONTRACT_VERSION) != OBSERVATION_CONTRACT_VERSION:
        raise RuntimeError(
            "Python/C++ TVT contract version mismatch: "
            f"binding={binding.CONTRACT_VERSION}, "
            f"python={OBSERVATION_CONTRACT_VERSION}"
        )
    return binding


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_rssi(value: float) -> float:
    low = NORMALIZATION["rssi_min_dbm"]
    high = NORMALIZATION["rssi_max_dbm"]
    return float((np.clip(value, low, high) - low) / (high - low))


def _global_grad_norm(parameters) -> float:
    """Pre-clip global L2 gradient norm.

    Same definition as the value clip_grad_norm_ compares against, and the
    same formula tvt_e2e uses, so baseline and E2E numbers are comparable.
    Read-only: it never touches .grad, the optimizer or any RNG.
    """
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm(2).item()) ** 2
    return math.sqrt(total)


@dataclass
class EpisodeMetrics:
    cumulative_reward: float = 0.0
    throughput_samples: list = None
    delay_samples: list = None
    loss_samples: list = None
    outage_duration_ms: float = 0.0
    qos_valid: int = 0
    qos_satisfied: int = 0
    switches: int = 0
    pingpongs: int = 0
    goal_switches: int = 0
    planner_fallbacks: int = 0
    pending_expired: int = 0
    pending_overflow: int = 0
    pending_unmatched: int = 0
    # Hold-removal evidence. hold_masked_steps must stay 0: it counts formal
    # decisions whose action set was smaller than {Uu, PC5, Both}.
    hold_masked_steps: int = 0
    hold_veto_steps: int = 0
    requested_equals_applied: int = 0
    requested_differs_applied: int = 0
    pending_apply_steps: int = 0
    current_dwell: int = 0

    def __post_init__(self):
        self.throughput_samples = []
        self.delay_samples = []
        self.loss_samples = []
        # Action dwell-time distribution, in decision ticks. A diagnostic:
        # with no hold, how long does an unconstrained learned policy actually
        # stay on one action?
        self.dwell_lengths = []
        # Diagnostics held here until Step 3-5 wires the logging schema.
        self.upper_grad_norms = []
        self.lower_grad_norms = []
        self.total_grad_norms = []
        self.train_time_ms_samples = []
        self.recv_wait_ms_samples = []
        self.clipped_flags = []


def run(args) -> None:
    """Retired: the two-action MP-QUIC-era agent.

    Its state layout (goal one-hot + 7 base features) and its binary action
    space do not exist any more. Rather than silently mis-running, refuse.
    """
    raise SystemExit(
        "the two-action goal-enum agent has been retired by the "
        "dual-connectivity refactor; use --goal-planner "
        f"{{{'|'.join(GOAL_PLANNERS)}}}")


def semantic_experiment_config(args) -> dict:
    """The experiment condition, architecture-relevant fields only.

    Seeds and per-run metadata are excluded on purpose: two seeds of the same
    condition must share one config hash so aggregation can group them.
    """
    config = {
        name: getattr(args, name)
        for name in _HASHED_GOAL_PLANNER if hasattr(args, name)
    }
    config["gamma"] = GAMMA
    config["batch_size"] = BATCH_SIZE
    schedule = resolved_epsilon_schedule(args)
    # The DERIVED decay is what actually ran, so it is the value that enters
    # the hash; the mode and the horizon go in too, so two runs cannot share a
    # hash by arriving at the same decay from different budgets.
    config["epsilon_decay"] = schedule["epsilon_decay"]
    config["epsilon_start"] = schedule["epsilon_start"]
    config["epsilon_min"] = schedule["epsilon_min"]
    config["epsilon_decay_mode"] = schedule["epsilon_decay_mode"]
    config["epsilon_schedule_version"] = schedule["epsilon_schedule_version"]
    config["epsilon_decay_rule"] = schedule["epsilon_decay_rule"]
    config["total_epsilon_decay_steps"] = schedule["total_epsilon_decay_steps"]
    config["training_horizon_N"] = schedule["training_horizon_N"]
    config["sim_time_E"] = schedule["sim_time_E"]
    config["n_formal"] = schedule["n_formal"]
    config["training_horizon_status"] = TRAINING_HORIZON_STATUS
    config["epsilon_annealing_steps"] = EPSILON_ANNEALING_STEPS
    config["epsilon_extension_policy"] = EPSILON_EXTENSION_POLICY
    config["horizon_sensitivity_arm"] = HORIZON_SENSITIVITY_ARM
    config["duplication_partial_window_policy"] = DUPLICATION_PARTIAL_WINDOW_POLICY
    config["duplication_violation_normalized"] = DUPLICATION_VIOLATION_NORMALIZED
    config["duplication_dual_cadence"] = DUPLICATION_DUAL_CADENCE
    config["observation_contract_version"] = OBSERVATION_CONTRACT_VERSION
    config["architecture_version"] = ARCHITECTURE_VERSION
    config["action_dim"] = ACTION_DIM
    config["goal_dim"] = GOAL_DIM
    config["mpquic_enabled"] = False
    config["lower_reward_version"] = LOWER_REWARD_VERSION
    config["upper_objective_version"] = UPPER_OBJECTIVE_VERSION
    config["timing_policy"] = E2E_TIMING_POLICY
    config["semantic_config_version"] = SEMANTIC_CONFIG_VERSION
    config["context_feature_dim"] = CONTEXT_FEATURE_DIM
    config["radio_state_dim"] = RADIO_STATE_DIM
    config["lower_input_dim"] = LOWER_INPUT_DIM
    config["lower_hidden_dim"] = LOWER_HIDDEN_DIM
    config["resource_cost_version"] = RESOURCE_COST_VERSION
    config["upper_objective_version"] = UPPER_OBJECTIVE_VERSION
    config["service_aggregation_mode"] = getattr(
        args, "service_aggregation", SERVICE_AGGREGATION_MODE)
    config["macro_horizon"] = MACRO_HORIZON
    config["goal_interval_ms"] = GOAL_INTERVAL_MS
    config["resource_constraint_version"] = RESOURCE_CONSTRAINT_VERSION
    config["action_hold_enabled"] = ACTION_HOLD_ENABLED
    config["action_transition_semantics"] = ACTION_TRANSITION_SEMANTICS
    config["macro_interval_basis"] = MACRO_INTERVAL_BASIS
    config["macro_terminal_convention"] = MACRO_TERMINAL_CONVENTION
    config["reliability_attribution_version"] = RELIABILITY_ATTRIBUTION_VERSION
    config["goal_exploration_version"] = GOAL_EXPLORATION_VERSION
    config["two_timescale_version"] = TWO_TIMESCALE_VERSION
    config["system_aggregation_mode"] = SYSTEM_AGGREGATION_MODE
    return config


def resolved_epsilon_schedule(args) -> dict:
    """The exploration schedule this run will use.

    In the formal protocol the decay is DERIVED from the frozen training
    budget, so that changing E or N cannot silently change exploration:

        n_formal(E) = 10E - 30
        D_action    = N * n_formal(E) - 1
        decay       = (eps_min / eps_start) ** (1 / D_action)

    D_action counts the decays that precede the run's LAST formal action,
    because the runner decays AFTER acting. The first formal action of the run
    therefore uses eps_start exactly and the last uses eps_min exactly.
    Reaching eps_min is exploration schedule completion, not convergence.
    """
    sim_time = float(getattr(args, "sim_time", FORMAL_EPISODE_LENGTH_S))
    horizon = int(getattr(args, "training_horizon", FORMAL_TRAINING_HORIZON))
    mode = str(getattr(args, "epsilon_decay_mode", EPSILON_DECAY_MODE))
    n_formal = formal_action_count(sim_time)
    if mode == "derived":
        # FROZEN EXTENSION POLICY (D-1E1): the annealing budget is pinned in
        # absolute formal action steps, so N > 161 continues training without
        # stretching exploration. Identical to the old derivation at N = 161.
        total_steps = frozen_epsilon_annealing_steps(sim_time, horizon)
        decay = frozen_epsilon_decay(sim_time, horizon,
                                     EPSILON_START, EPSILON_MIN)
    else:
        total_steps = None
        decay = float(getattr(args, "epsilon_decay", None) or 1.0)
    return {
        "epsilon_decay_mode": mode,
        "epsilon_schedule_version": EPSILON_SCHEDULE_VERSION,
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay": decay,
        "epsilon_decay_rule": EPSILON_DECAY_RULE,
        "total_epsilon_decay_steps": total_steps,
        "training_horizon_N": horizon,
        "sim_time_E": sim_time,
        "n_formal": n_formal,
    }


def resolved_reward_version(args) -> str:
    return getattr(args, "e2e_reward_version", DEFAULT_E2E_REWARD_VERSION)


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)


def _slug(value) -> str:
    text = str(value)
    return "".join(
        character if character.isalnum() else "-" for character in text
    ).strip("-") or "none"


def config_id(args) -> str:
    """Directory-safe identifier of the experiment condition (no seed).

    Deliberately prefixed with the planner name rather than the retired
    C/D/E/F1/F2 labels, so this campaign can never be confused with the
    two-action MP-QUIC campaign in a path, a manifest or a hash.
    """
    parts = [
        f"gp-{_slug(args.goal_planner)}",
        f"ctx-{_slug(args.service_context)}",
        f"flow-{_slug(args.service_flow_scenario)}",
    ]
    clip = getattr(args, "grad_clip", None)
    if clip is not None:
        parts.append(f"clip{clip:g}")
    return "__".join(parts)


def experiment_provenance(args) -> dict:
    """Provenance block shared by config.json, the launcher and the campaign."""
    semantic = semantic_experiment_config(args)
    hashed = hashlib.sha256(_canonical(semantic).encode("utf-8")).hexdigest()
    seed = int(getattr(args, "seed", 0))
    rng_run = getattr(args, "rng_run", None)
    episode_offset = int(getattr(args, "episode_offset", 0))
    episodes = int(getattr(args, "episodes", 0))
    first_episode = episode_offset + FIRST_EPISODE_INDEX
    last_episode = first_episode + max(0, episodes - 1)
    run_payload = {
        "config_hash": hashed,
        "seed": seed,
        # The run is identified by the RNG POLICY plus its base, not by one
        # rngRun: every episode now gets its own realization. Hashing the
        # scalar would make two runs with different episode counts look
        # identical, and would mislabel the realization actually used.
        "ns3_rng_run_base": None if rng_run is None else int(rng_run),
        "ns3_rng_run_policy": EPISODE_RNG_POLICY,
        "scenario_id": SCENARIO_ID,
    }
    run_hash = hashlib.sha256(
        _canonical(run_payload).encode("utf-8")).hexdigest()
    identifier = config_id(args)
    protocol = protocol_freeze_report({
        "duplication_budget": getattr(args, "duplication_budget", None),
        "duplication_window": getattr(args, "duplication_window", None),
        "duplication_dual_eta": getattr(args, "duplication_dual_eta", None),
        "beta": getattr(args, "beta", None),
        "planner_update_k": getattr(args, "planner_update_k", None),
        "critic_warmup_w": getattr(args, "critic_warmup_w", None),
    })
    protocol["run_mode"] = getattr(args, "run_mode", DEFAULT_RUN_MODE)
    if protocol["run_mode"] != "full":
        protocol["frozen"] = False
    return {
        "experiment_id": (
            f"{identifier}__seed{seed:03d}"
            # the RNG BASE, not an episode's rngRun: the latter varies within
            # a single experiment and would not identify it.
            f"__rngbase{-1 if rng_run is None else int(rng_run):03d}"
            f"__{hashed[:8]}"
        ),
        "config_id": identifier,
        "config_hash": hashed,
        "run_hash": run_hash,
        "architecture": getattr(args, "goal_planner", DEFAULT_GOAL_PLANNER),
        "goal_planner": getattr(args, "goal_planner", DEFAULT_GOAL_PLANNER),
        "architecture_version": ARCHITECTURE_VERSION,
        "seed": seed,
        # Provenance only: rngRun never reaches policy, reward, replay,
        # optimizer or action selection. It selects the ENVIRONMENT's
        # realization, one per episode.
        "scenario_id": SCENARIO_ID,
        "ns3_rng_run_policy": EPISODE_RNG_POLICY,
        "ns3_rng_run_base": None if rng_run is None else int(rng_run),
        "ns3_rng_run_first": (
            None if rng_run is None
            else episode_rng_run(rng_run, first_episode)),
        "ns3_rng_run_last": (
            None if rng_run is None
            else episode_rng_run(rng_run, last_episode)),
        "episode_offset": episode_offset,
        "first_episode_index": first_episode,
        "last_episode_index": last_episode,
        "lower_reward_version": LOWER_REWARD_VERSION,
        "upper_objective_version": UPPER_OBJECTIVE_VERSION,
        "timing_policy": E2E_TIMING_POLICY,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "action_hold_enabled": ACTION_HOLD_ENABLED,
        "protocol_freeze": protocol,
        "protocol_frozen": bool(protocol["frozen"]),
        "run_mode": protocol["run_mode"],
        "semantic_config": semantic,
    }


def _mean_or_none(samples):
    """Mean over actual samples; None when nothing was recorded.

    Used for gradient and timing means so a replay warm-up with no training
    steps is reported as "no samples" instead of being averaged down by zeros.
    """
    return float(np.mean(samples)) if samples else None


def _open_interface(binding, label: str, segment: str = None):
    """Create the ns3-ai shared-memory interface (identical to the baseline)."""
    segment = SHM_SEG if segment is None else segment
    interface = binding.Ns3AiMsgInterfaceImpl(
        True,
        False,
        False,
        SHM_SIZE,
        segment,
        SHM_ENV_NAME,
        SHM_ACT_NAME,
        SHM_SYNC_NAME,
    )
    print(
        f"[{label}] TVT shared memory ready: {segment}; "
        f"env_size={binding.env_struct_size()} act_size={binding.act_struct_size()}",
        flush=True,
    )
    return interface


def _protocol_values(args) -> dict:
    """The unresolved protocol numbers as this run supplied them."""
    return {
        "duplication_budget": args.duplication_budget,
        "duplication_window": args.duplication_window,
        "duplication_dual_eta": args.duplication_dual_eta,
        "beta": args.beta,
        "p_global": args.p_global,
        "planner_update_k": args.planner_update_k,
        "critic_warmup_w": args.critic_warmup_w,
    }


def enforce_run_mode(args) -> dict:
    """Full training refuses to start while any protocol number is unresolved.

    A smoke test may supply values, but they are recorded as TEST-ONLY: they
    exist to exercise the plumbing and must never be promoted to a formal
    default. That distinction is carried in every artefact this run writes.
    """
    values = _protocol_values(args)
    report = protocol_freeze_report(values)
    report["run_mode"] = args.run_mode
    if args.run_mode == "full":
        require_frozen_protocol(values)
        report["test_only"] = False
    else:
        report["test_only"] = bool(report["resolved"])
        report["frozen"] = False       # a smoke run is never a frozen protocol
    return report


class MacroRecord:
    """One goal interval, held open until its reliability cohorts mature.

    The macro return is completed from MATURED TICK COHORTS, never from the
    KPI window that happens to be current: a deadline outcome ripens ~250 ms
    after transmission, so crediting it to whatever goal is in force when it
    ripens would attribute an outcome to a goal that did not cause it.
    """

    __slots__ = ("macro_id", "context", "goal", "planner_goal", "ticks_opened",
                 "ticks_reported", "macro_return", "next_context", "closed",
                 "tick_services", "diag", "cohort_breakdowns")

    def __init__(self, macro_id, context, goal, planner_goal):
        self.macro_id = int(macro_id)
        self.context = np.asarray(context, dtype=np.float32).copy()
        self.goal = np.asarray(goal, dtype=np.float32).copy()
        self.planner_goal = np.asarray(planner_goal, dtype=np.float32).copy()
        self.ticks_opened = 0
        self.ticks_reported = set()
        self.macro_return = 0.0
        self.next_context = None
        self.closed = False
        self.tick_services = {}
        # Goal-emission diagnostics, captured once and carried to the harvest
        # sink so a screening artifact describes the macro that produced it.
        self.diag = {}
        self.cohort_breakdowns = {}

    @property
    def complete(self) -> bool:
        return (self.closed
                and self.next_context is not None
                and len(self.ticks_reported) >= self.ticks_opened)


def run_goal_planner(args, harvest_sink=None) -> None:
    """Macro/micro hierarchical runner.

        MACRO  every MACRO_HORIZON DECISION TICKS (not "valid reward steps"):
               pi_goal(context) -> g_k, held for exactly H ticks. The macro
               return is assembled from matured tick cohorts, so the goal is
               credited with the outcomes it actually caused, and the macro
               transition is pushed only once those cohorts have ripened.
        MICRO  every decision interval: Q_lower(radio, g_k) -> {Uu, PC5, Both}

    `harvest_sink`, when given, receives every completed macro transition
    together with its goal-emission diagnostics. It is how screening runs
    collect protocol-selection data; it changes nothing about the run itself.

    ACTION HOLD: none. Every formal decision may request any of the three
    actions, because an action change in this simulator is traffic steering
    and costs the modelled network nothing. switch_count, switch_rate,
    pingpong_count and the action dwell-time distribution are measured as
    secondary diagnostics; nothing penalises switching.

    Goal planning runs in BOTH train and eval; only learning and exploration
    are disabled in eval. Observation warm-up fills the planner history with
    real frames before any traffic, reward, replay or KPI accounting begins
    (ns-3 reports this through env.formal_accounting_active).
    """
    freeze = enforce_run_mode(args)
    set_global_seed(args.seed)
    rng = random.Random(args.seed)
    flow_timeline = parse_service_flow_scenario(args.service_flow_scenario)
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path) if args.model_path else (
        output_dir / GOAL_PLANNER_MODEL_NAME)

    transformer = TransformerConfig(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_ff, dropout=args.dropout)
    planner_kwargs = {}
    if args.goal_planner in ("transformer", "transformer_shuffled",
                             "transformer_h1"):
        planner_kwargs["config"] = transformer
    elif args.goal_planner == "fixed":
        planner_kwargs["fixed_weights"] = args.fixed_goal_weights

    history_len = 1 if args.goal_planner == "transformer_h1" else args.history_len
    agent = ConnectivityAgent(
        planner_kind=args.goal_planner,
        capacity=args.memory_capacity,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr=args.learning_rate,
        target_update=args.target_update,
        grad_clip=args.grad_clip,
        history_len=history_len,
        macro_horizon=MACRO_HORIZON,
        planner_update_k=args.planner_update_k,
        critic_warmup_w=args.critic_warmup_w,
        beta=args.beta,
        p_global=args.p_global,
        duplication_budget=args.duplication_budget,
        duplication_window=args.duplication_window,
        duplication_dual_eta=args.duplication_dual_eta,
        init_seed=args.seed,          # lower/critic init is planner-independent
        **planner_kwargs)
    schedule = resolved_epsilon_schedule(args)
    expected_n_formal = int(schedule["n_formal"])
    agent.attach_bookkeeping(
        epsilon_start=schedule["epsilon_start"],
        epsilon_min=schedule["epsilon_min"],
        epsilon_decay=schedule["epsilon_decay"],
        epsilon_decay_mode=schedule["epsilon_decay_mode"],
        epsilon_schedule_version=schedule["epsilon_schedule_version"],
        total_epsilon_decay_steps=schedule["total_epsilon_decay_steps"],
        training_horizon_n=schedule["training_horizon_N"],
        sim_time_e=schedule["sim_time_E"],
        n_formal=expected_n_formal,
        pending_max_entries=args.pending_max_entries,
        pending_max_age_s=args.pending_max_age_s)
    # Hand the sampling/action RNG to the agent BEFORE any load: it drives
    # both replay sampling and epsilon-greedy choice, so an exact resume has
    # to restore its stream position. Registering it changes nothing about
    # how it is used -- it is still passed into train_lower / train_macro /
    # select_action exactly as before.
    agent.attach_rng(rng)

    # ---- resume episode-offset cross-check -------------------------------
    # The episode-specific ns-3 RngRun is derived from the ABSOLUTE episode
    # index, so a resume that restarts the index silently replays the earlier
    # episodes' channel realizations. Verify the caller's claim against the
    # summary the previous invocation actually wrote, and refuse on mismatch
    # rather than producing a quietly wrong trajectory.
    if args.resume:
        summary_path = Path(args.output_dir) / "episode_summary.csv"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as handle:
                completed = max(0, sum(1 for _ in handle) - 1)  # minus header
            if completed != args.episode_offset:
                raise SystemExit(
                    "RESUME_EPISODE_OFFSET_MISMATCH: --episode-offset="
                    f"{args.episode_offset} but {summary_path} records "
                    f"{completed} completed episodes. The episode-specific "
                    "RngRun sequence would not continue correctly. Pass "
                    f"--episode-offset {completed}.")
        elif args.episode_offset != 0:
            raise SystemExit(
                "RESUME_EPISODE_OFFSET_MISMATCH: --episode-offset="
                f"{args.episode_offset} but {summary_path} does not exist, so "
                "no episode has been completed in this output directory.")

    if args.resume or args.eval:
        agent.load(model_path, resume_training=not args.eval)
    if args.eval:
        agent.epsilon = 0.0
        # EVAL: beta_effective = 0. No Dirichlet sample is drawn at all, so the
        # exploration RNG is not consumed and g_beh == g_det exactly.
        agent.beta = 0.0
    elif schedule["epsilon_decay_mode"] == "derived":
        # The horizon is a budget, not a suggestion: refuse an invocation that
        # would run past it, which would carry epsilon below eps_min for a
        # stretch the schedule never accounted for.
        done_episodes = agent.epsilon_decay_step // expected_n_formal
        try:
            validate_invocation_budget(
                done_episodes, int(args.episodes), int(args.training_horizon))
        except InvocationBudgetError as exc:
            raise SystemExit(str(exc)) from exc

    counts = agent.parameter_counts()
    configuration = serializable_configuration()
    configuration["cli"] = vars(args)
    configuration["model_path"] = str(model_path)
    configuration["algorithm"] = ALGORITHM
    configuration["architecture"] = args.goal_planner
    configuration["provenance"] = experiment_provenance(args)
    configuration["protocol_freeze"] = freeze
    configuration["goal_planner_run"] = {
        "goal_planner": args.goal_planner,
        "architecture_id": ARCHITECTURE_VERSION,
        "action_dim": ACTION_DIM,
        "goal_dim": GOAL_DIM,
        "mpquic_enabled": False,
        "action_hold_enabled": ACTION_HOLD_ENABLED,
        "action_transition_semantics": ACTION_TRANSITION_SEMANTICS,
        "physical_switching_cost_modeled": PHYSICAL_SWITCHING_COST_MODELED,
        "action_hold_status": ACTION_HOLD_STATUS,
        "history_len": history_len,
        "context_feature_dim": CONTEXT_FEATURE_DIM,
        "radio_state_dim": RADIO_STATE_DIM,
        "lower_input_dim": LOWER_INPUT_DIM,
        "macro_horizon": MACRO_HORIZON,
        "macro_interval_basis": MACRO_INTERVAL_BASIS,
        "goal_interval_ms": GOAL_INTERVAL_MS,
        "decision_interval_ms": DECISION_INTERVAL_MS,
        "macro_return_convention": MACRO_RETURN_CONVENTION,
        "macro_terminal_convention": MACRO_TERMINAL_CONVENTION,
        "reliability_attribution_version": RELIABILITY_ATTRIBUTION_VERSION,
        "goal_exploration_version": GOAL_EXPLORATION_VERSION,
        "two_timescale_version": TWO_TIMESCALE_VERSION,
        "planner_update_k": args.planner_update_k,
        "critic_warmup_w": args.critic_warmup_w,
        "transformer": asdict(transformer),
        "parameter_counts": counts,
        "lower_reward_version": LOWER_REWARD_VERSION,
        "upper_objective_version": UPPER_OBJECTIVE_VERSION,
        "service_aggregation_mode": args.service_aggregation,
        "system_aggregation_mode": SYSTEM_AGGREGATION_MODE,
        "resource_constraint_version": RESOURCE_CONSTRAINT_VERSION,
        "duplication_constraint": agent.duplication.diagnostics(),
        "service_flow_scenario": args.service_flow_scenario,
        "service_context_mode": args.service_context,
        "grad_clip": args.grad_clip,
        "timing_policy": E2E_TIMING_POLICY,
    }
    logger = ExperimentLogger(output_dir, configuration,
                              architecture=args.goal_planner)
    print(f"[GP] planner={args.goal_planner} H={MACRO_HORIZON} "
          f"hold={ACTION_HOLD_ENABLED} run_mode={args.run_mode} "
          f"frozen={freeze['frozen']} params={counts}", flush=True)
    if freeze["missing"]:
        print("[GP][PROTOCOL] unresolved (TEST-ONLY run): "
              + ", ".join(freeze["missing"]), flush=True)

    binding = load_binding()
    if bool(getattr(binding, "ACTION_HOLD_ENABLED", False)):
        raise RuntimeError(
            "ns-3 reports action_hold_enabled=1; this architecture requires "
            "the hold to be removed on both sides")
    shm_segment = sanitize_shm_segment(
        getattr(args, "shm_segment", SHM_SEG_DEFAULT))
    interface = _open_interface(binding, "GP", shm_segment)
    no_macro = int(binding.NO_MACRO)

    episode = 0
    total_steps = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            episode += 1
            # Absolute episode index across resumes. FIRST_EPISODE_INDEX fixes
            # the convention (first training episode = 1) in one place so the
            # driver and the runner cannot disagree.
            absolute_episode = (args.episode_offset + episode
                                - 1 + FIRST_EPISODE_INDEX)
            expected_rng_run = (
                None if args.rng_run is None
                else episode_rng_run(args.rng_run, absolute_episode))
            history = ContextHistory(history_len=history_len,
                                     feature_dim=CONTEXT_FEATURE_DIM)
            tracker = MetricTracker()
            metrics = EpisodeMetrics()
            agent.discard_pending()
            agent.duplication.reset()
            last_switch = None
            previous_action = None
            step = 0
            formal_steps = 0
            warmup_frames = 0

            # ---- macro state -------------------------------------------
            goal = None
            planner_goal = None
            # goal_support_distance_MACRO for the macro currently in force.
            # Held across its ticks so every row of the macro carries it, and
            # kept strictly separate from the per-planner-update batch mean.
            macro_support_distance = None
            macro_support_size = 0
            current_macro = None
            pending_macros = {}
            next_macro_id = 0
            context_history = None

            def try_push_macro(record, force_terminal=False):
                """Push a macro transition once its cohorts have matured.

                force_terminal is the episode-end path: the episode genuinely
                ends, so the truncated macro is pushed with done=True
                (bootstrap 0) and micro_steps = H_terminal, the number of
                ticks whose cohort actually matured. The discount exponent and
                the number of accumulated terms therefore always agree.
                """
                ticks = len(record.ticks_reported)
                if force_terminal:
                    if ticks == 0:
                        tracker.macros_discarded += 1
                        return False
                    nxt = (record.next_context if record.next_context is not None
                           else record.context)
                    done = True
                elif record.complete:
                    nxt = record.next_context
                    ticks = record.ticks_opened
                    done = False
                else:
                    return False
                agent.store_macro(MacroTransition(
                    context=record.context,
                    goal=record.goal.astype(np.float32),
                    macro_return=float(record.macro_return),
                    micro_steps=int(ticks),
                    next_context=np.asarray(nxt, dtype=np.float32),
                    done=done))
                tracker.note_macro_push(record.macro_return, ticks,
                                        truncated=force_terminal)
                if harvest_sink is not None:
                    harvest_sink.record_macro(
                        record=record, micro_steps=int(ticks), done=done,
                        next_context=np.asarray(nxt, dtype=np.float32),
                        agent=agent, truncated=bool(force_terminal))
                if not args.eval:
                    # MACRO CADENCE. The dual variable steps here, on the same
                    # event stream as the goal critic it constrains -- once per
                    # macro push, never once per 100 ms tick over a window that
                    # moved by a single sample. Measurement stays at tick
                    # resolution (duplication.observe below).
                    agent.duplication.dual_update()
                    macro_stats = agent.train_macro(rng)
                    if macro_stats is not None:
                        tracker.note_macro(macro_stats)
                        if harvest_sink is not None:
                            harvest_sink.record_training(macro_stats)
                return True

            while True:
                recv_start = time.perf_counter()
                interface.PyRecvBegin()
                recv_wait_ms = (time.perf_counter() - recv_start) * 1000.0
                env = interface.GetCpp2PyStruct()
                contract_version = int(env.contract_version)
                done = bool(env.done)
                decision_id = int(env.decision_id)
                applied_decision_id = int(env.applied_decision_id)
                applied_action = int(env.applied_action)
                apply_status = int(env.apply_status)
                transition_valid = bool(env.transition_valid)
                current_action = int(env.current_action)
                formal = bool(int(env.formal_accounting_active))
                interface.PyRecvEnd()

                inference_wall = time.time()
                inference_start = time.perf_counter()
                action = current_action
                q_values = np.zeros(ACTION_DIM, dtype=np.float64)
                reward_result = None
                stored_transition = False
                row = None
                macro_id_sent = no_macro
                macro_tick_sent = -1
                metrics.recv_wait_ms_samples.append(recv_wait_ms)

                if contract_version != OBSERVATION_CONTRACT_VERSION:
                    raise RuntimeError(
                        "ns-3/Python TVT observation contract mismatch: "
                        f"ns3={contract_version}, "
                        f"python={OBSERVATION_CONTRACT_VERSION}")
                if int(env.action_hold_enabled):
                    raise RuntimeError(
                        "ns-3 applied an action hold; this architecture "
                        "requires requested_action == applied_action")

                if not done:
                    simulation_time_s = float(env.simulation_time_s)
                    frame = build_context_frame(
                        env, flow_timeline,
                        service_context_mode=args.service_context)
                    history.append(frame)
                    context_history = history.snapshot()
                    radio_state = build_radio_state(frame)
                    active_services = active_services_for_time(
                        flow_timeline, simulation_time_s)
                    if not formal:
                        warmup_frames += 1

                    # ---- close the previous micro decision --------------
                    pending = agent.pending_for(applied_decision_id)
                    actual_switch = False
                    reverse_switch = False
                    if pending is not None and transition_valid:
                        source = int(pending["source_action"])
                        actual_switch = (
                            apply_status == 2 and applied_action != source)
                        reverse_switch = (
                            actual_switch
                            and last_switch == (applied_action, source))
                        if actual_switch:
                            metrics.switches += 1
                            if reverse_switch:
                                metrics.pingpongs += 1
                            last_switch = (source, applied_action)
                        reward_result = compute_reward(
                            env, applied_action,
                            goal_weights=pending["goal"],
                            active_services=active_services,
                            actual_switch=actual_switch,
                            reverse_switch=reverse_switch,
                            aggregation=args.service_aggregation)
                        # Warm-up steps never enter replay or formal KPI.
                        stored_transition = agent.consume_outcome(
                            applied_decision_id, applied_action, reward_result,
                            radio_state, transition_valid,
                            learn=(not args.eval) and formal)
                        if reward_result is not None and formal:
                            tracker.note_reward(reward_result)

                    # ---- matured tick cohort -> macro return -----------
                    cohort = read_tick_cohort(env)
                    cohort_breakdown = None
                    cohort_r_upper = None
                    pushed_record = None
                    if cohort is not None:
                        record = None
                        if cohort.macro_id == no_macro:
                            tracker.cohorts_dropped_no_macro += 1
                        else:
                            record = pending_macros.get(int(cohort.macro_id))
                            if record is None:
                                tracker.cohorts_dropped_no_macro += 1
                        if record is not None:
                            services = record.tick_services.get(
                                int(cohort.macro_tick), active_services)
                            cohort_breakdown = cohort_upper_objective(
                                cohort, services, args.service_aggregation)
                            tracker.note_cohort(cohort_breakdown)
                            record.cohort_breakdowns[int(cohort.macro_tick)] = (
                                None if cohort_breakdown is None else {
                                    "u_qos": float(cohort_breakdown["u_qos"]),
                                    "V_system": float(cohort_breakdown["V_system"]),
                                    "binding_service":
                                        cohort_breakdown["binding_service"],
                                    "binding_services":
                                        list(cohort_breakdown["binding_services"]),
                                    "binding_kpi": cohort_breakdown["binding_kpi"],
                                    "per_service": {
                                        name: {k: float(v)
                                               for k, v in values.items()}
                                        for name, values
                                        in cohort_breakdown["per_service"].items()},
                                })
                            # Constrained system objective. The duplication
                            # constraint is INERT until its three numbers are
                            # frozen, so today J == U_qos exactly.
                            agent.duplication.observe(cohort.duplication_rate)
                            if cohort_breakdown is None:
                                # protocol: a tick with no valid upper reward
                                # contributes 0. It does NOT extend the macro.
                                cohort_r_upper = 0.0
                            else:
                                cohort_r_upper = (
                                    agent.duplication.constrained_objective(
                                        float(cohort_breakdown["u_qos"])))
                            tick_index = int(cohort.macro_tick)
                            if 0 <= tick_index < MACRO_HORIZON:
                                record.macro_return += (
                                    GAMMA ** tick_index) * cohort_r_upper
                                record.ticks_reported.add(tick_index)
                            if record.complete:
                                if try_push_macro(record):
                                    pushed_record = record
                                pending_macros.pop(record.macro_id, None)
                            metrics.cumulative_reward += (
                                cohort_r_upper if cohort_r_upper else 0.0)

                    # ---- macro tier: a new goal every H DECISION TICKS --
                    goal_source = "warmup"
                    if formal:
                        goal_source = "macro"
                        if (current_macro is None
                                or current_macro.ticks_opened >= MACRO_HORIZON):
                            if current_macro is not None:
                                current_macro.closed = True
                                current_macro.next_context = (
                                    context_history.copy())
                                if current_macro.complete:
                                    try_push_macro(current_macro)
                                    pending_macros.pop(
                                        current_macro.macro_id, None)
                            goal, planner_goal = agent.plan_goal(
                                context_history,
                                explore=(not args.eval))
                            current_macro = MacroRecord(
                                next_macro_id, context_history, goal,
                                planner_goal)
                            pending_macros[next_macro_id] = current_macro
                            next_macro_id += 1
                            tracker.note_goal(goal, planner_goal)
                            # PRE-INSERTION SUPPORT. The support distance is
                            # measured against the behaviour goals ALREADY in
                            # the replay, before this macro's own behaviour
                            # goal exists -- the Dirichlet sample below has not
                            # been drawn yet, and this macro's transition is
                            # only pushed once its cohorts mature, several
                            # ticks later. A sample must never support itself,
                            # or the gate becomes true by construction.
                            macro_support_size = len(agent.macro_replay)
                            macro_support_distance = (
                                agent.goal_support_distance(planner_goal))
                            q_behaviour = agent.q_goal(context_history, goal)
                            q_planner = agent.q_goal(context_history, planner_goal)
                            tracker.note_goal_support(
                                macro_support_distance, q_behaviour, q_planner)
                            behaviour_stats = goal_statistics(goal)
                            emitted_stats = goal_statistics(planner_goal)
                            current_macro.diag = {
                                "episode": episode,
                                "simulation_time_s": simulation_time_s,
                                "goal_distance": float(np.linalg.norm(
                                    np.asarray(goal, dtype=np.float64)
                                    - np.asarray(planner_goal, dtype=np.float64))),
                                "pre_insert_support_distance":
                                    macro_support_distance,
                            "pre_insert_support_size": macro_support_size,
                                "behaviour_goal_entropy":
                                    behaviour_stats["goal_entropy"],
                                "planner_goal_entropy":
                                    emitted_stats["goal_entropy"],
                                "q_goal_behaviour": q_behaviour,
                                "q_goal_planner": q_planner,
                                "beta": agent.beta,
                                "realised_radius": float(np.linalg.norm(
                                    np.asarray(goal, dtype=np.float64)
                                    - np.asarray(planner_goal, dtype=np.float64))),
                                "expected_squared_radius":
                                    agent.expected_squared_radius(planner_goal),
                                "epsilon": float(agent.epsilon),
                                "active_services": [s.name
                                                    for s in active_services],
                            }

                        macro_tick_sent = current_macro.ticks_opened
                        macro_id_sent = current_macro.macro_id
                        current_macro.tick_services[macro_tick_sent] = (
                            active_services)
                        current_macro.ticks_opened += 1
                    else:
                        # Warm-up still needs a goal for the lower controller,
                        # but it opens no macro and never explores.
                        goal, planner_goal = agent.plan_goal(
                            context_history, explore=False)

                    # ---- micro tier: action ----------------------------
                    # NO HOLD. Every formal decision evaluates all three
                    # actions; nothing masks the action set.
                    valid_actions = list(range(ACTION_DIM))
                    action, q_values = agent.select_action(
                        radio_state, goal, valid_actions, rng)
                    tracker.note_action(action, actual_switch, reverse_switch)
                    if formal:
                        if previous_action is None or action == previous_action:
                            metrics.current_dwell += 1
                        else:
                            metrics.dwell_lengths.append(metrics.current_dwell)
                            metrics.current_dwell = 1
                    agent.remember_decision(
                        decision_id, radio_state, goal, action, current_action,
                        sim_time_s=simulation_time_s, epsilon=agent.epsilon)

                    if formal:
                        formal_steps += 1
                        if len(valid_actions) < ACTION_DIM:
                            metrics.hold_masked_steps += 1
                        # TEST B evidence. Only a RESOLVED status can be
                        # compared: while a change is still PENDING the echo
                        # deliberately reports the action still in force, and
                        # that is a 100 ms telemetry poll, not a veto. The
                        # authoritative check is ns-3's own
                        # [ACTION-MISMATCH][PROTOCOL-ERROR] counter.
                        if apply_status == 3:          # HOLD_VETO_RETIRED
                            metrics.hold_veto_steps += 1
                        if apply_status in (1, 2):     # STAY | APPLIED
                            if int(env.requested_action) == int(env.applied_action):
                                metrics.requested_equals_applied += 1
                            else:
                                metrics.requested_differs_applied += 1
                        elif apply_status == 6:        # PENDING (poll latency)
                            metrics.pending_apply_steps += 1
                        useful_bps = float(env.logical_goodput_bps)
                        metrics.throughput_samples.append(useful_bps)
                        if bool(env.delay_valid):
                            metrics.delay_samples.append(float(env.e2e_delay_ms))
                        if bool(env.deadline_loss_valid):
                            metrics.loss_samples.append(
                                float(env.deadline_loss_rate))
                        qos_result = qos_satisfied_multi(env, active_services)
                        if qos_result is not None:
                            metrics.qos_valid += 1
                            metrics.qos_satisfied += int(qos_result)
                    else:
                        useful_bps = 0.0

                    inference_time_ms = (
                        time.perf_counter() - inference_start) * 1000.0
                    stats = goal_statistics(goal)
                    planner_stats = goal_statistics(planner_goal)
                    dup = agent.duplication.diagnostics()
                    last_macro = tracker.last_macro or {}
                    row = {
                        "episode": episode,
                        "decision_id": decision_id,
                        "applied_decision_id": applied_decision_id,
                        "simulation_time_s": simulation_time_s,
                        "phase": "eval" if args.eval else "train",
                        "formal_accounting": int(formal),
                        "macro_step": macro_tick_sent,
                        "macro_id": (None if macro_id_sent == no_macro
                                     else int(macro_id_sent)),
                        "macro_tick": macro_tick_sent,
                        "goal_source": goal_source,
                        "uu_rssi": float(env.uu_rssi),
                        "pc5_rssi": float(env.pc5_rssi),
                        "uu_delay_ms": float(env.uu_delay_ms),
                        "pc5_delay_ms": float(env.pc5_delay_ms),
                        "uu_loss_rate": float(env.uu_loss_rate),
                        "pc5_loss_rate": float(env.pc5_loss_rate),
                        "uu_measurement_valid": int(env.uu_measurement_valid),
                        "pc5_measurement_valid": int(env.pc5_measurement_valid),
                        "uu_measurement_age_ms": float(env.uu_measurement_age_ms),
                        "pc5_measurement_age_ms": float(env.pc5_measurement_age_ms),
                        "active_services": [s.name for s in active_services],
                        # RETIRED column: the synthetic merged requirement is
                        # never used by the objective. Kept as an empty column
                        # so old readers do not silently misalign.
                        "effective_requirement": None,
                        **mobility_context_raw(env),
                        "service_context": frame[V_SLICE],
                        "context_features": frame,
                        "radio_state": radio_state,
                        "w_rel": float(goal[0]), "w_lat": float(goal[1]),
                        "w_thr": float(goal[2]), "w_cost": float(goal[3]),
                        "goal_entropy": stats["goal_entropy"],
                        "planner_w_rel": float(planner_goal[0]),
                        "planner_w_lat": float(planner_goal[1]),
                        "planner_w_thr": float(planner_goal[2]),
                        "planner_w_cost": float(planner_goal[3]),
                        "planner_goal_entropy": planner_stats["goal_entropy"],
                        "goal_distance": float(
                            np.linalg.norm(np.asarray(goal, dtype=np.float64)
                                           - np.asarray(planner_goal,
                                                        dtype=np.float64))),
                        "beta": agent.beta,
                        "expected_squared_radius": (
                            None if planner_goal is None
                            else agent.expected_squared_radius(planner_goal)),
                        # MACRO cadence, single goal, live context. The only
                        # support metric the formal sigma screen reads.
                        "goal_support_distance_macro": macro_support_distance,
                        "pre_insert_support_size": macro_support_size,
                        # PLANNER-UPDATE cadence, batch mean. Diagnostic only.
                        "goal_support_distance_update":
                            agent.last_goal_support_update,
                        "epsilon_decay_step": int(agent.epsilon_decay_step),
                        "q_goal_behaviour": (
                            tracker.q_goal_behaviour[-1]
                            if tracker.q_goal_behaviour else None),
                        "q_goal_planner": (
                            tracker.q_goal_planner[-1]
                            if tracker.q_goal_planner else None),
                        "q_goal_range": last_macro.get("q_goal_range"),
                        "critic_warmup_done": last_macro.get(
                            "critic_warmup_done"),
                        "planner_updated": last_macro.get("planner_updated"),
                        "action_hold_enabled": int(ACTION_HOLD_ENABLED),
                        "valid_actions": valid_actions,
                        "requested_action": action,
                        "applied_action": (
                            applied_action if transition_valid else None),
                        "action_name": ACTION_NAMES[action],
                        "applied_duplication_ratio": duplication_ratio(action),
                        "previous_action": previous_action,
                        # DIAGNOSTIC ONLY. Recorded because the campaign must
                        # be able to show how much the policy moves between
                        # duplication levels; never a state feature, never a
                        # reward term. is_switch is RETIRED: "the path
                        # changed" denotes nothing when the path never changes.
                        "duplication_delta": (
                            None if previous_action is None else
                            abs(duplication_ratio(action)
                                - duplication_ratio(previous_action))),
                        "duplication_action_changed": (
                            None if previous_action is None
                            else int(action != previous_action)),
                        "apply_status": apply_status,
                        "transition_valid": transition_valid,
                        **{f"q_{ACTION_NAMES[i]}": float(q_values[i])
                           for i in range(ACTION_DIM)},
                        "r_rel": (None if reward_result is None
                                  else reward_result.components["r_rel"]),
                        "r_lat": (None if reward_result is None
                                  else reward_result.components["r_lat"]),
                        "r_thr": (None if reward_result is None
                                  else reward_result.components["r_thr"]),
                        "r_cost": (None if reward_result is None
                                   else reward_result.components["r_cost"]),
                        "lower_reward": (None if reward_result is None
                                         else reward_result.lower_total),
                        "upper_reward": (None if reward_result is None
                                         else reward_result.upper_total),
                        "viol_delay": (None if reward_result is None
                                       else reward_result.violations["delay"]),
                        "viol_pdr": (None if reward_result is None
                                     else reward_result.violations["pdr"]),
                        "viol_throughput": (
                            None if reward_result is None
                            else reward_result.violations["throughput"]),
                        "viol_outage": (None if reward_result is None
                                        else reward_result.violations["outage"]),
                        "service_violations": (
                            None if reward_result is None
                            else reward_result.service_violations),
                        "binding_service": (None if reward_result is None
                                            else reward_result.binding_service),
                        "binding_services": (None if reward_result is None
                                             else reward_result.binding_services),
                        "binding_kpi": (None if reward_result is None
                                        else reward_result.binding_kpi),
                        "v_system": (None if reward_result is None
                                     else reward_result.v_system),
                        "u_qos_step": (None if reward_result is None
                                       else reward_result.upper_total),
                        "cohort_ready": int(cohort is not None),
                        "cohort_macro_id": (
                            None if cohort is None or cohort.macro_id == no_macro
                            else int(cohort.macro_id)),
                        "cohort_macro_tick": (None if cohort is None
                                              else int(cohort.macro_tick)),
                        "cohort_action": (None if cohort is None
                                          else int(cohort.action)),
                        "cohort_logical_tx": (None if cohort is None
                                              else int(cohort.logical_tx)),
                        "cohort_matured": (None if cohort is None
                                           else int(cohort.matured)),
                        "cohort_pdr": (None if cohort is None
                                       else float(cohort.pdr)),
                        "cohort_mean_delay_ms": (None if cohort is None
                                                 else cohort.mean_delay_ms),
                        "cohort_u_qos": (None if cohort_breakdown is None
                                         else float(cohort_breakdown["u_qos"])),
                        "cohort_r_upper": cohort_r_upper,
                        # CORE metric: the cohort's USEFUL (first-arrival,
                        # duplicate-suppressed) throughput. It already feeds
                        # the upper objective via TickCohort.throughput_bps();
                        # this only records the value that is already computed.
                        "cohort_logical_goodput_bps": (
                            None if cohort is None
                            else float(cohort.logical_goodput_bps)),
                        "cohort_v_system": (
                            None if cohort_breakdown is None
                            else float(cohort_breakdown["V_system"])),
                        "cohort_binding_service": (
                            None if cohort_breakdown is None
                            else cohort_breakdown["binding_service"]),
                        "cohort_binding_services": (
                            None if cohort_breakdown is None
                            else list(cohort_breakdown["binding_services"])),
                        "cohort_binding_kpi": (
                            None if cohort_breakdown is None
                            else cohort_breakdown["binding_kpi"]),
                        "cohort_service_violations": (
                            None if cohort_breakdown is None
                            else cohort_breakdown["per_service"]),
                        "cohort_duplication_rate": (
                            None if cohort is None
                            else float(cohort.duplication_rate)),
                        "cohort_backlog": (None if cohort is None
                                           else int(cohort.backlog)),
                        "macro_pushed": int(pushed_record is not None),
                        "macro_pushed_id": (None if pushed_record is None
                                            else pushed_record.macro_id),
                        "macro_pushed_return": (
                            None if pushed_record is None
                            else float(pushed_record.macro_return)),
                        "macro_pushed_ticks": (
                            None if pushed_record is None
                            else int(pushed_record.ticks_opened)),
                        "duplication_rate": (
                            None if reward_result is None
                            else reward_result.duplication_rate),
                        "dup_window_rate": dup["window_rate"],
                        "dup_window_full": int(bool(dup["window_full"])),
                        "dup_constraint_violation": dup["violation"],
                        "dup_violation_norm": dup["violation_norm"],
                        "dup_lambda": dup["lambda"],
                        "dup_penalty": dup["penalty"],
                        "dup_constraint_enabled": int(dup["enabled"]),
                        "lower_reward_version": LOWER_REWARD_VERSION,
                        "upper_objective_version": UPPER_OBJECTIVE_VERSION,
                        "deadline_loss_rate": float(env.deadline_loss_rate),
                        "e2e_delay_ms": float(env.e2e_delay_ms),
                        "useful_throughput_bps": useful_bps,
                        "logical_goodput_bps": float(env.logical_goodput_bps),
                        "uu_throughput_bps": float(env.uu_throughput_bps),
                        "pc5_throughput_bps": float(env.pc5_throughput_bps),
                        # The retired proxy, recomputed for audit only. Never a reward
                        # input; logged so the correction stays checkable from the log.
                        "retired_proxy_throughput_bps": max(
                            float(env.uu_throughput_bps), float(env.pc5_throughput_bps)),
                        "logical_tx_packets": int(env.logical_tx_packets),
                        "logical_rx_packets": int(env.logical_rx_packets),
                        "attempted_physical_copies_tx": int(
                            env.attempted_physical_copies_tx),
                        "successful_physical_copies_tx": int(
                            env.successful_physical_copies_tx),
                        "duplicate_copies_rx": int(env.duplicate_copies_rx),
                        "attempted_copies_per_logical":
                            attempted_copies_per_logical(env),
                        "successful_copies_per_logical":
                            successful_copies_per_logical(env),
                        "duplication_overhead": duplication_overhead(env),
                        "epsilon": agent.epsilon,
                        "td_loss": None, "critic_loss": None,
                        "planner_loss": None, "planner_grad_norm": None,
                        "critic_grad_norm": None, "lower_grad_norm": None,
                        "total_grad_norm": None, "grad_clipped": None,
                        "replay_size": len(agent.replay),
                        "macro_replay_size": len(agent.macro_replay),
                        "pending_size": None, "pending_expired": None,
                        "pending_overflow": None, "unmatched_outcome": None,
                        "inference_time_ms": inference_time_ms,
                        "train_time_ms": None,
                        "recv_wait_ms": recv_wait_ms,
                        "goal_planner": args.goal_planner,
                        "service_flow_scenario": args.service_flow_scenario,
                        "service_context_mode": args.service_context,
                    }
                    previous_action = action
                    step += 1
                    total_steps += 1

                action_sent_wall = time.time()
                interface.PySendBegin()
                response = interface.GetPy2CppStruct()
                response.contract_version = OBSERVATION_CONTRACT_VERSION
                response.decision_id = decision_id
                response.action = action
                response.macro_id = macro_id_sent
                response.macro_tick = macro_tick_sent
                response.infer_start_wall_s = inference_wall
                response.action_sent_wall_s = action_sent_wall
                interface.PySendEnd()

                # --- inference/control latency measurement ends above -------
                train_stats = None
                train_time_ms = 0.0
                if stored_transition and not args.eval:
                    train_start = time.perf_counter()
                    train_stats = agent.train_lower(rng)
                    if train_stats is not None:
                        train_time_ms = (
                            time.perf_counter() - train_start) * 1000.0
                        metrics.train_time_ms_samples.append(train_time_ms)
                        tracker.note_update(
                            train_stats["td_loss"],
                            train_stats["lower_grad_norm"],
                            bool(train_stats["lower_clipped"]),
                            train_time_ms)
                # The done message replays the last env, so `formal` is still
                # 1 there -- but no action was taken on it. Decaying there put
                # one extra decay into every episode (verified: the checkpoint
                # held decay**251, not decay**250, for a 250-action episode)
                # and, because epsilon carries across episodes, that offset
                # accumulated over the whole run.
                if not done and not args.eval and formal:
                    agent.decay_epsilon()

                if row is not None:
                    diag = agent.diagnostics()
                    row["pending_size"] = diag["pending_size"]
                    row["pending_expired"] = diag["expired_count"]
                    row["pending_overflow"] = diag["overflow_count"]
                    row["unmatched_outcome"] = diag["unmatched_count"]
                    if train_stats is not None:
                        row["td_loss"] = train_stats["td_loss"]
                        row["lower_grad_norm"] = train_stats["lower_grad_norm"]
                        row["grad_clipped"] = train_stats["lower_clipped"]
                        row["train_time_ms"] = train_time_ms
                    if tracker.last_macro is not None:
                        row["critic_loss"] = tracker.last_macro["critic_loss"]
                        row["planner_loss"] = tracker.last_macro["planner_loss"]
                        row["planner_grad_norm"] = tracker.last_macro[
                            "planner_grad_norm"]
                        row["critic_grad_norm"] = tracker.last_macro[
                            "critic_grad_norm"]
                    logger.step.write(row)

                if done:
                    # The exploration schedule is derived from a MIRROR of the
                    # ns-3 decision schedule, so verify the mirror rather than
                    # trusting it: a mismatch means E on the two sides differs
                    # and every derived epsilon value is wrong.
                    if formal_steps != expected_n_formal:
                        raise SystemExit(
                            f"formal action count mismatch: ns-3 produced "
                            f"{formal_steps}, the schedule derived from "
                            f"E={schedule['sim_time_E']}s expects "
                            f"{expected_n_formal}. Pass --sim-time equal to the "
                            "ns-3 --simTime.")
                    # Episode-terminal macros: the episode genuinely ends, so
                    # every macro still awaiting cohorts is pushed with
                    # done=True and H_terminal ticks, or discarded if no cohort
                    # of it ever matured. Nothing is fabricated either way.
                    if not args.eval:
                        for record in sorted(pending_macros.values(),
                                             key=lambda r: r.macro_id):
                            try_push_macro(record, force_terminal=True)
                    else:
                        for record in pending_macros.values():
                            if not record.ticks_reported:
                                tracker.macros_discarded += 1
                    pending_macros.clear()
                    if metrics.current_dwell:
                        metrics.dwell_lengths.append(metrics.current_dwell)
                    delays = np.asarray(metrics.delay_samples, dtype=np.float64)
                    diag = agent.diagnostics()
                    summary = tracker.summary()
                    payload = _episode_summary(
                        args, episode, env, metrics, summary, diag, delays,
                        formal_steps, warmup_frames, agent, freeze,
                        absolute_episode, expected_rng_run)
                    logger.summary.write(payload)
                    print(
                        "[GP] ep={} steps={} formal={} warmup={} "
                        "U_qos_cohort={} actions(dup0/dup50/dup100)={}/{}/{} "
                        "macros={} trunc={} macro_upd={} planner_upd={} "
                        "switches={} hold_masked={} dup_rate={:.3f}".format(
                            episode, step, formal_steps, warmup_frames,
                            ("n/a" if summary["mean_cohort_u_qos"] is None
                             else round(summary["mean_cohort_u_qos"], 4)),
                            summary.get("action_dup0_count", 0),
                            summary.get("action_dup50_count", 0),
                            summary.get("action_dup100_count", 0),
                            summary.get("macros_pushed", 0),
                            summary.get("macros_truncated", 0),
                            agent.macro_updates, agent.planner_updates,
                            metrics.switches, metrics.hold_masked_steps,
                            payload["duplication_rate"]),
                        flush=True)
                    break

            if not args.eval:
                agent.save(model_path)
            if args.eval:
                break
    except KeyboardInterrupt:
        if not args.eval:
            agent.save(model_path)
        print("[GP] interrupted by user", flush=True)
    finally:
        logger.close()
        print(f"[GP] stopped episodes={episode} total_steps={total_steps}",
              flush=True)


def _episode_summary(args, episode, env, metrics, summary, diag, delays,
                     formal_steps, warmup_frames, agent, freeze,
                     absolute_episode, expected_rng_run) -> dict:
    logical_tx = int(env.logical_tx_packets)
    payload = {
        "scenario_id": SCENARIO_ID,
        "absolute_episode": absolute_episode,
        "ns3_rng_run": expected_rng_run,
        "episode": episode, "seed": args.seed,
        "goal_planner": args.goal_planner,
        "cumulative_reward": metrics.cumulative_reward,
        "average_throughput_bps": (
            float(np.mean(metrics.throughput_samples))
            if metrics.throughput_samples else 0.0),
        "average_delay_ms": float(np.mean(delays)) if delays.size else 0.0,
        "p95_delay_ms": (float(np.percentile(delays, 95))
                         if delays.size else 0.0),
        "deadline_loss_rate": (float(np.mean(metrics.loss_samples))
                               if metrics.loss_samples else 0.0),
        "outage_duration_ms": 0.0,
        "qos_satisfaction_ratio": (metrics.qos_satisfied / metrics.qos_valid
                                   if metrics.qos_valid else 0.0),
        "switch_count": metrics.switches,
        "pingpong_count": metrics.pingpongs,
        "architecture": args.goal_planner,
        "reward_version": UPPER_OBJECTIVE_VERSION,
        "service_flow_scenario": args.service_flow_scenario,
        "service_context_mode": args.service_context,
        "formal_steps": formal_steps, "warmup_frames": warmup_frames,
        "macro_updates": agent.macro_updates,
        "planner_updates": agent.planner_updates,
        "mean_upper_grad_norm": summary["mean_planner_grad_norm"],
        "mean_planner_grad_norm": summary["mean_planner_grad_norm"],
        "mean_critic_grad_norm": summary["mean_critic_grad_norm"],
        "mean_lower_grad_norm": summary["mean_lower_grad_norm"],
        "mean_total_grad_norm": summary["mean_lower_grad_norm"],
        "mean_train_time_ms": summary["mean_train_time_ms"],
        "mean_recv_wait_ms": _mean_or_none(metrics.recv_wait_ms_samples),
        "pending_expired_count": diag["expired_count"],
        "pending_overflow_count": diag["overflow_count"],
        "unmatched_outcome_count": diag["unmatched_count"],
        "clipped_update_fraction": summary["clipped_update_fraction"],
        "logical_tx_packets": logical_tx,
        "logical_rx_packets": int(env.logical_rx_packets),
        "logical_deadline_success": int(env.logical_deadline_success),
        "logical_deadline_miss": int(env.logical_deadline_miss),
        "attempted_physical_copies_tx": int(env.attempted_physical_copies_tx),
        "successful_physical_copies_tx": int(env.successful_physical_copies_tx),
        "duplicate_copies_rx": int(env.duplicate_copies_rx),
        "duplicate_bytes_rx": int(env.duplicate_bytes_rx),
        "duplicated_logical_packets": int(env.duplicated_logical_packets),
        "mean_attempted_copies_per_logical": (
            float(env.attempted_physical_copies_tx) / logical_tx
            if logical_tx > 0 else 0.0),
        "mean_successful_copies_per_logical": (
            float(env.successful_physical_copies_tx) / logical_tx
            if logical_tx > 0 else 0.0),
        "duplication_action_ratio": (float(env.duplicated_logical_packets) / logical_tx
                                     if logical_tx > 0 else 0.0),
        "duplication_rate": (float(env.duplicated_logical_packets) / logical_tx
                             if logical_tx > 0 else 0.0),
        "duplicate_rx_ratio": (
            float(env.duplicate_copies_rx) / float(env.logical_rx_packets)
            if int(env.logical_rx_packets) > 0 else 0.0),
        "duplication_budget": agent.duplication.budget,
        "duplication_window": agent.duplication.window,
        "duplication_dual_eta": agent.duplication.eta,
        "duplication_constraint_enabled": int(agent.duplication.enabled),
        "duplication_window_rate": agent.duplication.window_rate(),
        "duplication_window_full": int(bool(agent.duplication.window_full)),
        "duplication_violation_norm": agent.duplication.violation_norm(),
        "duplication_dual_updates": int(agent.duplication.dual_updates),
        "duplication_lambda": float(agent.duplication.lambda_),
        # ---- hold removal + switching diagnostics ----------------------
        "action_hold_enabled": int(ACTION_HOLD_ENABLED),
        "hold_masked_steps": metrics.hold_masked_steps,
        "switch_rate": (metrics.switches / formal_steps
                        if formal_steps else 0.0),
        "requested_equals_applied_count": metrics.requested_equals_applied,
        "requested_differs_applied_count": metrics.requested_differs_applied,
        "hold_veto_steps": metrics.hold_veto_steps,
        "pending_apply_steps": metrics.pending_apply_steps,
        "mean_action_dwell_ticks": (float(np.mean(metrics.dwell_lengths))
                                    if metrics.dwell_lengths else None),
        "median_action_dwell_ticks": (float(np.median(metrics.dwell_lengths))
                                      if metrics.dwell_lengths else None),
        "max_action_dwell_ticks": (int(max(metrics.dwell_lengths))
                                   if metrics.dwell_lengths else None),
        "action_dwell_samples": len(metrics.dwell_lengths),
        # ---- macro tier / cohort attribution ---------------------------
        "macro_horizon": MACRO_HORIZON,
        "macro_interval_basis": MACRO_INTERVAL_BASIS,
        # ---- goal exploration / two-timescale --------------------------
        "beta": agent.beta,
        "exploration_coverage": (None if agent.beta is None
                                 else coverage_from_beta(agent.beta)),
        "exploration_protocol_version": EXPLORATION_PROTOCOL_VERSION,
        "goal_exploration_mechanism": GOAL_EXPLORATION_MECHANISM,
        "mixing_draws": int(agent.mixing_draws),
        "radius_bracket_low": (None if agent.beta is None
                               else exploration_radius_bracket(agent.beta)[0]),
        "radius_bracket_high": (None if agent.beta is None
                                else exploration_radius_bracket(agent.beta)[1]),
        # ---- exploration schedule ---------------------------------------
        "epsilon_start": float(agent.epsilon_start),
        "epsilon_min": float(agent.epsilon_min),
        "epsilon_decay": float(agent.epsilon_decay),
        "epsilon_decay_mode": agent.epsilon_decay_mode,
        "epsilon_schedule_version": agent.epsilon_schedule_version,
        "epsilon_decay_step": int(agent.epsilon_decay_step),
        "total_epsilon_decay_steps": agent.total_epsilon_decay_steps,
        "epsilon_final": float(agent.epsilon),
        "training_horizon_N": agent.training_horizon_n,
        "sim_time_E": agent.sim_time_e,
        "n_formal": agent.n_formal,
        "planner_update_k": agent.planner_update_k,
        "critic_warmup_w": agent.critic_warmup_w,
        # ---- protocol freeze state -------------------------------------
        "protocol_frozen": int(bool(freeze["frozen"])),
        "run_mode": args.run_mode,
        "unresolved_protocol_parameters": list(freeze["missing"]),
    }
    for key in ("cohorts_seen", "cohorts_dropped_no_macro",
                "macro_valid_ticks", "macro_invalid_ticks", "macros_pushed",
                "macros_truncated", "macros_discarded", "mean_macro_return",
                "mean_macro_ticks", "mean_cohort_u_qos",
                "binding_service_counts", "binding_kpi_counts",
                "dominant_binding_service", "dominant_binding_kpi",
                "mean_service_violations", "mean_goal_distance",
                "max_goal_distance", "median_goal_distance",
                "mean_goal_support_distance_macro",
                "median_goal_support_distance_macro",
                "max_goal_support_distance_macro",
                "mean_goal_support_distance_update",
                "median_goal_support_distance_update",
                "planner_goal_context_spread", "goal_identifiability_rho",
                "binding_tie_count", "mean_q_goal_behaviour",
                "mean_q_goal_planner", "mean_q_goal_range"):
        payload[key] = summary.get(key)
    # The action-count keys are DERIVED from ACTION_NAMES, not spelled out:
    # naming them literally is what let the retired Uu/PC5/Both keys survive a
    # rename and silently blank the duplication columns in every summary CSV.
    for key in (tuple(f"action_{name.lower()}_count" for name in ACTION_NAMES)
                + ("mean_applied_duplication_ratio",
                "mean_r_rel", "mean_r_lat", "mean_r_thr", "mean_r_cost",
                "mean_lower_reward", "mean_upper_reward",
                "mean_viol_delay", "mean_viol_pdr", "mean_viol_throughput",
                "mean_viol_outage")):
        if key in summary:
            payload[key] = summary[key]
    payload["mean_goal_entropy"] = summary.get("goal_entropy")
    payload["min_goal_entropy"] = summary.get("min_goal_entropy")
    payload["max_goal_entropy"] = summary.get("max_goal_entropy")
    for name in GOAL_WEIGHT_NAMES:
        payload[f"mean_{name}"] = summary.get(f"{name}_mean")
        payload[f"std_{name}"] = summary.get(f"{name}_std")
    return payload


def default_output_dir() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return str(Path("results") / "tvt" / stamp)


def _explicitly_set(parser, args, names):
    return [
        name
        for name in names
        if getattr(args, name) != parser.get_default(name)
    ]


def _warn_unused_options(parser, args) -> None:
    """Never silently reinterpret an option that this planner ignores."""
    if args.goal_planner != "transformer":
        for name in _explicitly_set(parser, args, TRANSFORMER_ONLY_OPTIONS):
            print(
                f"[CLI][WARN] --{name.replace('_', '-')} only affects the "
                f"transformer planner and is ignored with "
                f"--goal-planner {args.goal_planner}",
                flush=True)
    if args.goal_planner != "fixed" and _explicitly_set(
            parser, args, ("fixed_goal_weights",)):
        print("[CLI][WARN] --fixed-goal-weights is ignored unless "
              "--goal-planner fixed", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="TVT connectivity control: DDQN baselines and E2E "
                    "Transformer latent goal + DDQN")
    parser.add_argument(
        "--goal-planner", choices=GOAL_PLANNERS, default=DEFAULT_GOAL_PLANNER,
        help="which Goal Planner produces g_t; everything else is identical. "
             "transformer_shuffled / transformer_h1 are temporal-modelling "
             "ablations, not formal campaign configs")
    parser.add_argument(
        "--service-aggregation", choices=SERVICE_AGGREGATION_MODES,
        default=SERVICE_AGGREGATION_MODE,
        help="how the declared requirement's four KPI violations aggregate; "
             "both options are coefficient-free")
    # --- run mode / protocol freeze --------------------------------------
    parser.add_argument(
        "--run-mode", choices=RUN_MODES, default=DEFAULT_RUN_MODE,
        help="'smoke' is a correctness run: unresolved protocol numbers may "
             "be supplied as TEST-ONLY values and the run is recorded as "
             "protocol_frozen=false. 'full' is a campaign run and refuses to "
             "start until every unresolved protocol number is given")

    # --- two-timescale upper tier (UNRESOLVED: no defaults) --------------
    # lr_planner == lr_critic is a structural invariant, so --planner-learning-
    # rate is RETIRED: the timescale separation is expressed only by K and W.
    parser.add_argument(
        "--planner-update-k", type=int, default=None,
        help="K: the upper critic updates every macro transition, the planner "
             "once every K of them. UNRESOLVED protocol parameter -- there is "
             "no default, and full training refuses to start without it")
    parser.add_argument(
        "--critic-warmup-w", type=int, default=None,
        help="W: critic updates completed before the planner's first update, "
             "so the planner never climbs an untrained critic. UNRESOLVED "
             "protocol parameter -- no default")

    # --- goal exploration: simplex Dirichlet mixing (UNRESOLVED) ---------
    parser.add_argument(
        "--beta", type=float, default=None,
        help="goal-space exploration budget: g_beh = (1-beta) g_det + beta u "
             "with u ~ Dir(1,1,1,1). An exogenous budget, not a tuned "
             "hyperparameter -- no default. Prefer --exploration-coverage, "
             "which computes it from the guaranteed coverage fraction")
    parser.add_argument(
        "--p-global", type=float, default=None,
        help="probability that a behaviour goal is drawn WITHOUT reference to "
             "the planner (GLOBAL simplex exploration, the U8 fix). Separate "
             "from beta, which stays the LOCAL budget. UNRESOLVED -- selected "
             "by context-conditional coverage, never by performance.")
    parser.add_argument(
        "--exploration-coverage", type=float, default=None,
        help="guaranteed RMS coverage as a fraction of the centroid-to-vertex "
             "distance; beta is computed from it exactly. Mutually exclusive "
             "with --beta")

    # --- duplication constraint (UNRESOLVED: no defaults) ----------------
    parser.add_argument(
        "--duplication-budget", type=float, default=None,
        help="B_dup: allowed fraction of logical packets sent under Both. "
             "UNRESOLVED protocol parameter -- no default. While it is unset "
             "the constraint is inert and J == U_qos")
    parser.add_argument(
        "--duplication-window", type=int, default=None,
        help="sliding-window length (matured cohorts) over which the "
             "duplication rate is measured. UNRESOLVED -- no default")
    parser.add_argument(
        "--duplication-dual-eta", type=float, default=None,
        help="eta: dual ascent step for the duplication Lagrange multiplier. "
             "UNRESOLVED -- no default")
    parser.add_argument(
        "--fixed-goal-weights", type=float, nargs=4,
        default=list(FIXED_GOAL_WEIGHTS),
        metavar=("W_REL", "W_LAT", "W_THR", "W_COST"),
        help="FIXED planner goal vector (default 0.25 0.25 0.25 0.25, the "
             "formal FIXED config); override is for debugging only")
    parser.add_argument("--algorithm", default=ALGORITHM)
    parser.add_argument("--model-path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--episode-offset", type=int, default=0,
        help="episodes already completed before this invocation. A fresh run "
             "passes 0; a resume passes the number of completed episodes so "
             "the episode-specific ns-3 RngRun sequence continues instead of "
             "restarting. Cross-checked against episode_summary.csv.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--shm-segment", default=SHM_SEG_DEFAULT,
        help="ns3-ai shared-memory segment name. INFRASTRUCTURE ONLY: it "
             "isolates concurrent runs and changes no scientific quantity. "
             "The SAME name must be passed to ns-3 via --shmSegment. "
             f"Default {SHM_SEG_DEFAULT!r} reproduces sequential behaviour.")
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="episodes to run in THIS process invocation (resume-friendly). "
             "Distinct from --training-horizon, which is the whole run's "
             "budget. Unlimited (0) is refused in derived-epsilon mode")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--rng-run", type=int, default=None,
        help="ns-3 RngRun recorded for provenance only; it never affects "
             "policy, reward, replay, optimizer or action selection")
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--target-update", type=int, default=TARGET_UPDATE)
    parser.add_argument("--memory-capacity", type=int, default=MEMORY_CAPACITY)
    # --- training budget / exploration schedule --------------------------
    parser.add_argument(
        "--training-horizon", type=int, default=FORMAL_TRAINING_HORIZON,
        help="N: the ENTIRE training run's preregistered fixed optimization "
             "budget, in episodes. Fixes the derived epsilon schedule, the "
             "config hash and checkpoint compatibility. Structural floor 161. "
             "It is not a convergence claim")
    parser.add_argument(
        "--sim-time", type=float, default=FORMAL_EPISODE_LENGTH_S,
        help="E: ns-3 episode length in seconds. Must equal the ns-3 "
             "--simTime; the run verifies the resulting formal action count "
             "and aborts on a mismatch")
    parser.add_argument(
        "--epsilon-decay-mode", choices=EPSILON_DECAY_MODES,
        default=EPSILON_DECAY_MODE,
        help="'derived' computes the decay from (E, N) so exploration cannot "
             "silently change when the budget does; 'fixed' takes "
             "--epsilon-decay verbatim and is for reproducing older runs")
    parser.add_argument(
        "--epsilon-decay", type=float, default=None,
        help="per-step epsilon decay. Only valid with "
             "--epsilon-decay-mode fixed; in derived mode it is refused, "
             "because the schedule follows from the training budget")

    # --- concurrent service context (e2e-transformer) --------------------
    parser.add_argument(
        "--service-flow-scenario", default=DEFAULT_SERVICE_FLOW_SCENARIO,
        help="concurrent flow preset (legacy|video+safety|all) or "
             "'SERVICE@start-end,...'; e2e-transformer only")
    parser.add_argument(
        "--service-context", choices=SERVICE_CONTEXT_MODES,
        default=DEFAULT_SERVICE_CONTEXT_MODE,
        help="'multi' uses the concurrent requirement block, 'none' zeroes it "
             "while keeping the context dimension")

    # --- decision bookkeeping (all learning architectures) ---------------
    parser.add_argument(
        "--pending-max-age-s", type=float, default=PENDING_MAX_AGE_S,
        help="drop pending decisions older than this (simulation seconds)")
    parser.add_argument(
        "--pending-max-entries", type=int, default=PENDING_MAX_ENTRIES,
        help="bound on simultaneously pending decisions")

    # --- E2E network / training ------------------------------------------
    parser.add_argument(
        "--e2e-reward-version", default=DEFAULT_E2E_REWARD_VERSION,
        help="task-level reward version for e2e-transformer and "
             "goal-free-ddqn (initial fixed weighting, not a tuned optimum)")
    parser.add_argument("--history-len", type=int, default=HISTORY_LEN)
    # goal_dim is fixed by the architecture (4 normalized QoS weights)
    parser.add_argument(
        "--d-model", type=int, default=DEFAULT_TRANSFORMER.d_model)
    parser.add_argument("--nhead", type=int, default=DEFAULT_TRANSFORMER.nhead)
    parser.add_argument(
        "--num-layers", type=int, default=DEFAULT_TRANSFORMER.num_layers)
    parser.add_argument(
        "--dim-ff", type=int, default=DEFAULT_TRANSFORMER.dim_feedforward)
    parser.add_argument(
        "--dropout", type=float, default=DEFAULT_TRANSFORMER.dropout)
    parser.add_argument(
        "--no-save-replay", action="store_true",
        help="omit the replay buffer from e2e-transformer checkpoints")
    parser.add_argument(
        "--grad-clip", type=float, default=DEFAULT_GRAD_CLIP,
        help="global L2 gradient-norm clip applied to every learning "
             "architecture; an experimental condition included in the "
             "config hash (1.0 reproduces all pre-P1 runs)")
    parser.add_argument(
        "--train-every", type=int, default=1,
        help="run one update every N stored transitions (1 keeps the "
             "existing schedule); training always runs after the action is "
             "sent, outside the inference latency measurement interval")

    args = parser.parse_args(argv)
    if args.algorithm != ALGORITHM:
        parser.error("--algorithm currently supports only 'ddqn'")
    if args.episodes < 0:
        parser.error("--episodes must be >= 0")
    if args.training_horizon < 1:
        parser.error("--training-horizon must be >= 1")
    if args.sim_time <= 0.0:
        parser.error("--sim-time must be > 0")
    if args.epsilon_decay_mode == "derived":
        if args.epsilon_decay is not None:
            parser.error(
                "--epsilon-decay is not accepted with "
                "--epsilon-decay-mode derived: the decay follows from "
                "(--sim-time, --training-horizon)")
        if args.episodes == 0:
            parser.error(
                "--episodes 0 (unlimited) is not compatible with "
                "--epsilon-decay-mode derived: the schedule needs a finite "
                "training horizon")
        if args.episodes > args.training_horizon:
            parser.error(
                "--episodes exceeds --training-horizon; the horizon is the "
                "whole run's budget, --episodes is this invocation's share")
    elif args.epsilon_decay is None:
        parser.error("--epsilon-decay is required with "
                     "--epsilon-decay-mode fixed")
    try:
        formal_action_count(args.sim_time)
        resolved_epsilon_schedule(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.train_every < 1:
        parser.error("--train-every must be >= 1")
    if args.history_len < 1:
        parser.error("--history-len must be >= 1")

    if args.pending_max_entries < 1:
        parser.error("--pending-max-entries must be >= 1")
    if args.pending_max_age_s <= 0.0:
        parser.error("--pending-max-age-s must be > 0")
    if args.rng_run is not None and args.rng_run < 1:
        parser.error("--rng-run must be >= 1 (ns-3 RngRun starts at 1)")
    if args.episode_offset < 0:
        parser.error("--episode-offset must be >= 0")
    if args.episode_offset + args.episodes >= EPISODE_RNG_STRIDE:
        parser.error(
            f"--episode-offset + --episodes must stay below "
            f"{EPISODE_RNG_STRIDE} (the episode RNG stride)")
    if args.episode_offset > 0 and not args.resume:
        parser.error("--episode-offset > 0 requires --resume")
    if args.grad_clip <= 0.0:
        parser.error("--grad-clip must be > 0")
    if args.planner_update_k is not None and args.planner_update_k < 1:
        parser.error("--planner-update-k must be >= 1")
    if args.critic_warmup_w is not None and args.critic_warmup_w < 0:
        parser.error("--critic-warmup-w must be >= 0")
    if args.p_global is not None and not 0.0 <= args.p_global <= 1.0:
        parser.error("--p-global must lie in [0, 1]")
    if args.beta is not None and args.exploration_coverage is not None:
        parser.error("--beta and --exploration-coverage are mutually exclusive")
    if args.exploration_coverage is not None:
        if args.exploration_coverage <= 0.0:
            parser.error("--exploration-coverage must be > 0")
        args.beta = beta_from_coverage(args.exploration_coverage)
    if args.beta is not None:
        if not 0.0 <= args.beta <= 1.0:
            parser.error("--beta must lie in [0, 1]")
        if args.beta >= PLANNER_DOMINANT_EXPLORATION_POLICY_MAX_BETA:
            parser.error(
                f"--beta must stay below "
                f"{PLANNER_DOMINANT_EXPLORATION_POLICY_MAX_BETA} "
                "(PLANNER_DOMINANT_EXPLORATION_POLICY: the planner must remain "
                "the dominant component of its own behaviour policy)")
    if args.duplication_budget is not None and not 0.0 <= args.duplication_budget <= 1.0:
        parser.error("--duplication-budget must be in [0, 1]")
    if args.duplication_window is not None and args.duplication_window < 1:
        parser.error("--duplication-window must be >= 1")
    if args.duplication_dual_eta is not None and args.duplication_dual_eta <= 0.0:
        parser.error("--duplication-dual-eta must be > 0")
    # The agent cannot run without K and W, and there is deliberately no
    # default: an unfrozen number must never slip into a result.
    if args.planner_update_k is None or args.critic_warmup_w is None:
        parser.error(
            "--planner-update-k and --critic-warmup-w are required. They are "
            "unresolved protocol parameters with no default: pass explicit "
            "TEST-ONLY values for a smoke run, or freeze them for a campaign")
    # Fail fast on an unusable scenario or transformer shape, and report it as
    # a CLI usage error rather than a traceback.
    try:
        parse_service_flow_scenario(args.service_flow_scenario)
        TransformerConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_ff,
            dropout=args.dropout,
        )
    except ValueError as exc:
        parser.error(str(exc))

    # Unresolved scientific parameters must be supplied explicitly, never
    # defaulted into a run that could become a claim.
    if args.fixed_goal_weights is not None:
        weights = [float(w) for w in args.fixed_goal_weights]
        if any(w < 0.0 for w in weights):
            parser.error("--fixed-goal-weights must be >= 0")
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            parser.error(f"--fixed-goal-weights must sum to 1 (got {total})")

    _warn_unused_options(parser, args)
    return args


if __name__ == "__main__":
    run_goal_planner(parse_args())
