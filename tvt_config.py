"""Central experiment definitions for TVT Agentic Connectivity Planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class Goal(IntEnum):
    THROUGHPUT = 0
    RELIABILITY = 1
    LOW_LATENCY = 2
    STABILITY = 3
    RECOVERY = 4


class ServiceType(IntEnum):
    NORMAL = 0
    VIDEO = 1
    SAFETY = 2
    LATENCY_CRITICAL = 3


GOAL_DEFINITION_VERSION = "tvt_goal_v1"
PLANNER_VERSION = "agentic_connectivity_planner_v1"
REWARD_VERSION = "goal_policy_deadline_v1"
CHECKPOINT_SCHEMA_VERSION = 5
OBSERVATION_CONTRACT_VERSION = 8

# Architecture identity. A checkpoint written by the two-action MP-QUIC-era
# code must never load silently into this one; neither must one written before
# the action hold was removed and the macro tier became tick-exact and
# cohort-attributed; and neither must one written under the retired
# {Uu, PC5, Both} path-selection action space, whose Q head had three outputs
# over a categorical path choice rather than five over a duplication degree.
ARCHITECTURE_VERSION = "dynamic_reward_redundancy_controller_v2"
RETIRED_ARCHITECTURE_VERSIONS = (
    "dual_connectivity_goal_planner_v1",
    "dual_connectivity_goal_planner_v2",
    # v1 planners emit raw unbounded logits and collapsed to a simplex vertex
    # with dQ/dz ~ 1e-06. Their weights are not an initialization for the
    # ball-projected parameterization.
    "dynamic_reward_redundancy_controller_v1",
)

# ---------------------------------------------------------------- action hold
# REMOVED FROM THE ARCHITECTURE, not defaulted to zero. The hold-semantics
# audit established that an action change in this simulator is traffic
# steering, not a handover: no attach/detach, no bearer reconfiguration, no
# route change, no socket recreation, no signalling, no queue flush, no
# interruption and no modelled switching latency -- only which already
# established socket carries the next logical packet. With no modelled
# switching cost there is nothing a temporal hold could represent, so imposing
# one would inject a 500 ms handcrafted prior into an experiment about a
# LEARNED strategy (measured: it removed the action choice entirely in 73.8%
# of decisions). Every decision may request any of the three actions.
ACTION_HOLD_ENABLED = False
ACTION_TRANSITION_SEMANTICS = "traffic_steering"
PHYSICAL_SWITCHING_COST_MODELED = False
ACTION_HOLD_STATUS = (
    "removed_v2: RETIRED (LEGACY_ONLY) parameters minRatHoldTimeMs, "
    "ratSwitchHoldMs, duplicationModeHoldMs, remaining_hold_ms. Neither ns-3 "
    "nor the Python runner constrains the action set; requested_action == "
    "applied_action for every well-formed decision. switch_count and "
    "pingpong_count stay measured as secondary diagnostics and are not "
    "objective terms.")

# ---------------------------------------------------------------- actions
# PURE PACKET REDUNDANCY. The lower action is a DUPLICATION DEGREE, not a path
# identity. Uu and PC5 connectivity are both permanently established and the
# path roles are FIXED POLICY:
#
#     primary   = Uu   -- carries every logical packet, always
#     secondary = PC5  -- carries the duplicated fraction
#
# The controller therefore answers "how much redundancy?", never "which path?".
ACTION_SEMANTICS_VERSION = "packet_duplication_degree_v1"
PRIMARY_PATH_POLICY = "fixed_uu_primary_v1"
PRIMARY_PATH_INDEX = 0            # Uu
SECONDARY_PATH_INDEX = 1          # PC5 / RSU relay
PRIMARY_PATH_NAME = "Uu"
SECONDARY_PATH_NAME = "PC5"
PRIMARY_PATH_STATUS = (
    "Fixed transmission semantics for this implementation stage, recorded so "
    "it is an explicit protocol decision rather than an assumption. It is NOT "
    "a claim that Uu is the superior primary; a later sensitivity step may "
    "vary it. No learned primary selection exists.")

ACTION_DUP_0 = 0
ACTION_DUP_25 = 1
ACTION_DUP_50 = 2
ACTION_DUP_75 = 3
ACTION_DUP_100 = 4
ACTION_DIM = 5
ACTION_NAMES = ("dup0", "dup25", "dup50", "dup75", "dup100")

# AUTHORITATIVE action -> duplication ratio table. ns-3 holds the same table in
# TVT_AI_DUPLICATION_LEVELS and a regression compares the two element by
# element against the live binding, so an integer can never mean two things.
DUPLICATION_LEVELS: Tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00)

ACTION_SEMANTICS = {
    index: (f"{DUPLICATION_LEVELS[index]:.0%} of logical packets duplicated: "
            f"one copy on {PRIMARY_PATH_NAME} always, a second copy on "
            f"{SECONDARY_PATH_NAME} for that fraction; redundant duplication "
            f"of the SAME logical packet, not splitting/striping/multipath")
    for index in range(ACTION_DIM)
}

# The action is ORDINAL, not nominal: level 1 is nearer to level 0 than level 4
# is. The observation encodes it as one scalar ratio for exactly that reason,
# rather than as a one-hot that would throw the ordering away.
ACTION_REPRESENTATION = "ordinal_scalar_ratio_v1"


def duplication_ratio(action: int) -> float:
    """Duplication ratio for an action index. Raises on an invalid index."""
    index = int(action)
    if not 0 <= index < ACTION_DIM:
        raise ValueError(f"invalid duplication action: {action}")
    return float(DUPLICATION_LEVELS[index])


def action_from_ratio(ratio: float, tolerance: float = 1e-9) -> int:
    """Inverse of duplication_ratio. Exact match only -- no nearest-level
    snapping, because silently rounding an unrepresentable ratio would hide a
    mismatch between the two sides of the contract."""
    for index, level in enumerate(DUPLICATION_LEVELS):
        if abs(float(ratio) - level) <= tolerance:
            return index
    raise ValueError(f"ratio {ratio!r} is not a duplication level")


# Physical copies transmitted per logical packet, by action, in the LONG RUN.
# This is the protocol definition of the resource cost normaliser; the reward
# uses the MEASURED copy count reported by ns-3, never this table.
ACTION_PHYSICAL_COPIES = {
    index: 1.0 + DUPLICATION_LEVELS[index] for index in range(ACTION_DIM)
}
# Duplication overhead is measured against a single copy: one attempted copy
# per logical packet is the no-overhead baseline, two is full duplication.
RESOURCE_COST_BASELINE_COPIES = 1
# UNCHANGED by the action-space change. copies/logical - 1 already equals the
# realised duplication ratio for any mixture of 1- and 2-copy packets, so the
# formula generalises with no new number and the version must NOT be bumped.
RESOURCE_COST_VERSION = "attempted_duplication_v1"

# The duplication scheduler. Deterministic credit accumulator, never a random
# draw: the ns-3 RNG stream stays untouched, so changing the duplication degree
# cannot perturb the channel or mobility realisation.
DUPLICATION_SCHEDULER = "deterministic_credit_accumulator_v1"
DUPLICATION_SCHEDULER_CONSUMES_RNG = False
# After any number of packets at any sequence of ratios, |duplicates - sum of
# ratios| < 1 packet, because that difference IS the retained credit and the
# credit is held in [0, 1) by construction.
DUPLICATION_ACCUMULATOR_ERROR_BOUND_PACKETS = 1.0

# ------------------------------------------------- throughput semantics
# Two DIFFERENT quantities that contract v6 and earlier conflated.
#
#   PATH-CONDITION OBSERVATION   uu_throughput_bps, pc5_throughput_bps
#       The byte rate each path delivered over the window, duplicate arrivals
#       included. An observation of how a path is doing. Retained in the
#       context, never in a reward or an objective.
#
#   USEFUL LOGICAL GOODPUT       logical_goodput_bps
#       Unique logical application PAYLOAD bits delivered per second. Each
#       sequence number counts at most once however many copies arrived;
#       header bytes and measurement probes are excluded. This is the only
#       quantity a reward or an objective may read as "throughput".
#
# The retired proxy was max(uu_throughput_bps, pc5_throughput_bps). It is
# wrong for two independent reasons, and neither is a scaling issue that a
# reference value could absorb:
#   1. A logical packet is delivered when EITHER path delivers it first, so
#      the real quantity is a UNION over paths. A maximum cannot represent a
#      union whenever the secondary delivers packets the primary lost.
#   2. The per-path rates count duplicate arrivals, which are not goodput.
# It was tempting to argue the two coincide because the primary carries every
# logical packet -- but carrying every packet is not DELIVERING every packet.
# At the measured Uu delivery ratio near 0.61 a large share of deliveries at
# high duplication come from the secondary, and the proxy missed exactly those.
THROUGHPUT_PRIMITIVE_VERSION = "logical_goodput_v1"
# STATED LIMITATION, not a design choice made here. ns-3 measures ONE
# system-wide logical goodput; the ToD stream is a single application flow and
# its sequence numbers carry no service tag. Each service's min_throughput
# requirement is therefore evaluated against that one aggregate. Per-service
# goodput would need per-service sequence spaces in the simulator, which this
# step does not add and must not fake by apportioning the aggregate.
SERVICE_THROUGHPUT_SCOPE = "aggregate_logical_goodput"
SERVICE_THROUGHPUT_SCOPE_NOTE = (
    "One system-wide logical goodput is compared against every active "
    "service's min_throughput. No per-service goodput is measured, and none "
    "is synthesised by splitting the aggregate.")
RETIRED_THROUGHPUT_PRIMITIVE_VERSION = "max_path_throughput_proxy_v1"
THROUGHPUT_PRIMITIVE_SOURCE = "ns3_first_arrival_logical_payload_bytes"
PATH_THROUGHPUT_ROLE = "path_condition_observation_only"
# Measured under the retired proxy at seed 9001, simTime 28, forced levels.
# Preserved so the correction is auditable, not to be re-derived.
RETIRED_THROUGHPUT_PROXY_MEASURED_BIAS = {
    0.00: 0.0000, 0.25: -0.0753, 0.50: -0.1395,
    0.75: -0.1596, 1.00: -0.2153,
}
# RESOLVED in contract v7. The flag stays, reading False, so a reader cannot
# mistake its absence for the question never having been asked.
KNOWN_RISK_USEFUL_THROUGHPUT_PROXY = False
USEFUL_THROUGHPUT_PROXY_RISK = "RESOLVED"
USEFUL_THROUGHPUT_PROXY_NOTE = (
    "RETIRED in contract v7. The reward and the upper objective now read "
    "logical_goodput_bps, measured from ns-3's first-arrival logical payload "
    "accounting. max(uu_bps, pc5_bps) remains available as a path-condition "
    "observation and is never read as throughput by any reward or objective.")

# Screening Protocol v2 selected these under the RETIRED {Uu, PC5, Both} action
# space. The action space changes the lower policy's option set and therefore
# the macro return distribution the upper critic sees, so these values are NOT
# carried over as formal protocol constants. They may be used TEST_ONLY to make
# a smoke run runnable; protocol_frozen must be false whenever they are.
PROVISIONAL_OLD_ACTION_SCREENING_RESULT = {
    "status": "PROVISIONAL_OLD_ACTION_SCREENING_RESULT",
    "beta": 0.11180339887498948,
    "planner_update_k": 2,
    "screened_under_action_space": "uu_pc5_both_v1",
    "screened_under_contract_version": 5,
    "valid_for_current_action_space": False,
    "reason": ("selected on trajectories from a three-way path-selection "
               "action space; the pure-redundancy action space changes the "
               "lower option set and the macro return distribution, so both "
               "must be re-screened before any formal use"),
}

# ---------------------------------------------------------------- timescale
# Macro (goal) tier and micro (action) tier. A goal is held for exactly
# MACRO_HORIZON decision ticks -- a count of ticks, never a count of ticks
# that happened to produce a valid reward. A tick whose upper reward is
# unavailable contributes 0 to the macro return; it does not extend the
# interval, because stretching the interval would decouple the discount
# exponent from the wall-clock duration the goal was actually in force.
DECISION_INTERVAL_MS = 100.0          # micro / action tier (ns-3 owned)
MACRO_HORIZON = 5                     # decision ticks per goal interval
GOAL_INTERVAL_MS = DECISION_INTERVAL_MS * MACRO_HORIZON        # 500 ms
MACRO_INTERVAL_BASIS = "decision_ticks"   # NOT "valid reward steps"
OBSERVATION_WARMUP_STEPS = 10         # == HISTORY_LEN

# ---------------------------------------------------- frozen campaign geometry
# ns-3 owns the decision schedule: decisions start at DECISION_START_S and fire
# every DECISION_INTERVAL_MS; a decision is FORMAL once traffic, reward and KPI
# accounting have all begun together at TRAFFIC_START_S. Mirrored here (not
# guessed) so the Python side can derive the exploration schedule and then
# VERIFY the derivation against the count ns-3 actually produces.
DECISION_START_S = 2.0
TRAFFIC_START_S = DECISION_START_S + OBSERVATION_WARMUP_STEPS * (
    DECISION_INTERVAL_MS / 1000.0)          # = 3.0 s


def formal_action_count(episode_length_s: float) -> int:
    """n_formal(E): formal decision ticks in one episode.

        n_formal(E) = (E - TRAFFIC_START_S) / dt = 10E - 30   for dt = 0.1 s

    Verified against ns-3 at E = 14 s (110), 22 s (190) and 28 s (250).
    This is a MIRROR of the simulator's schedule, so every run re-checks it
    against the observed count and refuses to continue on a mismatch.
    """
    dt = DECISION_INTERVAL_MS / 1000.0
    count = int(round((float(episode_length_s) - TRAFFIC_START_S) / dt))
    if count < 1:
        raise ValueError(
            f"episode length {episode_length_s}s yields no formal decision "
            f"(traffic starts at {TRAFFIC_START_S}s)")
    return count


# FROZEN formal campaign protocol. These are preregistered values, not
# defaults to be revisited: the scenario, the episode length and the training
# horizon were fixed before any screening or training was run.
FORMAL_SERVICE_FLOW_SCENARIO = "video+safety"
FORMAL_EPISODE_LENGTH_S = 28.0
FORMAL_MACROS_PER_EPISODE = 50            # floor(2E - 6)
# A preregistered fixed optimization budget -- the minimum structurally
# admissible one. It is NOT a convergence claim: see the wording rules below.
FORMAL_TRAINING_HORIZON = 161
FORMAL_CRITIC_WARMUP_W = 19               # M_ep - (macro_batch_size - 1)
TRAINING_HORIZON_STATUS = (
    "preregistered fixed optimization budget. N = 161 is the smallest horizon "
    "satisfying every known structural lower bound: one full turnover of both "
    "replay buffers (>= 160) and, at the largest admissible planner update "
    "ratio K = 8, the planner optimizer's second-moment horizon (>= 161). It "
    "carries NO claim of policy convergence, critic convergence, an optimal "
    "horizon, or performance saturation.")
# Wording that must never appear about the training horizon.
FORBIDDEN_HORIZON_WORDING = (
    "policy convergence",
    "training converged",
    "sufficient for convergence",
    "fully trained",
)
HORIZON_SENSITIVITY_ARM = "NONE"          # not a factor in the primary protocol

# ---------------------------------------------------------------- exploration
# Lower-tier epsilon-greedy over {Uu, PC5, Both}. The schedule is DERIVED from
# the training budget rather than carried as an inherited constant, so that
# changing E or N cannot silently change the exploration schedule:
#
#   n_formal(E)        formal actions per episode
#   D_action(E, N)     = N * n_formal(E) - 1
#                        decays executed strictly BEFORE the run's last
#                        formal action (the code decays AFTER acting)
#   epsilon_decay      = (EPSILON_MIN / EPSILON_START) ** (1 / D_action)
#
# so the FIRST formal action of the run uses EPSILON_START exactly and the
# LAST formal action uses EPSILON_MIN exactly. Reaching EPSILON_MIN is
# "exploration schedule completion" / "prescribed epsilon annealing
# completion" -- never "convergence".
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_MODES = ("derived", "fixed")
EPSILON_DECAY_MODE = "derived"
EPSILON_SCHEDULE_VERSION = "annealing_completes_at_horizon_v1"
EPSILON_DECAY_RULE = "(EPSILON_MIN/EPSILON_START) ** (1 / (N * n_formal(E) - 1))"


def epsilon_decay_steps(episode_length_s: float, training_horizon: int) -> int:
    """D_action: decays that precede the run's LAST formal action.

    The runner decays after acting, so action k uses
    EPSILON_START * decay**(k-1). Anchoring the schedule on the number of
    decays before the final action is what makes EPSILON_MIN a value an action
    actually uses; anchoring on the total decay count would only reach it
    after the last action, i.e. never.
    """
    horizon = int(training_horizon)
    if horizon < 1:
        raise ValueError("training horizon must be >= 1")
    steps = formal_action_count(episode_length_s) * horizon - 1
    if steps < 1:
        raise ValueError("training budget yields no epsilon decay step")
    return steps


# ------------------------------------------- N > 161 EPSILON EXTENSION POLICY
# FROZEN in PHASE D-1E1, BEFORE any D-1E outcome and before any critic-maturity
# evidence existed. Purpose: a future maturity result must not be able to
# change the historical exploration trajectory.
#
# FORMAL_TRAINING_HORIZON is a FLOOR (>= 161), so a later campaign may legally
# use N > 161. Deriving the decay from that larger N would stretch the whole
# annealing trajectory and make every epsilon-dependent result incomparable
# across the boundary -- and that decision would be taken in light of an
# outcome. Instead the annealing budget is pinned, in ABSOLUTE formal
# lower-action steps, to the value the preregistered N = 161 campaign already
# implies:
#
#     EPSILON_ANNEALING_STEPS = n_formal(28.0) * 161 - 1 = 250 * 161 - 1 = 40249
#
# N therefore controls how long training continues; it does NOT stretch the
# exploration schedule. Past step 40249 the existing
# max(EPSILON_MIN, ...) clamp in NrAgent.decay_epsilon holds epsilon at
# EPSILON_MIN exactly, with no further decay.
#
# This is NOT epsilon tuning: for every N = 161 run the trajectory over steps
# 0..40249 is bit-identical to the previous derived-from-N implementation.
EPSILON_ANNEALING_STEPS = 40249
EPSILON_EXTENSION_POLICY = "absolute_steps_frozen_at_40249_v1"


def frozen_epsilon_annealing_steps(episode_length_s: float,
                                   training_horizon: int) -> int:
    """Annealing budget under the frozen extension policy.

    Identical to epsilon_decay_steps() whenever the derived budget does not
    exceed the frozen one, which is the case for every admissible formal
    campaign (E = 28.0, N = 161). For N > 161 the frozen budget binds, so the
    trajectory is NOT stretched.
    """
    derived = epsilon_decay_steps(episode_length_s, training_horizon)
    return min(derived, EPSILON_ANNEALING_STEPS)


def frozen_epsilon_decay(episode_length_s: float, training_horizon: int,
                         epsilon_start: float = EPSILON_START,
                         epsilon_min: float = EPSILON_MIN) -> float:
    """Per-step decay implied by the frozen annealing budget."""
    if not 0.0 < epsilon_min <= epsilon_start:
        raise ValueError("require 0 < epsilon_min <= epsilon_start")
    return float((float(epsilon_min) / float(epsilon_start))
                 ** (1.0 / frozen_epsilon_annealing_steps(episode_length_s,
                                                          training_horizon)))


class InvocationBudgetError(RuntimeError):
    """Raised when an invocation would run past the training horizon."""


def validate_invocation_budget(completed: int, requested: int,
                               horizon: int) -> int:
    """Refuse an invocation that would run past the preregistered horizon.

    The horizon is a budget, not a suggestion. Running past it would carry
    epsilon below epsilon_min for a stretch the derived schedule never
    accounted for, and would quietly turn a fixed optimization budget into an
    open-ended one. Returns the resulting episode total when the invocation
    fits.
    """
    completed, requested, horizon = int(completed), int(requested), int(horizon)
    if requested < 1:
        raise InvocationBudgetError("an invocation must run >= 1 episode")
    if completed < 0:
        raise InvocationBudgetError("completed episodes cannot be negative")
    total = completed + requested
    if total > horizon:
        raise InvocationBudgetError(
            f"this invocation would run {total} episodes of a {horizon}-episode "
            f"training horizon ({completed} already completed). The horizon is "
            "a preregistered fixed optimization budget.")
    return total


def derived_epsilon_decay(
    episode_length_s: float,
    training_horizon: int,
    epsilon_start: float = EPSILON_START,
    epsilon_min: float = EPSILON_MIN,
) -> float:
    """Per-step multiplicative decay implied by the training budget."""
    if not 0.0 < epsilon_min <= epsilon_start:
        raise ValueError("require 0 < epsilon_min <= epsilon_start")
    return float((float(epsilon_min) / float(epsilon_start))
                 ** (1.0 / epsilon_decay_steps(episode_length_s,
                                               training_horizon)))

# Upper (macro) Bellman, stated once so the implementation cannot drift:
#   R_macro_k = sum_{j=0..H-1} gamma^j * r_upper(tick_j)
#               r_upper(tick_j) = 0 when tick j has no valid upper reward
#   y_goal    = R_macro_k + gamma^H * Q_goal_target(c_{k+1}, pi_goal_target(c_{k+1}))
# R_macro is discounted WITHIN the interval starting at gamma^0, so bootstrapping
# with gamma^H continues the same series exactly. No double discounting, and the
# exponent H always equals the number of ticks the goal was held.
MACRO_RETURN_CONVENTION = "discounted_within_interval_gamma_pow_H_bootstrap"
# Episode-terminal macros: the episode genuinely ends, so the truncated macro
# is pushed with done=True (bootstrap 0) and micro_steps = H_terminal, the
# number of ticks whose cohort actually matured. Discount exponent and term
# count therefore always agree, in the truncated case as in the full one.
MACRO_TERMINAL_CONVENTION = "actual_H_terminal_with_zero_bootstrap"

# ---------------------------------------------------------- cohort attribution
# A deadline outcome ripens ~250 ms after transmission. Crediting it to the
# macro interval that happens to be open when it ripens attributes an outcome
# to a goal that did not cause it. ns-3 therefore tags every logical packet
# with the decision tick (and hence the macro interval) that TRANSMITTED it,
# and publishes a tick cohort only once every one of its packets has reached
# its deadline. Throughput and outage are snapshotted over exactly the window
# the cohort was transmitted in, so all four KPIs describe the same decision.
RELIABILITY_ATTRIBUTION_VERSION = "originating_macro_cohort_v1"
DEADLINE_MS = 250.0                   # matches the ns-3 application deadline

# legacy goal-enum state layout (kept only for the retired two-action agent)
GOAL_COUNT = len(Goal)

NORMALIZATION = {
    "rssi_min_dbm": -140.0,
    "rssi_max_dbm": -30.0,
    "throughput_ref_bps": 4_000_000.0,
    "delay_ref_ms": 250.0,
    "loss_ref_rate": 1.0,
    "outage_ref_ms": 1_000.0,
    "pingpong_window_ms": 1_000.0,
    # ---- contract-8 mobility context c_t -------------------------------
    # Both references are SCENARIO DESIGN constants that were frozen before
    # this phase. Neither was selected from QoS, reward, a learned policy, a
    # context x goal result or any other outcome.
    #
    #   speed_ref_mps = 14.7
    #       kActiveVehSpeedMax (tvt_ns3.cc:291) -- the scenario's declared
    #       maximum vehicle speed. The active-vehicle band is drawn from
    #       UniformRandomVariable[kActiveVehSpeedMin=5.0, kActiveVehSpeedMax]
    #       on stream kStreamVehSpeed (tvt_ns3.cc:3846-3848). Normalising by
    #       the declared ceiling makes ego_speed_norm == 1.0 exactly at the
    #       scenario's design speed.
    #
    #   distance_ref_m = 300.0
    #       the scenario's declared infrastructure-proximity scale, which
    #       appears twice independently: the gNB is deliberately placed 300 m
    #       south of the road centre (gnbY 6297.4 vs road centre 6597.4,
    #       tvt_ns3.cc:3523/3601/3925), and the scenario's own GEOMETRY-PROBE
    #       classifies "near" as d <= 300.0 m (tvt_ns3.cc:5590).
    #
    # clip01 saturates above these references by construction. That is a
    # consequence of a physically anchored scale, not something to be tuned
    # away from observed ranges.
    "speed_ref_mps": 14.7,
    "distance_ref_m": 300.0,
}

# Initial exploratory thresholds are tied to the 4 Mbps ToD load and 250 ms KPI.
# They must be calibrated with multi-seed baseline distributions before paper runs.
QOS_THRESHOLDS = {
    ServiceType.VIDEO: {
        "min_throughput_bps": 2_000_000.0,
        "max_delay_ms": 250.0,
        "min_pdr": 0.90,
        "max_outage_ms": 500.0,
    },
    ServiceType.SAFETY: {
        "min_throughput_bps": 500_000.0,
        "max_delay_ms": 100.0,
        "min_pdr": 0.99,
        "max_outage_ms": 200.0,
    },
    ServiceType.NORMAL: {
        "min_throughput_bps": 1_000_000.0,
        "max_delay_ms": 250.0,
        "min_pdr": 0.90,
        "max_outage_ms": 500.0,
    },
    ServiceType.LATENCY_CRITICAL: {
        "min_throughput_bps": 500_000.0,
        "max_delay_ms": 50.0,
        "min_pdr": 0.99,
        "max_outage_ms": 100.0,
    },
}


@dataclass(frozen=True)
class RewardPolicy:
    policy_id: str
    throughput: float
    delay: float
    loss: float
    outage: float
    switch: float
    pingpong: float


REWARD_POLICIES: Dict[Goal, RewardPolicy] = {
    Goal.THROUGHPUT: RewardPolicy(
        "throughput_v1", 0.60, 0.15, 0.15, 0.05, 0.03, 0.02),
    Goal.RELIABILITY: RewardPolicy(
        "reliability_v1", 0.10, 0.15, 0.45, 0.20, 0.05, 0.05),
    Goal.LOW_LATENCY: RewardPolicy(
        "low_latency_v1", 0.10, 0.55, 0.15, 0.10, 0.05, 0.05),
    Goal.STABILITY: RewardPolicy(
        "stability_v1", 0.15, 0.15, 0.15, 0.10, 0.25, 0.20),
    Goal.RECOVERY: RewardPolicy(
        "recovery_v1", 0.15, 0.10, 0.25, 0.30, 0.10, 0.10),
}


@dataclass(frozen=True)
class PlannerThresholds:
    transient_degradation_ms: float = 500.0
    persistent_degradation_ms: float = 1_000.0
    outage_threshold_ms: float = 500.0
    rssi_degraded_dbm: float = -100.0
    loss_degraded_rate: float = 0.30


@dataclass(frozen=True)
class ServicePeriod:
    start_s: float
    end_s: float
    service: ServiceType


DEFAULT_SERVICE_TIMELINE = [
    ServicePeriod(0.0, 15.0, ServiceType.VIDEO),
    ServicePeriod(15.0, 25.0, ServiceType.SAFETY),
    ServicePeriod(25.0, 45.0, ServiceType.NORMAL),
    ServicePeriod(45.0, 60.0, ServiceType.LATENCY_CRITICAL),
]


def goal_one_hot(goal: Goal | int):
    import numpy as np

    try:
        parsed = Goal(int(goal))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Goal id: {goal}") from exc
    values = np.zeros(GOAL_COUNT, dtype=np.float32)
    values[int(parsed)] = 1.0
    if values.sum() != 1.0:
        raise AssertionError("Goal one-hot must contain exactly one active element")
    return values


def service_for_time(timeline: Iterable[ServicePeriod], simulation_time_s: float) -> ServiceType:
    periods = list(timeline)
    if not periods:
        raise ValueError("service timeline must not be empty")
    for period in periods:
        if period.start_s <= simulation_time_s < period.end_s:
            return period.service
    return periods[-1].service


def parse_service_scenario(spec: str) -> List[ServicePeriod]:
    presets = {
        "default": DEFAULT_SERVICE_TIMELINE,
        "normal": [ServicePeriod(0.0, float("inf"), ServiceType.NORMAL)],
        "video": [ServicePeriod(0.0, float("inf"), ServiceType.VIDEO)],
        "safety": [ServicePeriod(0.0, float("inf"), ServiceType.SAFETY)],
        "latency": [
            ServicePeriod(0.0, float("inf"), ServiceType.LATENCY_CRITICAL)
        ],
    }
    key = spec.strip().lower()
    if key in presets:
        return list(presets[key])

    periods: List[ServicePeriod] = []
    for item in spec.split(","):
        try:
            interval, service_name = item.split(":", maxsplit=1)
            start, end = interval.split("-", maxsplit=1)
            service = ServiceType[service_name.strip().upper()]
            periods.append(ServicePeriod(float(start), float(end), service))
        except (ValueError, KeyError) as exc:
            raise ValueError(
                "service scenario must be a preset or "
                "'start-end:SERVICE,...'"
            ) from exc
    periods.sort(key=lambda period: period.start_s)
    if not periods or any(period.end_s <= period.start_s for period in periods):
        raise ValueError("service periods must have positive duration")
    for previous, current in zip(periods, periods[1:]):
        if current.start_s < previous.end_s:
            raise ValueError("service periods must not overlap")
    return periods


def serializable_configuration():
    return {
        "goal_definition_version": GOAL_DEFINITION_VERSION,
        "planner_version": PLANNER_VERSION,
        "reward_version": REWARD_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "vehicle_context_dim": VEHICLE_CONTEXT_DIM,
        "vehicle_context_feature_names": list(VEHICLE_CONTEXT_FEATURE_NAMES),
        "context_block_order": list(CONTEXT_BLOCK_ORDER),
        "context_block_slices": {
            "c_t": [C_SLICE.start, C_SLICE.stop],
            "v_t": [V_SLICE.start, V_SLICE.stop],
            "n_t": [N_SLICE.start, N_SLICE.stop],
        },
        "architecture_version": ARCHITECTURE_VERSION,
        "action_dim": ACTION_DIM,
        "action_names": list(ACTION_NAMES),
        "action_semantics": ACTION_SEMANTICS_VERSION,
        "action_semantics_detail": {
            int(k): v for k, v in ACTION_SEMANTICS.items()},
        "action_representation": ACTION_REPRESENTATION,
        "duplication_levels": list(DUPLICATION_LEVELS),
        "duplication_scheduler": DUPLICATION_SCHEDULER,
        "duplication_scheduler_consumes_rng":
            DUPLICATION_SCHEDULER_CONSUMES_RNG,
        "primary_path_policy": PRIMARY_PATH_POLICY,
        "planner_parameterization": PLANNER_PARAMETERIZATION,
        "planner_logit_bound": PLANNER_LOGIT_BOUND,
        "e2e_timescale_protocol": E2E_TIMESCALE_PROTOCOL,
        "planner_lr_scaled_by_k": PLANNER_LR_SCALED_BY_K,
        "exact_resume_required_state": list(EXACT_RESUME_REQUIRED_STATE),
        "exact_resume_status": EXACT_RESUME_STATUS,
        "e2e_timescale_protocol_status": E2E_TIMESCALE_PROTOCOL_STATUS,
        "planner_max_representable_dominance":
            PLANNER_MAX_REPRESENTABLE_DOMINANCE,
        "planner_min_representable_component":
            PLANNER_MIN_REPRESENTABLE_COMPONENT,
        "goal_exploration_modes": list(GOAL_EXPLORATION_MODES),
        "p_global_candidates": list(P_GLOBAL_CANDIDATES),
        "p_global_lower_bound": P_GLOBAL_LOWER_BOUND,
        "p_global_upper_bound": P_GLOBAL_UPPER_BOUND,
        "p_global_status": P_GLOBAL_STATUS,
        "primary_path": PRIMARY_PATH_NAME,
        "secondary_path": SECONDARY_PATH_NAME,
        "primary_path_status": PRIMARY_PATH_STATUS,
        "retired_network_context_features":
            list(RETIRED_NETWORK_CONTEXT_FEATURES),
        "throughput_primitive_version": THROUGHPUT_PRIMITIVE_VERSION,
        "throughput_primitive_source": THROUGHPUT_PRIMITIVE_SOURCE,
        "retired_throughput_primitive_version":
            RETIRED_THROUGHPUT_PRIMITIVE_VERSION,
        "path_throughput_role": PATH_THROUGHPUT_ROLE,
        "known_risk_useful_throughput_proxy":
            KNOWN_RISK_USEFUL_THROUGHPUT_PROXY,
        "useful_throughput_proxy_risk": USEFUL_THROUGHPUT_PROXY_RISK,
        "useful_throughput_proxy_note": USEFUL_THROUGHPUT_PROXY_NOTE,
        "retired_throughput_proxy_measured_bias": dict(
            RETIRED_THROUGHPUT_PROXY_MEASURED_BIAS),
        "provisional_old_action_screening": dict(
            PROVISIONAL_OLD_ACTION_SCREENING_RESULT),
        "goal_dim": GOAL_DIM,
        "goal_weight_names": list(GOAL_WEIGHT_NAMES),
        "goal_semantics": dict(GOAL_SEMANTICS),
        "mpquic_enabled": False,
        "lower_reward_version": LOWER_REWARD_VERSION,
        "lower_reward_components": list(LOWER_REWARD_COMPONENTS),
        "reward_normalization_notes": dict(REWARD_NORMALIZATION_NOTES),
        "upper_objective_version": UPPER_OBJECTIVE_VERSION,
        "goal_critic_operator": GOAL_CRITIC_OPERATOR,
        "upper_objective_input": UPPER_OBJECTIVE_INPUT,
        "upper_objective_uses_effective_requirement":
            UPPER_OBJECTIVE_USES_EFFECTIVE_REQUIREMENT,
        "service_aggregation_mode": SERVICE_AGGREGATION_MODE,
        "system_aggregation_mode": SYSTEM_AGGREGATION_MODE,
        "resource_constraint_version": RESOURCE_CONSTRAINT_VERSION,
        "duplication_budget": DUPLICATION_BUDGET,
        "duplication_window": DUPLICATION_WINDOW,
        "duplication_dual_eta": DUPLICATION_DUAL_ETA,
        "goal_exploration_version": GOAL_EXPLORATION_VERSION,
        "exploration_protocol_version": EXPLORATION_PROTOCOL_VERSION,
        "goal_exploration_mechanism": GOAL_EXPLORATION_MECHANISM,
        "retired_goal_exploration_version": RETIRED_GOAL_EXPLORATION_VERSION,
        "beta": BETA,
        "beta_selection_type": BETA_SELECTION_TYPE,
        "beta_coverage_levels": list(BETA_COVERAGE_LEVELS),
        "coverage_per_beta": COVERAGE_PER_BETA,
        "planner_dominant_max_beta": PLANNER_DOMINANT_EXPLORATION_POLICY_MAX_BETA,
        "dirichlet_trace_cov": DIRICHLET_TRACE_COV,
        "exploration_radius_bracket_factors": [
            EXPLORATION_RADIUS_MIN_FACTOR, EXPLORATION_RADIUS_MAX_FACTOR],
        "screening_protocol_version": SCREENING_PROTOCOL_VERSION,
        "screening_protocol_v1_record": SCREENING_PROTOCOL_V1_RECORD,
        "screening_harvest_episodes": SCREENING_HARVEST_EPISODES,
        "k_stability_bound_version": K_STABILITY_BOUND_VERSION,
        "two_timescale_version": TWO_TIMESCALE_VERSION,
        "planner_lr_equals_critic_lr": PLANNER_LR_EQUALS_CRITIC_LR,
        "action_hold_enabled": ACTION_HOLD_ENABLED,
        "action_transition_semantics": ACTION_TRANSITION_SEMANTICS,
        "physical_switching_cost_modeled": PHYSICAL_SWITCHING_COST_MODELED,
        "action_hold_status": ACTION_HOLD_STATUS,
        "macro_horizon": MACRO_HORIZON,
        "macro_interval_basis": MACRO_INTERVAL_BASIS,
        "goal_interval_ms": GOAL_INTERVAL_MS,
        "decision_interval_ms": DECISION_INTERVAL_MS,
        "macro_return_convention": MACRO_RETURN_CONVENTION,
        "macro_terminal_convention": MACRO_TERMINAL_CONVENTION,
        "reliability_attribution_version": RELIABILITY_ATTRIBUTION_VERSION,
        "formal_service_flow_scenario": FORMAL_SERVICE_FLOW_SCENARIO,
        "formal_episode_length_s": FORMAL_EPISODE_LENGTH_S,
        "formal_macros_per_episode": FORMAL_MACROS_PER_EPISODE,
        "formal_training_horizon": FORMAL_TRAINING_HORIZON,
        "formal_critic_warmup_w": FORMAL_CRITIC_WARMUP_W,
        "training_horizon_status": TRAINING_HORIZON_STATUS,
        "horizon_sensitivity_arm": HORIZON_SENSITIVITY_ARM,
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay_mode": EPSILON_DECAY_MODE,
        "epsilon_schedule_version": EPSILON_SCHEDULE_VERSION,
        "epsilon_decay_rule": EPSILON_DECAY_RULE,
        "duplication_partial_window_policy": DUPLICATION_PARTIAL_WINDOW_POLICY,
        "duplication_violation_normalized": DUPLICATION_VIOLATION_NORMALIZED,
        "duplication_dual_cadence": DUPLICATION_DUAL_CADENCE,
        "binding_tie_policy": BINDING_TIE_POLICY,
        "unresolved_parameters": {
            "names": list(UNRESOLVED_PROTOCOL_PARAMETERS),
            "duplication": RESOURCE_CONSTRAINT_STATUS,
            "beta": BETA_STATUS,
            "two_timescale": TWO_TIMESCALE_STATUS,
        },
        "upper_objective_includes_switching": UPPER_OBJECTIVE_INCLUDES_SWITCHING,
        "resource_cost_version": RESOURCE_COST_VERSION,
        "fixed_goal_weights": list(FIXED_GOAL_WEIGHTS),
        "rule_goal_mapping_version": RULE_GOAL_MAPPING_VERSION,
        "legacy_task_reward_version": LEGACY_TASK_REWARD_VERSION,
        "goals": {goal.name: int(goal) for goal in Goal},
        "services": {service.name: int(service) for service in ServiceType},
        "normalization": dict(NORMALIZATION),
        "qos_thresholds": {
            service.name: values for service, values in QOS_THRESHOLDS.items()
        },
        "reward_policies": {
            goal.name: asdict(policy) for goal, policy in REWARD_POLICIES.items()
        },
        "default_service_timeline": [
            {**asdict(period), "service": period.service.name}
            for period in DEFAULT_SERVICE_TIMELINE
        ],
        "planner_thresholds": asdict(PlannerThresholds()),
    }


# ============================================================================
# E2E extension: Transformer latent Connectivity Goal + DDQN
# ----------------------------------------------------------------------------
# Append-only. Every symbol above this banner keeps its original value and
# semantics so the goal-ddqn baseline stays bit-compatible.
#
# Research framing (fixed wording): this block encodes "heterogeneous
# concurrent QoS requirements over a shared, RAT-switchable bearer".
# ns-3 measures one KPI cohort (the ToD flow), so the service block below is a
# concurrent *requirement declaration*, never a per-flow measurement. Do not
# describe it as "concurrent measured flows" or "per-flow QoS optimization".
# ============================================================================

# ---------------------------------------------------------------- variants
# Only the Goal Planner differs between these four. The ns-3 environment, the
# state features, the lower DDQN, the action space, the reward components, the
# fixed upper objective, the optimizer, lr, grad_clip, horizon and seeds are
# identical by construction.
GOAL_PLANNERS = ("fixed", "rule", "mlp", "transformer",
                 # temporal-modelling ablations: NOT formal campaign configs
                 "transformer_shuffled", "transformer_h1")
FORMAL_GOAL_PLANNERS = ("fixed", "rule", "mlp", "transformer")
TEMPORAL_ABLATION_PLANNERS = ("transformer", "transformer_shuffled",
                              "transformer_h1")
DEFAULT_GOAL_PLANNER = "transformer"
GOAL_PLANNER_LABELS = {
    "fixed": "FIXED",
    "rule": "RULE",
    "mlp": "MLP",
    "transformer": "TRANSFORMER",
    "transformer_shuffled": "TRANSFORMER_SHUFFLED",
    "transformer_h1": "TRANSFORMER_H1",
}
# Deliberately NOT reusing the retired C/D/E/F1/F2 names: those identify the
# two-action MP-QUIC campaign and must not collide with this one.
ARCHITECTURES = GOAL_PLANNERS
DEFAULT_ARCHITECTURE = DEFAULT_GOAL_PLANNER

E2E_ARCHITECTURE = "dual_connectivity_goal_planner_v1"
# v8: exact-resume state. A checkpoint now carries the lower replay, the
#     macro replay, the exploration RNG, the replay/action RNG and the torch
#     RNG, because an audit found that restoring weights alone produced a NEW
#     trajectory (empty replays, rewound exploration stream) while reporting
#     itself as a continuation.
E2E_CHECKPOINT_SCHEMA_VERSION = 8
EXACT_RESUME_REQUIRED_STATE = (
    "replay", "macro_replay", "mixing_rng_state",
    "action_rng_state", "torch_rng_state")
EXACT_RESUME_STATUS = (
    "Training resume demands schema 8 state. Evaluation and offline analysis "
    "still accept legacy checkpoints, so the STEP 4B/4C evidence stays "
    "readable.")
E2E_REWARD_VERSION = "task_level_e2e_v1"

# Training runs outside the inference latency measurement interval
# (infer_start_wall_s .. action_sent_wall_s). This separates the *measurement*
# only; it does not mean training runs in parallel with the ns-3 simulation.
# The handshake stays lock-step, so training can still extend total simulation
# wall-clock progression.
E2E_TIMING_POLICY = "training_after_send_v1"

# ---------------------------------------------------------------- dimensions
HISTORY_LEN = 10

NETWORK_CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    # per-path radio state (both paths, always observable)
    "uu_rssi_norm",              # 0
    "pc5_rssi_norm",             # 1
    # COMMANDED duplication degree currently in force, as an ORDINAL SCALAR in
    # {0, .25, .5, .75, 1}. It replaces the retired action_uu/action_pc5/
    # action_both one-hot: the action is a control level with an order and a
    # distance, and a one-hot would discard both -- it would make 0 -> .25 look
    # exactly as far as 0 -> 1.
    "applied_duplication_ratio",  # 2
    # aggregate task KPI (logical, duplicate-suppressed)
    "delay_norm",                # 3
    "loss_norm",                 # 4
    # per-path measurement, kept fresh regardless of the degree in force. Both
    # paths stay observable although the controller no longer chooses between
    # them: deciding whether a second copy is worth it requires the secondary
    # path's state.
    "uu_thpt_norm",              # 5
    "pc5_thpt_norm",             # 6
    "uu_delay_norm",             # 7
    "pc5_delay_norm",            # 8
    "uu_loss_rate",              # 9
    "pc5_loss_rate",             # 10
    "uu_meas_valid",             # 11
    "pc5_meas_valid",            # 12
    # control / validity
    "rx_gap_norm",               # 13
    "delay_valid",               # 14
    "deadline_loss_valid",       # 15
    # MEASURED duplication: attempted copies above one copy per logical packet.
    # Distinct from applied_duplication_ratio, which is what was COMMANDED: the
    # two differ by the accumulator's sub-packet quantisation and by any send
    # the socket refused.
    "duplication_overhead",      # 16
    # measurement freshness: how long ago each path was last measured
    "uu_measurement_age_norm",   # 17
    "pc5_measurement_age_norm",  # 18
)
# RETIRED from the network context with the pure-redundancy action space:
#   action_uu, action_pc5, action_both
#       nominal one-hot of a path choice that no longer exists.
#   is_switch
#       "the path changed" has no meaning when the path never changes. The
#       duplication-level analogue is kept, but as a LOGGED DIAGNOSTIC ONLY
#       (see DUPLICATION_CHANGE_DIAGNOSTICS) and never as a state feature or a
#       reward term, so nothing in the objective can reward or punish moving
#       between levels.
RETIRED_NETWORK_CONTEXT_FEATURES: Tuple[str, ...] = (
    "action_uu", "action_pc5", "action_both", "is_switch")
# Recorded per step, never in the context frame and never in any reward.
DUPLICATION_CHANGE_DIAGNOSTICS: Tuple[str, ...] = (
    "duplication_delta",             # |d_t - d_{t-1}|
    "duplication_action_changed",    # int(d_t != d_{t-1})
)

# An unmeasured path must not look like a perfect path. A normalized
# "lower is better" feature has 0.0 = best, so 0.0 is the WRONG sentinel for
# "unknown". The midpoint of the normalized range carries no optimism or
# pessimism and is not an invented tunable.
UNMEASURED_SENTINEL = 0.5

# 19, derived from NETWORK_CONTEXT_FEATURE_NAMES rather than asserted: 22 in
# contract v5, minus the three action_uu/action_pc5/action_both one-hot entries
# and minus is_switch, plus the one applied_duplication_ratio scalar.
#
# The STEP 1 audit projected 20 for this constant. The difference is exactly
# is_switch: that projection assumed it stayed in the context, and the
# pure-redundancy action space removes it from formal state because "the path
# changed" no longer denotes anything. RADIO_STATE_DIM and LOWER_INPUT_DIM are
# unaffected and match the projection exactly.
NETWORK_CONTEXT_DIM = len(NETWORK_CONTEXT_FEATURE_NAMES)

# ------------------------------------------------- mobility context  c_t
# Contract 8. Road-plane kinematics of the ego vehicle, consumed by the UPPER
# tier only (Temporal Transformer reward designer + GoalCritic). The lower
# DDQN never sees these: see RADIO_SLICE_INDICES below, which stays inside the
# network block.
#
# Both features are STRICT_EXOGENOUS. They are mobility-model state driven by
# the replayed SUMO trace, so the duplication action cannot move them, and
# ns-3 samples them before Python selects this tick's goal and action.
#
# Neither carries a missing value. Once mobility is installed both are always
# defined, so ns-3 aborts loudly on an impossible missing condition rather
# than folding it into a valid numeric extreme -- 0 m/s and 0 m are both
# legitimate observations, which is exactly why UNMEASURED_SENTINEL must not
# be reused here.
#
# Naming: "ego_rsu_distance", not "nearest-RSU distance". numRsu == 1, so no
# nearest-of-many selection takes place, and the quantity is 2-D horizontal
# because the semantic is road-plane geometry rather than link geometry.
VEHICLE_CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    "ego_speed_norm",            # 0  clip01(|v|_2D / speed_ref_mps)
    "ego_rsu_distance_norm",     # 1  clip01(d_2D(ego, RSU) / distance_ref_m)
)
VEHICLE_CONTEXT_DIM = len(VEHICLE_CONTEXT_FEATURE_NAMES)

SERVICE_SLOTS: Tuple[ServiceType, ...] = (
    ServiceType.NORMAL,
    ServiceType.VIDEO,
    ServiceType.SAFETY,
    ServiceType.LATENCY_CRITICAL,
)
SERVICE_SLOT_COUNT = len(SERVICE_SLOTS)
SERVICE_SLOT_FEATURES = 5
SERVICE_CONTEXT_DIM = SERVICE_SLOT_COUNT * SERVICE_SLOT_FEATURES

# ------------------------------------------------------- x_t = [c_t, v_t, n_t]
# Contract 8 frame layout, physical order == logical order:
#
#     [ mobility c_t (2) | service requirements v_t (20) | network n_t (19) ]
#
# Named slices, so nothing indexes the frame by an undocumented magic offset.
CONTEXT_FEATURE_DIM = (
    VEHICLE_CONTEXT_DIM + SERVICE_CONTEXT_DIM + NETWORK_CONTEXT_DIM)
C_SLICE = slice(0, VEHICLE_CONTEXT_DIM)
V_SLICE = slice(VEHICLE_CONTEXT_DIM,
                VEHICLE_CONTEXT_DIM + SERVICE_CONTEXT_DIM)
N_SLICE = slice(VEHICLE_CONTEXT_DIM + SERVICE_CONTEXT_DIM,
                CONTEXT_FEATURE_DIM)
CONTEXT_BLOCK_ORDER: Tuple[str, ...] = ("c_t", "v_t", "n_t")
if VEHICLE_CONTEXT_DIM + SERVICE_CONTEXT_DIM + NETWORK_CONTEXT_DIM != 41:
    raise AssertionError(
        "contract 8 requires 2 + 20 + 19 == 41 context features, got "
        f"{VEHICLE_CONTEXT_DIM} + {SERVICE_CONTEXT_DIM} + "
        f"{NETWORK_CONTEXT_DIM}")
if CONTEXT_FEATURE_DIM != 41:
    raise AssertionError(f"CONTEXT_FEATURE_DIM must be 41, got {CONTEXT_FEATURE_DIM}")
if (C_SLICE.stop, V_SLICE.start, V_SLICE.stop, N_SLICE.start) != (2, 2, 22, 22):
    raise AssertionError("contract 8 block slices are misaligned")

# The lower DDQN reads the instantaneous radio state by slicing the freshly
# built context frame, so normalization lives in exactly one place. The slice
# is resolved by NAME inside the network block and then offset by N_SLICE, so
# adding the mobility block in front cannot silently repoint it.
#
# To choose a duplication degree the lower controller needs both RSSIs, the
# degree currently in force, and both per-path delays and losses. The secondary
# path's state is kept precisely because the decision "is a second copy worth
# it" is about that path, even though the controller no longer selects paths.
# Resolved by NAME so a reordering of the feature list cannot silently
# repoint the slice at the wrong columns.
_RADIO_SLICE_FEATURES: Tuple[str, ...] = (
    "uu_rssi_norm", "pc5_rssi_norm", "applied_duplication_ratio",
    "uu_delay_norm", "pc5_delay_norm", "uu_loss_rate", "pc5_loss_rate",
)
# Network-block-relative indices. FROZEN: this tuple defines WHICH seven
# features the lower controller sees, and contract 8 does not change it.
_RADIO_SLICE_NETWORK_INDICES: Tuple[int, ...] = tuple(
    NETWORK_CONTEXT_FEATURE_NAMES.index(name) for name in _RADIO_SLICE_FEATURES)
# Absolute frame indices, derived from the layout rather than hard-coded.
RADIO_SLICE_INDICES: Tuple[int, ...] = tuple(
    N_SLICE.start + i for i in _RADIO_SLICE_NETWORK_INDICES)
RADIO_STATE_DIM = len(RADIO_SLICE_INDICES)

# ---------------------------------------------------------------- goal
# The Goal Planner emits a normalized QoS preference mixture, NOT a discrete
# mode. w_i >= 0 and sum(w_i) == 1, enforced by a softmax at the planner head.
GOAL_DIM = 4
GOAL_WEIGHT_NAMES: Tuple[str, ...] = ("w_rel", "w_lat", "w_thr", "w_cost")
GOAL_SEMANTICS = {
    "w_rel": "reliability / deadline delivery importance",
    "w_lat": "latency importance",
    "w_thr": "useful throughput importance",
    "w_cost": "communication resource efficiency importance",
}
GOAL_SUM_TOLERANCE = 1e-5
LOWER_INPUT_DIM = RADIO_STATE_DIM + GOAL_DIM

SERVICE_CONTEXT_MODES = ("multi", "none")
DEFAULT_SERVICE_CONTEXT_MODE = "multi"

SERVICE_SLOT_FEATURE_NAMES: Tuple[str, ...] = (
    "active",
    "min_throughput_norm",
    "max_delay_norm",
    "min_pdr",
    "max_outage_norm",
)
# Contract-8 frame names, in physical == logical order [c_t, v_t, n_t].
SERVICE_CONTEXT_FEATURE_NAMES: Tuple[str, ...] = tuple(
    f"{service.name.lower()}_{feature}"
    for service in SERVICE_SLOTS
    for feature in SERVICE_SLOT_FEATURE_NAMES
)
CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    VEHICLE_CONTEXT_FEATURE_NAMES
    + SERVICE_CONTEXT_FEATURE_NAMES
    + NETWORK_CONTEXT_FEATURE_NAMES
)
if len(SERVICE_CONTEXT_FEATURE_NAMES) != SERVICE_CONTEXT_DIM:
    raise AssertionError("service feature names must match SERVICE_CONTEXT_DIM")
# Each named block must sit exactly on its declared slice.
if CONTEXT_FEATURE_NAMES[C_SLICE] != VEHICLE_CONTEXT_FEATURE_NAMES:
    raise AssertionError("C_SLICE does not cover the mobility block")
if CONTEXT_FEATURE_NAMES[V_SLICE] != SERVICE_CONTEXT_FEATURE_NAMES:
    raise AssertionError("V_SLICE does not cover the service block")
if CONTEXT_FEATURE_NAMES[N_SLICE] != NETWORK_CONTEXT_FEATURE_NAMES:
    raise AssertionError("N_SLICE does not cover the network block")


# ------------------------------------------------------------- upper encoder
@dataclass(frozen=True)
class TransformerConfig:
    """Upper encoder hyperparameters. Trained from scratch, never pretrained."""

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if min(self.d_model, self.nhead, self.num_layers, self.dim_feedforward) <= 0:
            raise ValueError("transformer dimensions must be positive")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


DEFAULT_TRANSFORMER = TransformerConfig()


# ------------------------------------------------------------- E2E reward
# Initial fixed task-level reward for the E2E architecture. It is the centroid
# of the five hand-tuned goal policies above, so the reward scale stays
# identical to REWARD_POLICIES and GAMMA/LR/grad-clip need no recalibration.
# This is an *initial* fixed weighting, not a claimed optimum: reward ablations
# add a new key here and select it via --e2e-reward-version.
# RETIRED as a learning objective. It trained the two-action MP-QUIC campaign
# and is kept only so those runs stay reproducible and their logs readable. The
# dual-connectivity campaign uses UPPER_OBJECTIVE_VERSION instead; this key is
# renamed so the two can never be confused in a checkpoint or a config hash.
E2E_REWARD_POLICIES: Dict[str, RewardPolicy] = {
    "legacy_task_level_e2e_v1": RewardPolicy(
        "legacy_task_level_e2e_v1", 0.22, 0.22, 0.23, 0.15, 0.10, 0.08),
}
LEGACY_TASK_REWARD_VERSION = "legacy_task_level_e2e_v1"
DEFAULT_E2E_REWARD_VERSION = LEGACY_TASK_REWARD_VERSION

# ------------------------------------------------- dynamic lower reward
# The four normalized components the Goal Planner mixes. Every one is
# "larger = better", 0 = bad, 1 = good.
#
#   r_rel  = 1 - deadline_loss_rate                    (logical, dedup'd)
#   r_lat  = 1 - clip01(delay_ms   / delay_ref_ms)
#   r_thr  =     clip01(useful_bps / throughput_ref_bps)
#   r_cost = 1 - clip01(attempted_copies_per_logical - 1)
#
# r_lower = w_rel*r_rel + w_lat*r_lat + w_thr*r_thr + w_cost*r_cost
#
# r_cost reads ATTEMPTED copies, not successful ones: a Both decision whose
# PC5 send failed still cost the controller a duplication decision, and must
# not be scored as cheaply as a single path.
#   Uu   -> 1 attempted -> overhead 0 -> r_cost 1
#   PC5  -> 1 attempted -> overhead 0 -> r_cost 1
#   Both -> 2 attempted -> overhead 1 -> r_cost 0
# v2: r_thr's INPUT semantics changed from max(path throughput) to measured
# unique logical goodput. The four-component weighted form is identical; what
# the third component measures is not, so an artifact written under v1 must
# not be pooled with one written under v2.
LOWER_REWARD_VERSION = "dynamic_qos_goal_v2"
RETIRED_LOWER_REWARD_VERSIONS = ("dynamic_qos_goal_v1",)
LOWER_REWARD_COMPONENTS: Tuple[str, ...] = ("r_rel", "r_lat", "r_thr", "r_cost")

REWARD_NORMALIZATION_NOTES = {
    "r_rel": (
        "1 - deadline_loss_rate. Already a rate in [0,1]; loss_ref_rate=1.0 is "
        "an exact match."),
    "r_lat": (
        "delay_ref_ms=250.0 is reused. It is the same constant as the ns-3 "
        "application deadline, so r_lat reaches 0 exactly when the mean delay "
        "reaches the deadline. Exact semantic match."),
    "r_thr": (
        "throughput_ref_bps=4e6 is reused. It is the offered ToD load, so "
        "r_thr=1 means the path carried the full offered rate. Exact match."),
    "r_cost": (
        "Not covered by the pre-existing NORMALIZATION block: there was no "
        "duplication cost before. Normalised as overhead above a single "
        "attempted copy, so a single path scores 1 and full duplication 0, "
        "matching the 0=bad/1=good direction of the other three."),
}

# ------------------------------------------------- fixed upper objective
# The system-level objective that trains the Goal Planner. Fixed by the
# researcher, never produced by the planner, never changed during training.
#
#   V_s      = max( viol_delay_s, viol_pdr_s, viol_thr_s, viol_outage_s )
#   V_system = max_s V_s
#   U_qos    = 1 - V_system
#
# Each active service is evaluated INDEPENDENTLY against its own declared
# requirement. effective_requirement() -- the synthetic strictest-of-all merge
# -- is deliberately NOT used: merging first can only be equivalent under a
# max aggregator, and even there it destroys the attribution the analysis
# needs ("which service's which requirement bound the system?"). It stays an
# evaluation/reference metric, exactly as its docstring always said.
#
# Violations are normalized against each requirement itself, so no coefficient
# is invented at either aggregation layer: both are max.
#
# Resource / duplication is NOT a term here. It is an explicit constraint on
# duplication_rate (see below), so a resource weight never has to be invented.
#
# Switching is NOT a term here either. With the hold removed the honest reason
# is stated plainly: an action change costs the modelled network nothing, so
# there is nothing to charge for. switch_count, switch_rate, pingpong_count
# and the action dwell-time distribution stay measured as SECONDARY
# DIAGNOSTICS, so how oscillatory an unconstrained learned policy actually is
# can be observed rather than assumed.
# v3: the throughput KPI the service violation is evaluated against changed
# from max(path throughput) to measured logical goodput. The service-wise
# max-over-KPI / max-over-service STRUCTURE is untouched.
# ------------------------------------------- GoalCritic Bellman operator
# WHICH goal the bootstrap is evaluated at. This is a SEMANTIC contract, not a
# tunable: it decides what Q_goal(c, g) means, so a checkpoint trained under
# one operator is not interchangeable with one trained under the other even
# though the tensor shapes are identical.
#
#   planner_target_bootstrap_v1 (RETIRED)
#       y = R_macro + gamma^H (1-done) Q_target(c', planner_target(c'))
#       Q_goal(c, g) = value of holding g for ONE macro and then following the
#       target planner. The current goal therefore enters the target only
#       through R_macro, which is bounded by sum_{j<H} gamma^j = 4.900995,
#       while the bootstrap grows with the value scale and does not depend on
#       g at all. MEASURED consequence (PHASE D-3B, two seeds, ep161): the
#       goal-attributable share of the target was 0.89% / 1.13% and had
#       degraded 720x from 29.1 at ep1; the critic responded by learning a
#       ~99% affine goal surface (mean R2_affine 0.991 / 0.996) whose slope
#       direction was nearly context-invariant (C_dir 0.923 / 0.916), which
#       specifies a context-INDEPENDENT optimal vertex on the simplex. The
#       planner, ascending that surface correctly, collapsed to a constant
#       goal by ep8 (D1 1.15e-01 -> 6.26e-04).
#
#   goal_consistent_bootstrap_v1 (CURRENT)
#       y = R_macro + gamma^H (1-done) Q_target(c', g)
#       with g the EXACT goal stored on the current macro transition and used
#       on the prediction side. Q_goal(c, g) now approximately represents the
#       expected upper return when g is held CONSISTENTLY over the value
#       horizon from c. The purpose is to propagate goal-specific credit
#       through the bootstrap horizon instead of confining it to one macro:
#       the accumulation available is 1/(1-gamma^H) = 20.40 macros against a
#       measured 4.66x / 3.22x under the retired operator.
#
# gamma, H, N, tau, the learning rates, the clip, K, W, beta, p_global, the
# projection radius and both architectures are UNCHANGED by this.
GOAL_CRITIC_OPERATOR = "goal_consistent_bootstrap_v1"
GOAL_CRITIC_SEMANTIC_LABEL = "persistent-goal counterfactual evaluator"
# The bootstrap evaluates continuation under the SAME goal, so Q_goal is a
# goal-consistent long-horizon value surrogate. The planner still re-emits a
# goal at every macro decision, so Q_goal is NOT the exact Bellman
# action-value function of the reselect-every-macro upper policy, and must
# never be described as such. This distinction must survive into the paper.
GOAL_CRITIC_SEMANTIC_NOTE = (
    "persistent-goal counterfactual GoalCritic / goal-consistent long-horizon "
    "value surrogate. NOT the exact Bellman action-value function of the "
    "reselect-every-macro upper policy.")

RETIRED_GOAL_CRITIC_OPERATORS = ("planner_target_bootstrap_v1",)
GOAL_CRITIC_OPERATOR_STATUS = (
    "goal_consistent_bootstrap_v1 is a PREREGISTERED single-change remediation "
    "of the PHASE D-3B diagnosis. It introduces no new hyperparameter. Its "
    "falsification rule (C_dir >= 0.90 AND mean R2_affine >= 0.97 at ep161) was "
    "fixed before the run; tau = 0.005416 is NOT inherited, because it was "
    "anchored to the retired operator's value interpretation.")

UPPER_OBJECTIVE_VERSION = "service_wise_violation_v3"
RETIRED_UPPER_OBJECTIVE_VERSIONS = ("service_wise_violation_v1",
                                    "service_wise_violation_v2")
# How the four KPI violations of ONE service combine.
SERVICE_AGGREGATION_MODES = ("worst", "mean")
# 'worst' is the formal choice: QoS is conjunctive, so the binding requirement
# is the honest measure and a met requirement must not compensate a violated
# one. 'mean' exists as a diagnostic and is not a formal campaign condition.
SERVICE_AGGREGATION_MODE = "worst"
FORMAL_SERVICE_AGGREGATION_MODE = "worst"
# How services combine. Always max: a met video requirement must never dilute
# a violated safety requirement.
SYSTEM_AGGREGATION_MODE = "max_over_services"
UPPER_OBJECTIVE_INCLUDES_SWITCHING = False
UPPER_OBJECTIVE_INCLUDES_RESOURCE = False
UPPER_OBJECTIVE_USES_EFFECTIVE_REQUIREMENT = False
UPPER_OBJECTIVE_INPUT = "matured_tick_cohort"   # never the raw KPI window

# ------------------------------------------------- resource constraint
# Duplication is a budgeted resource, not a weighted objective term:
#
#   constraint_violation = max(0, dup_rate_window - B_dup)
#   J        = U_qos - lambda * constraint_violation
#   lambda  <- max(0, lambda + eta * (dup_rate_window - B_dup))
#
# The STRUCTURE is implemented (metric, sliding window, dual variable,
# logging). The three numbers are NOT chosen: choosing them now, before the
# protocol fixes them, is exactly the "pick a coefficient after looking at
# results" failure mode the design forbids. They stay REQUIRED_BUT_UNSET and
# full training refuses to start without them.
REQUIRED_BUT_UNSET = "REQUIRED_BUT_UNSET"
RESOURCE_CONSTRAINT_VERSION = "duplication_lagrangian_v2"
RESOURCE_CONSTRAINT_METRIC = "duplication_rate"   # Both decisions / logical tx
DUPLICATION_BUDGET: Optional[float] = None            # REQUIRED_BUT_UNSET
DUPLICATION_WINDOW: Optional[int] = None              # REQUIRED_BUT_UNSET
DUPLICATION_DUAL_ETA: Optional[float] = None          # REQUIRED_BUT_UNSET
DUPLICATION_LAMBDA_INIT = 0.0     # structural, not a tunable: dual starts off
# v2 semantics, all three frozen by protocol review:
#   partial window : while the sliding window holds fewer than L samples the
#       constraint is INACTIVE (rate None, violation 0, no dual step). A
#       partial sample mean is not a duty cycle, and reading it as one made a
#       single early Both decision look like 100% duplication.
#   normalized violation : v_norm = v / (1 - B_dup) is used by BOTH the dual
#       step and the objective penalty, so lambda = 1 means "penalty equal to
#       the whole objective range" independently of B_dup.
#   dual cadence : the dual variable steps once per macro transition push, on
#       the same event stream as the goal critic, never once per 100 ms tick
#       over a window that changed by one sample.
DUPLICATION_PARTIAL_WINDOW_POLICY = "inactive_until_full"
DUPLICATION_VIOLATION_NORMALIZED = True
DUPLICATION_DUAL_CADENCE = "macro_push"
DUPLICATION_NORM_EPS = 1e-12      # NUMERICAL_TOLERANCE, not a science parameter

# ---------------------------------------------------------------- binding tie
# Two active services can hold the identical worst violation. Reporting one of
# them by slot order is deterministic but arbitrary, and in a saturated regime
# it silently over-attributes to whichever service sits earlier in
# SERVICE_SLOTS. The tie is preserved instead. V_system is unchanged: max does
# not depend on which tied name is reported.
BINDING_TIE_POLICY = "tie_label_plus_list"
BINDING_TIE_LABEL = "TIE"
BINDING_TIE_TOLERANCE = 1e-12     # NUMERICAL_TOLERANCE, not a science parameter
RESOURCE_CONSTRAINT_STATUS = (
    "REQUIRED_BUT_UNSET: duplication_rate is measured, windowed and handed to "
    "a dual-variable constraint evaluator, but B_dup, the window length and "
    "the dual step eta all require protocol confirmation. Until they are set "
    "lambda stays 0, J == U_qos, and --run-mode full is refused.")

# ------------------------------------------------- goal exploration
# The upper tier had NO exploration: plan_goal() was deterministic, so
# Q_goal(c, g) saw a single goal per context and the planner gradient pushed
# against a critic with no counterfactual support. Noise is added to the
# planner LOGITS, before the softmax, so the exploratory goal is still exactly
# a point on the simplex (w_i >= 0, sum w_i == 1) rather than a clipped or
# renormalised vector. FIXED and RULE have no logits and take no noise; eval
# always uses noise 0.
# RETIRED. Screening protocol v1 identified a failure of the logit-space
# exploration parameterization: the effective goal-space exploration radius
# collapsed as the deterministic softmax policy saturated. Because the
# perturbation entered through the softmax Jacobian, which vanishes at a
# simplex vertex, sigma stopped controlling the exploration radius entirely
# (measured: radius fell 3-5 orders of magnitude by episode 4-8, and was not
# even monotone in sigma for the MLP planner). The string is kept so v1
# artifacts stay readable; it is never emitted by this protocol version.
RETIRED_GOAL_EXPLORATION_VERSION = "logit_gaussian_presoftmax_v1"
GOAL_EXPLORATION_VERSION = "RETIRED_LOGIT_GAUSSIAN"

# ---------------------------------------------- simplex-space exploration
# Formal exploration for screening protocol v2.
#
#     g_det = softmax(planner_logits)
#     u     ~ Dirichlet(1,1,1,1)          uniform on the goal simplex
#     g_beh = (1 - beta) * g_det + beta * u
#     g_eval = g_det
#
# A convex combination of two simplex points, so g_beh lies on the simplex by
# construction: no projection, no clipping, no renormalisation. And because
#
#     g_beh - g_det = beta * (u - g_det)
#
# the radius carries NO softmax Jacobian, which is exactly what made the
# retired mechanism collapse. With E[u] = c (the centroid) and
# trace(Cov(u)) = 0.15 for Dirichlet(1,1,1,1) on four components,
#
#     E||g_beh - g_det||^2 = beta^2 * ( 0.15 + ||c - g_det||^2 )
#
# so the RMS radius is bracketed by beta * [sqrt(0.15), sqrt(0.90)] for every
# g_det on the simplex, and is LARGEST at a vertex -- the opposite of the
# retired mechanism.
EXPLORATION_PROTOCOL_VERSION = "simplex_dirichlet_local_global_v2"
RETIRED_EXPLORATION_PROTOCOL_VERSIONS = (
    "logit_gaussian_presoftmax_v1", "simplex_dirichlet_mixing_v1")
GOAL_EXPLORATION_MECHANISM = "SIMPLEX_DIRICHLET_LOCAL_GLOBAL"
DIRICHLET_CONCENTRATION = 1.0          # Dir(1,1,1,1) = uniform on the simplex
# Exact moments of Dirichlet(1,...,1) on GOAL_DIM components.
DIRICHLET_TRACE_COV = (GOAL_DIM - 1.0) / (GOAL_DIM * (GOAL_DIM + 1.0))   # 0.15
# Geometric scales of the goal simplex.
SIMPLEX_CENTROID_VERTEX_DIST = ((1.0 - 1.0 / GOAL_DIM) ** 2
                                + (GOAL_DIM - 1) * (1.0 / GOAL_DIM) ** 2) ** 0.5
# Worst case (g_det at the centroid) and best case (g_det at a vertex).
EXPLORATION_RADIUS_MIN_FACTOR = DIRICHLET_TRACE_COV ** 0.5
EXPLORATION_RADIUS_MAX_FACTOR = (
    DIRICHLET_TRACE_COV + SIMPLEX_CENTROID_VERTEX_DIST ** 2) ** 0.5
# Guaranteed RMS coverage, as a fraction of the centroid-to-vertex distance.
COVERAGE_PER_BETA = EXPLORATION_RADIUS_MIN_FACTOR / SIMPLEX_CENTROID_VERTEX_DIST

BETA: Optional[float] = None                          # REQUIRED_BUT_UNSET
BETA_SELECTION_TYPE = "exogenous_goal_space_exploration_budget"
# A preregistered DESIGN RULE, not a mathematical necessity and not a simplex
# theoretical bound: beta = 0.5 is simply the point at which the planner and
# the uniform reference contribute equally to E[g_beh], so below it the
# planner remains the dominant component of its own behaviour policy.
PLANNER_DOMINANT_EXPLORATION_POLICY_MAX_BETA = 0.5
# Preregistered guaranteed-coverage levels. The levels themselves are an
# exogenous exploration budget choice; they are NOT claimed to be forced by
# the geometry. Only the coverage -> beta map is forced.
BETA_COVERAGE_LEVELS: Tuple[float, ...] = (0.05, 0.10, 0.20)
BETA_STATUS = (
    "REQUIRED_BUT_UNSET: beta is an exogenous goal-space exploration budget. "
    "The simplex geometry fixes the map from a guaranteed coverage fraction "
    "to beta exactly, but not which coverage to buy.")


# ------------------------------------------- global simplex exploration (U8)
# The STEP 4A audit found that beta-mixing alone ANCHORS every behaviour goal
# to the planner's own output. Once the planner collapsed to a vertex, the
# critic only ever saw goals within ~0.09 of that vertex, and w_thr / w_cost
# were never observed above ~0.3 -- which is exactly where the environment's
# context x goal interaction lives (0.40-0.52 SD on w_cost). Raising beta does
# not fix this: it widens the ball but keeps the anchor.
#
# The behaviour policy therefore gains a second mode that does not reference
# the planner at all:
#
#   LOCAL   (prob 1 - p_global)   g_beh = (1-beta) g_det + beta u,  u ~ Dir(1)
#   GLOBAL  (prob p_global)       g_beh ~ Dir(1,1,1,1)
#
# GLOBAL goals are BEHAVIOUR ONLY. They are not a manual reward rule, not a
# planner target, not a supervision label, and never the deployed policy: the
# planner still emits g_det = pi(c) and is still trained through the critic.
GOAL_EXPLORATION_MODES: Tuple[str, ...] = ("local", "global")

# Candidate derivation -- from the coverage requirement, never from performance.
#   LOWER BOUND. One coordinate's marginal under Dir(1,...,1) is Beta(1, n-1).
#     Require that within the SMALLER service regime at least one global draw
#     exceeds that marginal's p90, with probability >= 0.95:
#         1 - 0.9^k >= 0.95  ->  k_min = ceil(ln 0.05 / ln 0.9) = 29
#     The smaller regime holds 160 of a 400-macro harvest (measured), so
#         p_global >= 29 / 160 = 0.18125.
#   UPPER BOUND. LOCAL must stay the majority mode, because the DPG planner
#     gradient is evaluated at pi(c) and the critic needs dense support there.
#     p_global = 0.50 is the equal-mixture endpoint and the structural limit.
P_GLOBAL_COVERAGE_QUANTILE = 0.90
P_GLOBAL_COVERAGE_CONFIDENCE = 0.95
P_GLOBAL_MIN_GLOBAL_DRAWS_PER_REGIME = 29
P_GLOBAL_BINDING_REGIME_MACROS = 160
P_GLOBAL_LOWER_BOUND = (P_GLOBAL_MIN_GLOBAL_DRAWS_PER_REGIME
                        / P_GLOBAL_BINDING_REGIME_MACROS)      # 0.18125
P_GLOBAL_UPPER_BOUND = 0.50
P_GLOBAL_CANDIDATES: Tuple[float, ...] = (0.20, 0.35, 0.50)
P_GLOBAL_STATUS = (
    "REQUIRED_BUT_UNSET: selected by CONTEXT-CONDITIONAL COVERAGE, never by "
    "performance. beta is unchanged and remains the LOCAL exploration budget.")


# ------------------------------------------ planner anti-saturation (U7)
# The STEP 4A audit measured the collapse mechanism exactly: the planner
# reached a simplex vertex (entropy 1.4e-04 / 2.0e-03 against a ceiling of
# log 4), the logit gap grew to 11.93 / 9.30, and the softmax Jacobian
# diag(g) - g g^T then drove dQ/dz to 1e-06 / 2.4e-05. The planner froze, and
# its logits kept responding to context (MLP centred-logit std 0.90) while its
# OUTPUT no longer did -- which is precisely the observed D1-D5 ~ 1e-05.
#
# The fix is a parameterization, not a loss term. No entropy bonus: the goal
# distribution is not pushed toward uniform, it is only prevented from running
# away to a gradient-free corner.
#
#   z_raw     = planner head output          (unchanged network)
#   z_centred = z_raw - mean(z_raw)          (removes the softmax null mode,
#                                             which carries no information and
#                                             was free to grow without bound)
#   z_bounded = z_centred * R / max(||z_centred||, R)     (ball projection)
#   g_det     = softmax(z_bounded)
#
# WHY A BALL PROJECTION AND NOT A SQUASHING NONLINEARITY. An elementwise
# squash such as R*tanh(z/R) was tried first and REJECTED on measurement: it
# keeps the output off the vertex, but its own derivative vanishes, so a raw
# logit that keeps growing still kills the gradient -- at raw gap 50 the
# transmitted |dQ/dz| measured exactly 0. That moves the saturation from the
# softmax to the squash instead of removing it.
#
# The projection behaves differently, and the difference is the whole point.
# Inside the ball it is the identity. Outside it is z -> R*z/||z||, whose
# Jacobian is (R/||z||)(I - zz^T/||z||^2): the RADIAL direction gets exactly
# zero gradient, so nothing pushes the logits further out and the goal cannot
# get sharper -- while every TANGENTIAL direction keeps a gradient of
# magnitude R/||z||. The planner can therefore always change WHICH coordinate
# it prefers, which is precisely the context-responsiveness that collapsed.
# The decay is 1/||z||, not exponential.
#
# R IS DERIVED, NOT TUNED. The planner must not be able to express a goal
# sharper than the sharpest region its own critic has support on. After the U8
# fix that support is exactly Dir(1,...,1), whose expected smallest component
# is 1/n^2. On the one-dominant centred direction (n-1, -1, ..., -1)/||.|| the
# softmax gap is R*sqrt(n/(n-1)), so requiring the sharpest representable goal
# to have smallest component 1/n^2 gives
#
#       exp(R * sqrt(n/(n-1))) = n^2 - n + 1
#         ->  R = ln(n^2 - n + 1) * sqrt((n-1)/n)
#
# Both inputs are already-frozen constants: n = GOAL_DIM and the Dirichlet
# concentration 1. For n = 4: R = ln(13)*sqrt(3/4) = 2.2213113030, the
# sharpest representable goal is (0.8125, 0.0625, 0.0625, 0.0625), the worst
# corner for a single component is (0.325, 0.325, 0.325, 0.025), and the
# centroid (0.25, 0.25, 0.25, 0.25) stays exactly representable at z = 0.
PLANNER_PARAMETERIZATION = "centred_ball_projected_logits_v1"
RETIRED_PLANNER_PARAMETERIZATIONS = (
    "raw_unbounded_logits_v1", "centred_tanh_bounded_logits_v1")
DIRICHLET_EXPECTED_MIN_COMPONENT = 1.0 / (GOAL_DIM ** 2)
PLANNER_LOGIT_BOUND = (math.log(GOAL_DIM * GOAL_DIM - GOAL_DIM + 1.0)
                       * math.sqrt((GOAL_DIM - 1.0) / GOAL_DIM))
PLANNER_MAX_REPRESENTABLE_DOMINANCE = (
    1.0 - (GOAL_DIM - 1) * DIRICHLET_EXPECTED_MIN_COMPONENT)
# R is fixed by the ONE-DOMINANT case (R, -R, ..., -R). The worst case for the
# smallest component is the opposite corner (R, ..., R, -R), so the GUARANTEED
# floor on any representable component is smaller than 1/n^2 -- but it is a
# hard floor, which is the whole point: the collapse being fixed drove the
# smallest component to ~1e-06.
# Worst corner for a single component is the direction (1, ..., 1, -(n-1)):
# gap = R * sqrt(n/(n-1)) again, but shared by n-1 large components.
PLANNER_MIN_REPRESENTABLE_COMPONENT = 1.0 / (
    (GOAL_DIM - 1) * math.exp(PLANNER_LOGIT_BOUND
                              * math.sqrt(GOAL_DIM / (GOAL_DIM - 1.0))) + 1.0)
# ------------------------------------------- E2E timescale protocol (4B-3)
# MEASURED, not assumed. K alone cannot express a two-timescale separation
# under Adam:
#
#   critic  gradient 7 -> 102,   realised step/gradient 4.5e-04 -> 2.9e-05
#   planner gradient 0.017 -> 0.070, realised step/gradient 1.25 -> 0.38
#
# Adam sets the per-parameter step to ~lr regardless of gradient magnitude, so
# K divides the planner's step COUNT and leaves its step SIZE untouched. The
# predicted normalized movement ratio planner/critic is sqrt(n_p/n_c)/K = 0.80
# -- the planner should move LESS -- but the measured ratio is 1.09-2.35, and
# the planner completes 31-45% of its total parameter movement while the critic
# is still below 20% of its steady-state value scale. The two seeds that
# diverged were already in opposite simplex corners at 368 critic updates, when
# the critic sat at 9.6% of steady state.
#
# The correction is on the axis the measurement implicates -- effective step
# size -- and introduces NO new number: K is already frozen and lr_critic is
# unchanged. Setting the planner's step size smaller by the same K makes K
# carry the separation in both count and size.
#
#   lr_planner = lr_critic / K
#
# K itself is NOT changed. The screening result K = 2 stands as it was
# selected; this is a separate protocol quantity and is recorded separately.
# ROLLED BACK after controlled validation (STEP 4C, §21 CASE D).
# The correction did exactly what it was designed to do -- normalized movement
# ratio planner/critic fell from 1.09-2.35 to 0.59-1.27, planner RMS/step fell
# from 2.2e-04 to 1.4e-04, and the early commitment shrank (logit gap at 368
# updates 2.02 -> 0.23, entropy 0.73 -> 1.07) -- and the END-TO-END outcome got
# worse: planner responsiveness replication went from 1/2 to 0/2, and the one
# seed that had reached D3 = 0.963 under the equal-lr protocol fell to
# 1.19e-04. Slowing the planner is therefore NOT the fix, and the measured
# timescale mismatch is not by itself the cause of the seed-dependent
# emergence. The equal-lr protocol is restored unchanged.
E2E_TIMESCALE_PROTOCOL = "equal_lr_k_only_v1"
RETIRED_E2E_TIMESCALE_PROTOCOLS = ("planner_lr_scaled_by_k_v1",)
PLANNER_LR_SCALED_BY_K = False
E2E_TIMESCALE_PROTOCOL_STATUS = (
    "equal_lr_k_only_v1 restored after controlled validation. The measured "
    "timescale mismatch is REAL and unexplained, but correcting it on the "
    "learning-rate axis reduced responsiveness replication from 1/2 to 0/2, "
    "so the mismatch is not the cause of seed-dependent emergence.")


PLANNER_PARAMETERIZATION_STATUS = (
    "R is fixed by GOAL_DIM and the Dirichlet reference concentration, both "
    "already frozen. It is NOT a temperature to be tuned against performance.")


def bounded_centred_logits(z):
    """z_raw -> centred logits projected onto the ball of radius R.

    Shared by every neural planner so the two architectures cannot drift into
    different output semantics. Identity inside the ball; outside, the radial
    component is frozen and the tangential ones keep a gradient of magnitude
    R/||z||. See the derivation above for why this and not a squash.
    """
    import torch
    centred = z - z.mean(dim=-1, keepdim=True)
    norm = centred.norm(dim=-1, keepdim=True)
    scale = PLANNER_LOGIT_BOUND / torch.clamp(norm, min=PLANNER_LOGIT_BOUND)
    return centred * scale


def beta_from_coverage(coverage_fraction: float) -> float:
    """beta implied by a GUARANTEED RMS coverage fraction.

        f_guaranteed = COVERAGE_PER_BETA * beta      (worst case: g_det = c)
        beta         = f_guaranteed / COVERAGE_PER_BETA

    Candidates are computed from this map rather than written as rounded
    constants, so the recorded beta and the recorded coverage can never drift.
    """
    coverage = float(coverage_fraction)
    if not 0.0 < coverage:
        raise ValueError("coverage fraction must be > 0")
    return coverage / COVERAGE_PER_BETA


def coverage_from_beta(beta: float) -> float:
    """Inverse of beta_from_coverage."""
    return float(beta) * COVERAGE_PER_BETA


def expected_squared_radius(beta: float, g_det) -> float:
    """Closed form E||g_beh - g_det||^2 = beta^2 (trace(Cov) + ||c - g||^2)."""
    import numpy as np

    goal = np.asarray(g_det, dtype=np.float64)
    centroid = np.full(goal.shape, 1.0 / goal.shape[-1])
    return float(beta) ** 2 * (DIRICHLET_TRACE_COV
                               + float(((centroid - goal) ** 2).sum()))


def exploration_radius_bracket(beta: float) -> Tuple[float, float]:
    """Analytic two-sided bracket on RMS||g_beh - g_det||, valid for any g_det."""
    return (float(beta) * EXPLORATION_RADIUS_MIN_FACTOR,
            float(beta) * EXPLORATION_RADIUS_MAX_FACTOR)


def k_stability_bounds(gamma: float) -> Dict[str, float]:
    """Numerical-runaway bounds for the upper critic, derived not invented.

    With the resource constraint INERT the per-tick upper reward is
    J = U_qos in [0, 1], so the discounted value is bounded by 1/(1-gamma)
    regardless of the macro horizon. These detect numerical runaway only;
    they never require the critic loss to improve, because K is a timescale
    separation parameter and not a performance parameter.
    """
    q_bound = 1.0 / (1.0 - float(gamma))
    return {"version": K_STABILITY_BOUND_VERSION,
            "q_bound": q_bound, "td_bound": (2.0 * q_bound) ** 2}


K_STABILITY_BOUND_VERSION = "unconstrained_upper_return_v1"
K_STABILITY_BOUND_STATUS = (
    "valid only while the duplication constraint is INERT (J in [0,1]). With "
    "the constraint active J is in [-lambda_max, 1] and the lower bound "
    "depends on lambda, so this version must not be reused automatically.")

# ------------------------------------------------- two-timescale upper tier
# The critic must be a usable estimate before the planner climbs it, and the
# planner must move slowly relative to the critic it is climbing. Both are
# expressed WITHOUT touching learning rates -- lr_planner == lr_critic is a
# structural invariant -- so the only timescale knobs are:
#   K : planner updates once every K critic updates
#   W : critic updates completed before the planner's first update
TWO_TIMESCALE_VERSION = "critic_faster_than_planner_v1"
PLANNER_LR_EQUALS_CRITIC_LR = True
PLANNER_UPDATE_K: Optional[int] = None                # REQUIRED_BUT_UNSET
CRITIC_WARMUP_W: Optional[int] = None                 # REQUIRED_BUT_UNSET
TWO_TIMESCALE_STATUS = (
    "REQUIRED_BUT_UNSET: K and W are protocol numbers. They are exposed as "
    "explicit CLI parameters with no default, and --run-mode full refuses to "
    "start until both are given.")

# ------------------------------------------------- full-training guard
# Every scientific number that is not yet frozen, in one place. A full
# training run must supply all of them; a smoke test may not, and is marked
# protocol_frozen=false everywhere it is recorded.
UNRESOLVED_PROTOCOL_PARAMETERS: Tuple[str, ...] = (
    "duplication_budget",
    "duplication_window",
    "duplication_dual_eta",
    # v2: three logit-noise numbers replaced by one exploration budget
    "beta",
    "planner_update_k",
    "critic_warmup_w",
)
# Screening seeds produce PROTOCOL-SELECTION data. Campaign seeds produce
# FINAL SCIENTIFIC EVALUATION data. A seed used for one must never be reused
# for the other, or a protocol choice leaks into the result it was supposed to
# be independent of.
SCREENING_PROTOCOL_VERSION = 2
SCREENING_PROTOCOL_V1_RECORD = (
    "Screening protocol v1 identified a failure of the logit-space "
    "exploration parameterization: the effective goal-space exploration "
    "radius collapsed as the deterministic softmax policy saturated.")
# Fixed screening engineering budget. NOT a planner convergence horizon, and
# NOT extended adaptively after seeing a result.
SCREENING_HARVEST_EPISODES = 8
SCREENING_SEEDS = (9001, 9002, 9003)
PRIMARY_SCREENING_SEED = 9001
ROBUSTNESS_SCREENING_SEEDS = (9002, 9003)
RETIRED_SEEDS = tuple(range(101, 111))     # two-action campaign, never reused
# FROZEN at PHASE E-0.5 by human approval. Deterministic arithmetic sequence
# 7001 + 10k, k = 0..9: transparent, reproducible, disjoint from
# SCREENING_SEEDS (9001, 9002, 9003) and RETIRED_SEEDS (101..110), with 10-wide
# gaps so episode_rng_run = seed*100000 + episode cannot collide across seeds.
# A failed run is recovered under exact resume with the SAME seed; a seed is
# NEVER replaced because of runtime, convergence, performance or crashes.
TRAIN_SEEDS: Tuple[int, ...] = (7001, 7011, 7021, 7031, 7041,
                                7051, 7061, 7071, 7081, 7091)
# PROPOSED at PHASE E-0.5, NOT yet human-approved: the project had no prior
# convention for the NUMBER of evaluation runs per trained model, so the count
# below is a precision/cost judgement rather than a derived quantity. Five
# deterministic seeds 8001 + 10k, k = 0..4, used identically for every arm and
# every evaluation load level, giving CRN across arms and loads at evaluation.
# rngRun = seed*100000 + episode keeps these disjoint from the 7xxxxxxxx
# training run IDs.
EVAL_SEEDS: Tuple[int, ...] = (8001, 8011, 8021, 8031, 8041)
EVAL_SEEDS_STATUS = "PROPOSED_NEEDS_HUMAN_APPROVAL"


def assert_seed_separation() -> None:
    """SCREENING_SEEDS must stay disjoint from every campaign seed set."""
    screening = set(SCREENING_SEEDS)
    for name, other in (("TRAIN_SEEDS", TRAIN_SEEDS),
                        ("EVAL_SEEDS", EVAL_SEEDS),
                        ("RETIRED_SEEDS", RETIRED_SEEDS)):
        overlap = screening & set(other)
        if overlap:
            raise RuntimeError(
                f"SCREENING_SEEDS overlaps {name}: {sorted(overlap)}")


RUN_MODES = ("smoke", "full")
DEFAULT_RUN_MODE = "smoke"


class ProtocolNotFrozenError(RuntimeError):
    """Raised when full training is attempted with unresolved parameters."""


def protocol_freeze_report(values: Dict[str, object]) -> Dict[str, object]:
    """Which unresolved protocol numbers are still missing, and which are set.

    `values` maps the names in UNRESOLVED_PROTOCOL_PARAMETERS to whatever the
    run supplied. None means "not supplied".
    """
    missing = [
        name for name in UNRESOLVED_PROTOCOL_PARAMETERS
        if values.get(name) is None
    ]
    return {
        "required": list(UNRESOLVED_PROTOCOL_PARAMETERS),
        "missing": missing,
        "resolved": {
            name: values.get(name)
            for name in UNRESOLVED_PROTOCOL_PARAMETERS
            if values.get(name) is not None
        },
        "frozen": not missing,
    }


def require_frozen_protocol(values: Dict[str, object],
                            test_only_flags: Sequence[str] = ()) -> None:
    """Refuse full training while any protocol number is unresolved.

    Also refuses when a TEST-ONLY value is in force: a value that exists only
    to make a smoke test run must never silently become a campaign parameter.
    """
    report = protocol_freeze_report(values)
    problems: List[str] = []
    if report["missing"]:
        problems.append(
            "unresolved protocol parameters: " + ", ".join(report["missing"]))
    if test_only_flags:
        problems.append(
            "TEST-ONLY values in force: " + ", ".join(sorted(test_only_flags)))
    if problems:
        raise ProtocolNotFrozenError(
            "full training refused -- " + "; ".join(problems)
            + ". These are scientific protocol decisions; they must be frozen "
              "explicitly, never defaulted into a run that could become a "
              "claim.")

# ------------------------------------------------- FIXED planner weights
# The FIXED baseline emits this constant goal vector. It is the same balanced
# mixture as the upper objective, which makes FIXED the natural "no planning"
# control: identical priorities at every timestep whatever the context does.
# The FIXED baseline holds one constant preference. It is no longer identical
# to the system objective (which is now service-violation based), so FIXED is a
# genuine "no planning" control rather than a pass-through of the objective.
FIXED_GOAL_WEIGHTS: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

# ------------------------------------------------- RULE planner mapping
# The RULE baseline must emit the same 4-dim normalized vector. The
# threshold-to-weight mapping is UNRESOLVED: it has to come from the existing
# QOS_THRESHOLDS / physical definitions, never from tuning against results.
# RULE emits weights proportional to how badly each objective is currently
# doing, using the same normalized components the reward already computes.
# No thresholds, no learned parameters, no history, no tuning knobs: epsilon
# exists only for numerical stability when every objective is already met.
RULE_GOAL_MAPPING_VERSION = "normalized_deficit_v1"
RULE_EPSILON = 1e-6

# QoS-satisfaction bonus is wired but disabled in v1 so the E2E reward matches
# the existing normalization and scale exactly.
E2E_QOS_BONUS_WEIGHT = 0.0

# gradient clipping is an experimental condition from semantic_config_version 2
# onward: the value reaches the agents through --grad-clip and is part of the
# experiment hash. 1.0 reproduces every run made before that change.
DEFAULT_GRAD_CLIP = 1.0
# v3: three-action space, 4-dim normalized goal vector, duplication accounting.
# v4: action hold removed, tick-exact macro interval, cohort-attributed
#     reliability, service-wise upper objective, Lagrangian duplication
#     constraint and goal exploration wired as protocol placeholders.
# v6: logit-space Gaussian goal exploration RETIRED, replaced by simplex-space
#     Dirichlet convex mixing. The training behaviour policy is part of the
#     experimental condition, so it enters the config hash.
# v5: Phase 1 -- partial-window duplication gate, normalized dual violation on
#     both the dual step and the objective penalty, macro-cadence dual update,
#     binding-service TIE reporting, split goal-support metrics, and a DERIVED
#     epsilon schedule anchored on the frozen training budget.
# v6: pure packet-redundancy lower action. {Uu, PC5, Both} retired; the action
#     is a duplication degree over a fixed primary/secondary pair, realised by
#     a deterministic credit accumulator; the action one-hot and is_switch
#     leave the context in favour of one ordinal applied_duplication_ratio.
# v9: U8 + U7. Behaviour goals gain a planner-independent GLOBAL mode, and the
#     planner logits are centred and projected onto a derived ball so the
#     softmax can no longer saturate to a gradient-free vertex.
# v10: observation contract 8. x_t gains the mobility context block c_t =
#      [ego_speed_norm, ego_rsu_distance_norm], giving x_t = [c_t, v_t, n_t]
#      at 2 + 20 + 19 = 41 features. UPPER tier only: the lower radio state,
#      the goal dimension, LOWER_INPUT_DIM and the action space are unchanged.
# v11: GoalCritic Bellman operator goal_consistent_bootstrap_v1. The bootstrap
#      is evaluated at the transition's OWN goal instead of planner_target(c'),
#      so Q_goal(c, g) becomes the value of holding g over the value horizon.
#      Nothing else moves: same scenario, contract 8, lower tier, action space,
#      reward primitives, gamma, H, K, W, beta, p_global, R, lrs and clips.
#      See GOAL_CRITIC_OPERATOR for the measured reason.
SEMANTIC_CONFIG_VERSION = 11

# ------------------------------------------------------------- pending pool
# Bounded, age-gated decision bookkeeping. Tunable because the real
# PENDING -> APPLIED latency is only known from run logs.
PENDING_MAX_ENTRIES = 32
PENDING_MAX_AGE_S = 1.0


# --------------------------------------------------- concurrent service flows
@dataclass(frozen=True)
class ServiceFlowPeriod:
    """One flow activation interval.

    Unlike ServicePeriod, intervals of different services MAY overlap: that
    overlap is the point. Same-service intervals are interpreted as a union.
    """

    start_s: float
    end_s: float
    service: ServiceType


def _legacy_flow_timeline() -> List[ServiceFlowPeriod]:
    """DEFAULT_SERVICE_TIMELINE, 1:1. At most one service active at any time."""
    return [
        ServiceFlowPeriod(period.start_s, period.end_s, period.service)
        for period in DEFAULT_SERVICE_TIMELINE
    ]


def _video_safety_flow_timeline() -> List[ServiceFlowPeriod]:
    """Continuous video plus a concurrent safety burst.

    The 15-25 s safety window is reused verbatim from DEFAULT_SERVICE_TIMELINE
    so E2E and baseline runs stay time-aligned.
    """
    return [
        ServiceFlowPeriod(0.0, float("inf"), ServiceType.VIDEO),
        ServiceFlowPeriod(15.0, 25.0, ServiceType.SAFETY),
    ]


def _all_flow_timeline() -> List[ServiceFlowPeriod]:
    """Every requirement in force at once (maximum-constraint stress case)."""
    return [
        ServiceFlowPeriod(0.0, float("inf"), service) for service in SERVICE_SLOTS
    ]


MULTI_SERVICE_PRESETS: Dict[str, Callable[[], List[ServiceFlowPeriod]]] = {
    "legacy": _legacy_flow_timeline,
    "video+safety": _video_safety_flow_timeline,
    "all": _all_flow_timeline,
}
# SCENARIO REDESIGN TODO (PHASE C) -- NOT changed in PHASE B.
#
# The formal scenario is FORMAL_SERVICE_FLOW_SCENARIO == "video+safety", but
# this CLI default is still "legacy". Under "legacy" a 28 s episode runs
# VIDEO(0-15) -> SAFETY(15-25) -> NORMAL(25-28) with at most one service
# active at a time, which is NOT the paper's concurrent VIDEO + SAFETY burst.
#
# No archived result is affected: every formal driver passes
# `--service-flow-scenario video+safety` explicitly (verified across all
# longrun / longrun_ts / split drivers), and a PHASE B regression pins that.
# Flipping the default changes scenario semantics, so it belongs to the
# scenario redesign phase, not to a cleanup phase.
DEFAULT_SERVICE_FLOW_SCENARIO = "legacy"


# ---------------------------------------------------------------- scenario id
# TVT_CONTENDED_CORRIDOR_V1 (PHASE C-2). The environment the agent trains in,
# distinct from the architecture it trains WITH. Results produced under a
# different scenario_id are not comparable row-for-row with the earlier
# single-vehicle runs, so every output directory records it.
# v2 differs from v1 in exactly one semantic: mobilityStartOffsetS 92.0 -> 289.0.
# PHASE C-4 moved the episode window because under v1 the closest RSU approach
# (episode t = 18.20 s) fell inside the SAFETY requirement interval [15,25),
# aligning "SAFETY active" with "best PC5 geometry" deterministically. Under v2
# the closest approach is at t = 9.9 s, inside VIDEO-only, and the closest
# region (d_RSU <= 100 m) has 0% overlap with SAFETY.
#
# v1 artifacts keep their own identifier: C-3B's 60-run grid characterises v1
# at offset 92 and must never be relabelled as v2.
SCENARIO_ID = "tvt_contended_corridor_v2"
SCENARIO_MOBILITY_OFFSET_S = 289.0
RETIRED_SCENARIO_IDS = ("tvt_single_vehicle_v0", "tvt_contended_corridor_v1")

# Episode-specific deterministic ns-3 RngRun.
#
# Every episode previously replayed one identical channel realization, because
# the driver passed a constant --rngRun. That made 88 episodes a single
# trajectory rather than 88 samples. The policy below gives each episode its
# own realization while staying exactly reproducible from the master seed.
#
# Collision-free: with 0 <= episode < EPISODE_RNG_STRIDE, the pair
# (master, episode) maps injectively. ns-3 takes RngRun as uint64_t and jumps
# to the substream in O(64) matrix ops regardless of magnitude, and the stream
# holds 2**51 substreams, so a 9-digit value costs and collides nothing.
EPISODE_RNG_STRIDE = 100_000
EPISODE_RNG_POLICY = "master_seed*100000+episode"
# Episode numbering: the FIRST training episode is 1, not 0. Fixed here so the
# driver and the runner cannot disagree about the convention.
FIRST_EPISODE_INDEX = 1


def episode_rng_run(master_seed: int, episode_index: int) -> int:
    """ns-3 RngRun for one episode. Deterministic and collision-free."""
    master = int(master_seed)
    episode = int(episode_index)
    if master < 1:
        raise ValueError(f"master_seed must be >= 1, got {master}")
    if not 0 <= episode < EPISODE_RNG_STRIDE:
        raise ValueError(
            f"episode_index must satisfy 0 <= e < {EPISODE_RNG_STRIDE}, "
            f"got {episode}")
    return master * EPISODE_RNG_STRIDE + episode


def parse_service_flow_scenario(spec: str) -> List[ServiceFlowPeriod]:
    """Parse a preset name or 'SERVICE@start-end,...' (overlap allowed).

    The '@' form deliberately differs from parse_service_scenario's
    'start-end:SERVICE' so the two scenario languages cannot be confused.
    """
    key = spec.strip().lower()
    if key in MULTI_SERVICE_PRESETS:
        return MULTI_SERVICE_PRESETS[key]()

    periods: List[ServiceFlowPeriod] = []
    for item in spec.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        try:
            service_name, interval = chunk.split("@", maxsplit=1)
            start, end = interval.split("-", maxsplit=1)
            service = ServiceType[service_name.strip().upper()]
            periods.append(
                ServiceFlowPeriod(float(start), float(end), service))
        except (ValueError, KeyError) as exc:
            raise ValueError(
                "service flow scenario must be a preset "
                f"({'|'.join(MULTI_SERVICE_PRESETS)}) or 'SERVICE@start-end,...'"
            ) from exc
    if not periods:
        raise ValueError("service flow scenario must not be empty")
    if any(period.start_s < 0.0 for period in periods):
        raise ValueError("service flow periods must start at or after 0")
    if any(period.end_s <= period.start_s for period in periods):
        raise ValueError("service flow periods must have positive duration")
    # Overlap across services is intentionally NOT rejected.
    periods.sort(key=lambda period: (int(period.service), period.start_s))
    return periods


def active_services_for_time(
    timeline: Iterable[ServiceFlowPeriod],
    simulation_time_s: float,
) -> Tuple[ServiceType, ...]:
    """Services whose requirements are in force at simulation_time_s.

    Order follows SERVICE_SLOTS (ServiceType enum order), never activation
    order, so the returned tuple is deterministic.
    """
    periods = list(timeline)
    return tuple(
        service
        for service in SERVICE_SLOTS
        if any(
            period.service == service
            and period.start_s <= simulation_time_s < period.end_s
            for period in periods
        )
    )


def qos_requirement_vector(service: ServiceType | int):
    """Normalized QoS requirement of one service: 4 values in (0, 1].

    Every component is strictly positive for every defined service, which is
    what makes zero-masking of inactive slots unambiguous.
    """
    import numpy as np

    try:
        parsed = ServiceType(int(service))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ServiceType id: {service}") from exc
    qos = QOS_THRESHOLDS[parsed]
    return np.array(
        [
            qos["min_throughput_bps"] / NORMALIZATION["throughput_ref_bps"],
            qos["max_delay_ms"] / NORMALIZATION["delay_ref_ms"],
            qos["min_pdr"],
            qos["max_outage_ms"] / NORMALIZATION["outage_ref_ms"],
        ],
        dtype=np.float32,
    )


def build_service_context(
    timeline: Iterable[ServiceFlowPeriod],
    simulation_time_s: float,
    mode: str = DEFAULT_SERVICE_CONTEXT_MODE,
):
    """Fixed-slot concurrent service requirement block (SERVICE_CONTEXT_DIM).

    Slot layout is SERVICE_SLOTS order, SERVICE_SLOT_FEATURES values each:
    [active, min_throughput_norm, max_delay_norm, min_pdr, max_outage_norm].
    Inactive slots are zero-masked so an inactive service cannot present a
    phantom requirement. mode='none' zeroes the whole block while keeping
    CONTEXT_FEATURE_DIM unchanged, which makes it a checkpoint-compatible
    ablation switch.

    No goal, priority or dominance is derived here: the caller receives raw
    concurrent constraints only.
    """
    import numpy as np

    if mode not in SERVICE_CONTEXT_MODES:
        raise ValueError(f"unsupported service context mode: {mode}")
    values = np.zeros(SERVICE_CONTEXT_DIM, dtype=np.float32)
    if mode == "none":
        return values

    active = active_services_for_time(timeline, simulation_time_s)
    for slot, service in enumerate(SERVICE_SLOTS):
        if service not in active:
            continue
        base = slot * SERVICE_SLOT_FEATURES
        values[base] = 1.0
        values[base + 1:base + SERVICE_SLOT_FEATURES] = qos_requirement_vector(
            service)
    return values


def effective_requirement(
    services: Sequence[ServiceType],
) -> Optional[Dict[str, float]]:
    """Strictest QoS constraint across concurrently active services.

    EVALUATION / REFERENCE METRIC ONLY. Never feed this into the context
    frame, the goal, the lower reward or the upper objective: collapsing
    concurrent requirements into one synthetic service reintroduces exactly
    the hand-written heuristic this architecture is meant to replace, and it
    destroys the attribution the analysis needs -- once merged, no log can say
    which service's which requirement bound the system.

    The formal upper objective evaluates each service independently; see
    UPPER_OBJECTIVE_VERSION and tvt_e2e.service_violation_breakdown.
    """
    active = [ServiceType(int(service)) for service in services]
    if not active:
        return None
    return {
        "min_throughput_bps": max(
            QOS_THRESHOLDS[s]["min_throughput_bps"] for s in active),
        "max_delay_ms": min(QOS_THRESHOLDS[s]["max_delay_ms"] for s in active),
        "min_pdr": max(QOS_THRESHOLDS[s]["min_pdr"] for s in active),
        "max_outage_ms": min(
            QOS_THRESHOLDS[s]["max_outage_ms"] for s in active),
    }


def qos_satisfied_multi(env, services: Sequence[ServiceType]) -> Optional[bool]:
    """Whether every concurrently active requirement holds. Metric only.

    Mirrors the single-service qos_satisfied() validity gate: returns None when
    the KPI cohort is invalid or when no service is active.
    """
    requirement = effective_requirement(services)
    if requirement is None:
        return None
    if not (bool(env.delay_valid) and bool(env.deadline_loss_valid)):
        return None
    # A requirement on min_throughput is a requirement on DELIVERED useful
    # application data, so it is evaluated against the measured logical
    # goodput, never against a path's byte rate.
    throughput = float(env.logical_goodput_bps)
    return bool(
        throughput >= requirement["min_throughput_bps"]
        and float(env.e2e_delay_ms) <= requirement["max_delay_ms"]
        and 1.0 - float(env.deadline_loss_rate) >= requirement["min_pdr"]
        and float(env.time_since_last_rx_ms) <= requirement["max_outage_ms"]
    )


def _serializable_time(value: float):
    """inf-safe time for JSON: strict parsers reject bare Infinity."""
    return value if value == value and abs(value) != float("inf") else str(value)


def serializable_e2e_configuration():
    """E2E-only configuration block.

    Kept separate from serializable_configuration() so goal-ddqn config.json
    output stays byte-identical to previous runs.
    """
    return {
        "architectures": list(ARCHITECTURES),
        "architecture_id": E2E_ARCHITECTURE,
        "checkpoint_schema_version": E2E_CHECKPOINT_SCHEMA_VERSION,
        "reward_version": E2E_REWARD_VERSION,
        "timing_policy": E2E_TIMING_POLICY,
        "history_len": HISTORY_LEN,
        "network_context_dim": NETWORK_CONTEXT_DIM,
        "service_context_dim": SERVICE_CONTEXT_DIM,
        "context_feature_dim": CONTEXT_FEATURE_DIM,
        "context_feature_names": list(CONTEXT_FEATURE_NAMES),
        "radio_slice_indices": list(RADIO_SLICE_INDICES),
        "radio_state_dim": RADIO_STATE_DIM,
        "goal_dim": GOAL_DIM,
        "lower_input_dim": LOWER_INPUT_DIM,
        "service_slots": [service.name for service in SERVICE_SLOTS],
        "service_slot_features": list(SERVICE_SLOT_FEATURE_NAMES),
        "service_context_modes": list(SERVICE_CONTEXT_MODES),
        "default_service_context_mode": DEFAULT_SERVICE_CONTEXT_MODE,
        "default_service_flow_scenario": DEFAULT_SERVICE_FLOW_SCENARIO,
        "service_flow_presets": {
            name: [
                {
                    "service": period.service.name,
                    "start_s": _serializable_time(period.start_s),
                    "end_s": _serializable_time(period.end_s),
                }
                for period in factory()
            ]
            for name, factory in MULTI_SERVICE_PRESETS.items()
        },
        "transformer": asdict(DEFAULT_TRANSFORMER),
        "reward_policies": {
            name: asdict(policy)
            for name, policy in E2E_REWARD_POLICIES.items()
        },
        "default_reward_version": DEFAULT_E2E_REWARD_VERSION,
        "qos_bonus_weight": E2E_QOS_BONUS_WEIGHT,
        "pending_max_entries": PENDING_MAX_ENTRIES,
        "pending_max_age_s": PENDING_MAX_AGE_S,
    }


def _validate_e2e_definitions() -> None:
    """Fail at import time, not hours into a training run."""
    problems: List[str] = []

    if DEFAULT_ARCHITECTURE not in ARCHITECTURES:
        problems.append("DEFAULT_ARCHITECTURE not in ARCHITECTURES")
    if set(SERVICE_SLOTS) != set(ServiceType):
        problems.append("SERVICE_SLOTS must cover every ServiceType exactly once")
    if SERVICE_SLOT_COUNT != len(ServiceType):
        problems.append("SERVICE_SLOT_COUNT != len(ServiceType)")
    if len(SERVICE_SLOT_FEATURE_NAMES) != SERVICE_SLOT_FEATURES:
        problems.append("SERVICE_SLOT_FEATURE_NAMES length mismatch")
    if SERVICE_CONTEXT_DIM != SERVICE_SLOT_COUNT * SERVICE_SLOT_FEATURES:
        problems.append("SERVICE_CONTEXT_DIM mismatch")
    if len(NETWORK_CONTEXT_FEATURE_NAMES) != NETWORK_CONTEXT_DIM:
        problems.append("NETWORK_CONTEXT_FEATURE_NAMES length mismatch")
    if CONTEXT_FEATURE_DIM != (
            VEHICLE_CONTEXT_DIM + SERVICE_CONTEXT_DIM + NETWORK_CONTEXT_DIM):
        problems.append("CONTEXT_FEATURE_DIM mismatch")
    if len(VEHICLE_CONTEXT_FEATURE_NAMES) != VEHICLE_CONTEXT_DIM:
        problems.append("VEHICLE_CONTEXT_FEATURE_NAMES length mismatch")
    for _ref in ("speed_ref_mps", "distance_ref_m"):
        if NORMALIZATION.get(_ref, 0.0) <= 0.0:
            problems.append(f"NORMALIZATION[{_ref!r}] must be positive")
    if len(CONTEXT_FEATURE_NAMES) != CONTEXT_FEATURE_DIM:
        problems.append("CONTEXT_FEATURE_NAMES length mismatch")
    if len(set(RADIO_SLICE_INDICES)) != RADIO_STATE_DIM:
        problems.append("RADIO_SLICE_INDICES must be unique")
    if _RADIO_SLICE_NETWORK_INDICES and (
            max(_RADIO_SLICE_NETWORK_INDICES) >= NETWORK_CONTEXT_DIM):
        problems.append("RADIO_SLICE_INDICES must stay inside the network block")
    if RADIO_SLICE_INDICES and (
            min(RADIO_SLICE_INDICES) < N_SLICE.start
            or max(RADIO_SLICE_INDICES) >= N_SLICE.stop):
        problems.append("RADIO_SLICE_INDICES must resolve inside N_SLICE")
    if VEHICLE_CONTEXT_DIM + SERVICE_CONTEXT_DIM + NETWORK_CONTEXT_DIM != (
            CONTEXT_FEATURE_DIM):
        problems.append("c_t + v_t + n_t must equal CONTEXT_FEATURE_DIM")
    if LOWER_INPUT_DIM != RADIO_STATE_DIM + GOAL_DIM:
        problems.append("LOWER_INPUT_DIM mismatch")
    if ACTION_HOLD_ENABLED:
        problems.append("ACTION_HOLD_ENABLED must be False in this architecture")
    if "remaining_hold_ratio" in NETWORK_CONTEXT_FEATURE_NAMES:
        problems.append("remaining_hold_ratio must not be a context feature")
    # The retired path-selection features must not creep back in: each would
    # silently change the encoder's input meaning while every dimension check
    # above still passed.
    for retired in RETIRED_NETWORK_CONTEXT_FEATURES:
        if retired in NETWORK_CONTEXT_FEATURE_NAMES:
            problems.append(
                f"{retired} is RETIRED and must not be a context feature")
    if "applied_duplication_ratio" not in NETWORK_CONTEXT_FEATURE_NAMES:
        problems.append("applied_duplication_ratio must be a context feature")
    if len(DUPLICATION_LEVELS) != ACTION_DIM:
        problems.append("DUPLICATION_LEVELS length must equal ACTION_DIM")
    if list(DUPLICATION_LEVELS) != sorted(DUPLICATION_LEVELS):
        problems.append("DUPLICATION_LEVELS must be ordinal (ascending)")
    if DUPLICATION_LEVELS[0] != 0.0 or DUPLICATION_LEVELS[-1] != 1.0:
        problems.append("DUPLICATION_LEVELS must span [0, 1] inclusive")
    if len(set(DUPLICATION_LEVELS)) != ACTION_DIM:
        problems.append("DUPLICATION_LEVELS must be distinct")
    if len(ACTION_NAMES) != ACTION_DIM:
        problems.append("ACTION_NAMES length must equal ACTION_DIM")
    if PRIMARY_PATH_INDEX == SECONDARY_PATH_INDEX:
        problems.append("primary and secondary path must differ")
    if DUPLICATION_SCHEDULER_CONSUMES_RNG:
        problems.append(
            "the duplication scheduler must not consume the ns-3 RNG stream")
    if THROUGHPUT_PRIMITIVE_VERSION != "logical_goodput_v1":
        problems.append("THROUGHPUT_PRIMITIVE_VERSION must be logical_goodput_v1")
    if KNOWN_RISK_USEFUL_THROUGHPUT_PROXY:
        problems.append(
            "the max-path-throughput proxy must not feed any reward")
    if PATH_THROUGHPUT_ROLE != "path_condition_observation_only":
        problems.append(
            "per-path throughput is an observation, never a reward input")
    if RESOURCE_COST_VERSION != "attempted_duplication_v1":
        problems.append(
            "RESOURCE_COST_VERSION must stay attempted_duplication_v1: the "
            "formula did not change with the action space")
    if MACRO_INTERVAL_BASIS != "decision_ticks":
        problems.append("MACRO_INTERVAL_BASIS must be decision_ticks")
    if abs(GOAL_INTERVAL_MS - DECISION_INTERVAL_MS * MACRO_HORIZON) > 1e-9:
        problems.append("GOAL_INTERVAL_MS must equal H * DECISION_INTERVAL_MS")
    if UPPER_OBJECTIVE_USES_EFFECTIVE_REQUIREMENT:
        problems.append(
            "the upper objective must evaluate services independently")
    if FORMAL_SERVICE_AGGREGATION_MODE not in SERVICE_AGGREGATION_MODES:
        problems.append("FORMAL_SERVICE_AGGREGATION_MODE unknown")
    if FORMAL_SERVICE_FLOW_SCENARIO not in MULTI_SERVICE_PRESETS:
        problems.append("FORMAL_SERVICE_FLOW_SCENARIO unknown")
    if formal_action_count(FORMAL_EPISODE_LENGTH_S) != 250:
        problems.append(
            "formal_action_count(28.0) must be 250 (ns-3 verified)")
    if FORMAL_MACROS_PER_EPISODE != int(2 * FORMAL_EPISODE_LENGTH_S - 6):
        problems.append("FORMAL_MACROS_PER_EPISODE != floor(2E - 6)")
    if FORMAL_TRAINING_HORIZON < 161:
        problems.append(
            "FORMAL_TRAINING_HORIZON below the structural floor of 161")
    if EPSILON_DECAY_MODE not in EPSILON_DECAY_MODES:
        problems.append("EPSILON_DECAY_MODE unknown")
    if not 0.0 < EPSILON_MIN <= EPSILON_START:
        problems.append("require 0 < EPSILON_MIN <= EPSILON_START")
    if DUPLICATION_PARTIAL_WINDOW_POLICY != "inactive_until_full":
        problems.append("partial-window policy must be inactive_until_full")
    if not DUPLICATION_VIOLATION_NORMALIZED:
        problems.append("the duplication violation must be normalized")
    if DUPLICATION_DUAL_CADENCE != "macro_push":
        problems.append("the dual variable must step once per macro push")
    try:
        assert_seed_separation()
    except RuntimeError as exc:
        problems.append(str(exc))
    if GOAL_EXPLORATION_VERSION != "RETIRED_LOGIT_GAUSSIAN":
        problems.append("logit-space goal exploration must stay retired")
    if GOAL_EXPLORATION_MECHANISM != "SIMPLEX_DIRICHLET_LOCAL_GLOBAL":
        problems.append("formal exploration must be simplex Dirichlet "
                        "local+global mixing")
    if "simplex_dirichlet_mixing_v1" not in RETIRED_EXPLORATION_PROTOCOL_VERSIONS:
        problems.append("the planner-anchored-only exploration must stay retired")
    if not (P_GLOBAL_LOWER_BOUND <= min(P_GLOBAL_CANDIDATES)
            and max(P_GLOBAL_CANDIDATES) <= P_GLOBAL_UPPER_BOUND):
        problems.append("P_GLOBAL_CANDIDATES must lie inside the derived bounds")
    # The anti-saturation bound must stay derived, never hand-set.
    expected_r = (math.log(GOAL_DIM * GOAL_DIM - GOAL_DIM + 1.0)
                  * math.sqrt((GOAL_DIM - 1.0) / GOAL_DIM))
    if abs(PLANNER_LOGIT_BOUND - expected_r) > 1e-12:
        problems.append("PLANNER_LOGIT_BOUND must equal ln(n^2-n+1)*sqrt((n-1)/n)")
    if abs(DIRICHLET_EXPECTED_MIN_COMPONENT - 1.0 / GOAL_DIM ** 2) > 1e-15:
        problems.append("DIRICHLET_EXPECTED_MIN_COMPONENT must equal 1/n^2")
    if PLANNER_PARAMETERIZATION != "centred_ball_projected_logits_v1":
        problems.append("planner parameterization must be the anti-saturation one")
    if PLANNER_LR_SCALED_BY_K:
        problems.append(
            "planner_lr scaling was rolled back: it corrected the measured "
            "timescale mismatch but reduced responsiveness replication to 0/2")
    if "planner_lr_scaled_by_k_v1" not in RETIRED_E2E_TIMESCALE_PROTOCOLS:
        problems.append("the rolled-back lr-scaling protocol must stay retired")
    if abs(DIRICHLET_TRACE_COV - 0.15) > 1e-12:
        problems.append("Dirichlet(1,1,1,1) trace(Cov) must be 0.15 for 4 goals")
    if abs(SIMPLEX_CENTROID_VERTEX_DIST - 0.75 ** 0.5) > 1e-12:
        problems.append("centroid-to-vertex distance must be sqrt(3)/2")
    if not 0.0 < PLANNER_DOMINANT_EXPLORATION_POLICY_MAX_BETA <= 1.0:
        problems.append("planner-dominant beta bound must be in (0, 1]")
    for level in BETA_COVERAGE_LEVELS:
        beta = beta_from_coverage(level)
        if not 0.0 < beta < PLANNER_DOMINANT_EXPLORATION_POLICY_MAX_BETA + 1e-12:
            problems.append(
                f"coverage {level} implies beta {beta} outside the "
                "planner-dominant exploration policy")
        if abs(coverage_from_beta(beta) - level) > 1e-12:
            problems.append("coverage <-> beta map is not invertible")
    if SCREENING_PROTOCOL_VERSION != 2:
        problems.append("SCREENING_PROTOCOL_VERSION must be 2")
    if HORIZON_SENSITIVITY_ARM != "NONE":
        problems.append(
            "the primary protocol carries no horizon-sensitivity arm")
    if not PLANNER_LR_EQUALS_CRITIC_LR:
        problems.append(
            "lr_planner must equal lr_critic; the timescale difference is "
            "expressed only through K and W")
    if HISTORY_LEN < 1 or GOAL_DIM < 1:
        problems.append("HISTORY_LEN and GOAL_DIM must be positive")
    if PENDING_MAX_ENTRIES < 1 or PENDING_MAX_AGE_S <= 0.0:
        problems.append("pending pool bounds must be positive")
    if DEFAULT_SERVICE_CONTEXT_MODE not in SERVICE_CONTEXT_MODES:
        problems.append("DEFAULT_SERVICE_CONTEXT_MODE unknown")
    if DEFAULT_SERVICE_FLOW_SCENARIO not in MULTI_SERVICE_PRESETS:
        problems.append("DEFAULT_SERVICE_FLOW_SCENARIO unknown")
    if DEFAULT_E2E_REWARD_VERSION not in E2E_REWARD_POLICIES:
        problems.append("DEFAULT_E2E_REWARD_VERSION unknown")

    for service in SERVICE_SLOTS:
        if service not in QOS_THRESHOLDS:
            problems.append(f"QOS_THRESHOLDS missing {service.name}")
            continue
        qos = QOS_THRESHOLDS[service]
        normalized = (
            qos["min_throughput_bps"] / NORMALIZATION["throughput_ref_bps"],
            qos["max_delay_ms"] / NORMALIZATION["delay_ref_ms"],
            qos["min_pdr"],
            qos["max_outage_ms"] / NORMALIZATION["outage_ref_ms"],
        )
        # Zero-masking of inactive slots is only unambiguous while every active
        # requirement is strictly positive.
        if any(value <= 0.0 for value in normalized):
            problems.append(
                f"{service.name} requirement must normalize to strictly "
                "positive values for zero-masking to be unambiguous")
        if any(value > 1.0 for value in normalized):
            problems.append(
                f"{service.name} requirement exceeds its normalization "
                "reference")

    for name, policy in E2E_REWARD_POLICIES.items():
        total = (policy.throughput + policy.delay + policy.loss
                 + policy.outage + policy.switch + policy.pingpong)
        if abs(total - 1.0) > 1e-9:
            problems.append(f"E2E reward policy {name} weights sum to {total}")

    if problems:
        raise RuntimeError(
            "invalid E2E configuration: " + "; ".join(problems))


_validate_e2e_definitions()
