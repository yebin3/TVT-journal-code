#!/usr/bin/env python3
"""Dual-connectivity goal-planner architecture.

    V2X context history [B, T, CONTEXT_FEATURE_DIM]
        -> Goal Planner (transformer | mlp | rule | fixed)
        -> normalized QoS goal g_t = [w_rel, w_lat, w_thr, w_cost], sum = 1
        -> concat(radio state [B, RADIO_STATE_DIM], g_t)
        -> Goal-conditioned lower DDQN
        -> Q [B, 3] over {Uu, PC5, Both}
        -> ns-3 -> reward components -> next state

Role separation, which the code enforces rather than merely documents:

  * The planner decides WHAT to optimise right now. It never selects a path.
  * The lower DDQN decides WHICH path(s) satisfy that goal. It reads g_t as an
    explicit input, not as a latent smuggled inside an encoder output.

Two rewards, deliberately separate (see docs in tvt_config):

  * r_lower = g_t . [r_rel, r_lat, r_thr, r_cost]      -- planner-shaped
  * r_upper = 0.25 . [r_rel, r_lat, r_thr, r_cost]     -- fixed, balanced
              the planner is judged by this, so it cannot raise its own score
              by re-weighting what it happens to be good at.

Both = redundant duplication of one logical application packet over both paths.
It is never splitting, striping, alternation or multipath transport.
"""

from __future__ import annotations

import dataclasses
import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tvt_config import (
    ACTION_DIM,
    GOAL_CRITIC_OPERATOR,
    ACTION_NAMES,
    DUPLICATION_LEVELS,
    CONTEXT_FEATURE_DIM,
    DEFAULT_GOAL_PLANNER,
    DEFAULT_SERVICE_CONTEXT_MODE,
    DEFAULT_TRANSFORMER,
    GOAL_DIM,
    bounded_centred_logits,
    duplication_ratio,
    GOAL_PLANNERS,
    GOAL_SUM_TOLERANCE,
    GOAL_WEIGHT_NAMES,
    HISTORY_LEN,
    LOWER_INPUT_DIM,
    LOWER_REWARD_VERSION,
    NETWORK_CONTEXT_DIM,
    OBSERVATION_CONTRACT_VERSION,
    SERVICE_CONTEXT_DIM,
    VEHICLE_CONTEXT_DIM,
    VEHICLE_CONTEXT_FEATURE_NAMES,
    C_SLICE,
    V_SLICE,
    N_SLICE,
    NORMALIZATION,
    PENDING_MAX_AGE_S,
    PENDING_MAX_ENTRIES,
    QOS_THRESHOLDS,
    RADIO_SLICE_INDICES,
    RADIO_STATE_DIM,
    RESOURCE_COST_BASELINE_COPIES,
    RESOURCE_COST_VERSION,
    MACRO_HORIZON,
    MACRO_RETURN_CONVENTION,
    RULE_EPSILON,
    UNMEASURED_SENTINEL,
    SERVICE_AGGREGATION_MODE,
    SYSTEM_AGGREGATION_MODE,
    UPPER_OBJECTIVE_VERSION,
    RELIABILITY_ATTRIBUTION_VERSION,
    RESOURCE_CONSTRAINT_VERSION,
    RESOURCE_CONSTRAINT_METRIC,
    DUPLICATION_LAMBDA_INIT,
    DUPLICATION_NORM_EPS,
    DUPLICATION_PARTIAL_WINDOW_POLICY,
    DUPLICATION_VIOLATION_NORMALIZED,
    DUPLICATION_DUAL_CADENCE,
    BINDING_TIE_LABEL,
    BINDING_TIE_POLICY,
    BINDING_TIE_TOLERANCE,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY_MODE,
    EPSILON_SCHEDULE_VERSION,
    SERVICE_SLOTS,
    GOAL_EXPLORATION_VERSION,
    E2E_TIMESCALE_PROTOCOL,
    EXPLORATION_PROTOCOL_VERSION,
    PLANNER_LOGIT_BOUND,
    PLANNER_LR_SCALED_BY_K,
    PLANNER_PARAMETERIZATION,
    GOAL_EXPLORATION_MECHANISM,
    DIRICHLET_CONCENTRATION,
    exploration_radius_bracket,
    expected_squared_radius,
    TWO_TIMESCALE_VERSION,
    ACTION_HOLD_ENABLED,
    MACRO_TERMINAL_CONVENTION,
    MACRO_INTERVAL_BASIS,
    QOS_THRESHOLDS as _QOS_THRESHOLDS,
    ServiceType,
    TransformerConfig,
    build_service_context,
)

E2E_GRAD_CLIP = 1.0
LOWER_HIDDEN_DIM = 128
# MLP planner width ONLY. Chosen analytically, not by screening: at the
# already-defined MLP depth (41 -> h -> h -> 4) the parameter count is
# h^2 + 47h + 4, and h = 243 is the unique integer minimizing the distance to
# the Transformer planner's 70,532 parameters -> 70,474 = 99.918%. The old
# value 128 gave 22,404 = 31.76%, which failed the +/-20% capacity-fairness
# target for the MLP-vs-TRANSFORMER comparison. Referenced by MLPGoalPlanner
# and nothing else, so the Transformer and GoalCritic are untouched.
PLANNER_HIDDEN_DIM = 243

_FEATURE_TOLERANCE = 1e-6


# ============================================================================
# Context / radio state construction
# ============================================================================
def _normalize_rssi(value: float) -> float:
    low = NORMALIZATION["rssi_min_dbm"]
    high = NORMALIZATION["rssi_max_dbm"]
    return float((np.clip(float(value), low, high) - low) / (high - low))


def _clip01(value: float, reference: float) -> float:
    return float(np.clip(float(value) / float(reference), 0.0, 1.0))


def attempted_copies_per_logical(env) -> float:
    """Copies the action CALLED FOR, per logical packet, over the interval.

    This is the resource-cost basis. It counts what the controller decided to
    transmit, not what the radio managed to accept: a Both decision whose PC5
    send failed still cost a duplication decision and must not be scored as
    cheaply as a single path.
    Returns the single-path baseline when no logical packet was generated.
    """
    logical = float(int(env.window_logical_tx))
    if logical <= 0.0:
        return float(RESOURCE_COST_BASELINE_COPIES)
    return float(int(env.window_attempted_copies)) / logical


def successful_copies_per_logical(env) -> float:
    """Copies a socket actually accepted, per logical packet. Diagnostics."""
    logical = float(int(env.window_logical_tx))
    if logical <= 0.0:
        return 0.0
    return float(int(env.window_successful_copies)) / logical


def duplication_overhead(env) -> float:
    """Attempted duplication above a single copy, clipped to [0, 1]."""
    return float(np.clip(
        attempted_copies_per_logical(env) - RESOURCE_COST_BASELINE_COPIES,
        0.0, 1.0))


def build_context_frame(
    env,
    flow_timeline: Iterable,
    service_context_mode: str = DEFAULT_SERVICE_CONTEXT_MODE,
) -> np.ndarray:
    """One timestep of V2X context: shape (CONTEXT_FEATURE_DIM,) float32.

    Contract 8 layout, physical order == the paper's logical order:

        x_t = [ c_t (2) | v_t (20) | n_t (19) ]   -> 41 features

      c_t  mobility context    ego_speed_norm, ego_rsu_distance_norm
      v_t  service requirements the concurrent requirement block
      n_t  network/KPI context  the per-path and aggregate measurements

    c_t is consumed by the UPPER tier only. build_radio_state() slices the
    seven lower-controller features out of n_t via RADIO_SLICE_INDICES, so the
    lower DDQN never receives mobility.

    The mobility block carries NO missing-value encoding: ns-3 aborts if the
    quantities are undefined rather than emitting a sentinel, because 0 m/s and
    0 m are both legitimate observations. All other sentinel handling for the
    ns-3 observation lives here and nowhere else:

      * e2e_delay_ms == -1        -> delay_norm 0.0 with delay_valid 0.0
      * deadline_loss invalid     -> loss_norm 0.0 with deadline_loss_valid 0.0
      * per-path delay == -1      -> 0.0 with the path's meas_valid flag 0.0
      * time_since_last_rx 1e9    -> rx_gap_norm clips to 1.0

    Both paths are reported on every frame regardless of the duplication degree
    in force, so a path that is currently carrying no application data never
    goes stale -- the 100 ms measurement probe covers it.
    """
    action = int(env.current_action)
    uu_valid = bool(int(env.uu_measurement_valid))
    pc5_valid = bool(int(env.pc5_measurement_valid))
    network = np.array(
        [
            _normalize_rssi(env.uu_rssi),                                  # 0
            _normalize_rssi(env.pc5_rssi),                                 # 1
            # ORDINAL scalar, not a one-hot: the duplication degree has an
            # order and a distance, and encoding it as a category would make
            # 0 -> 25% look exactly as far as 0 -> 100%.
            duplication_ratio(action),                                     # 2
            (
                _clip01(env.e2e_delay_ms, NORMALIZATION["delay_ref_ms"])
                if bool(env.delay_valid)
                else 0.0
            ),                                                             # 3
            (
                _clip01(env.deadline_loss_rate, NORMALIZATION["loss_ref_rate"])
                if bool(env.deadline_loss_valid)
                else 0.0
            ),                                                             # 4
            _clip01(env.uu_throughput_bps,
                    NORMALIZATION["throughput_ref_bps"]),                  # 5
            _clip01(env.pc5_throughput_bps,
                    NORMALIZATION["throughput_ref_bps"]),                  # 6
            # Unmeasured -> neutral midpoint, never 0.0: on a "lower is
            # better" feature 0.0 would advertise an unmeasured path as the
            # best possible one.
            (
                _clip01(env.uu_delay_ms, NORMALIZATION["delay_ref_ms"])
                if uu_valid and float(env.uu_delay_ms) >= 0.0
                else UNMEASURED_SENTINEL
            ),                                                             # 7
            (
                _clip01(env.pc5_delay_ms, NORMALIZATION["delay_ref_ms"])
                if pc5_valid and float(env.pc5_delay_ms) >= 0.0
                else UNMEASURED_SENTINEL
            ),                                                             # 8
            (float(np.clip(float(env.uu_loss_rate), 0.0, 1.0))
             if uu_valid else UNMEASURED_SENTINEL),                        # 9
            (float(np.clip(float(env.pc5_loss_rate), 0.0, 1.0))
             if pc5_valid else UNMEASURED_SENTINEL),                       # 10
            float(uu_valid),                                               # 11
            float(pc5_valid),                                              # 12
            _clip01(env.time_since_last_rx_ms,
                    NORMALIZATION["outage_ref_ms"]),                       # 13
            # No remaining-hold feature and no is_switch: the hold is gone, and
            # "the path changed" denotes nothing when the path never changes.
            # The duplication-level analogue is a logged diagnostic, never a
            # state feature and never a reward term.
            float(bool(int(env.delay_valid))),                             # 14
            float(bool(int(env.deadline_loss_valid))),                     # 15
            # MEASURED overhead, as distinct from the COMMANDED ratio at 2.
            duplication_overhead(env),                                     # 16
            # freshness, so a stale-but-valid measurement is distinguishable
            # from a fresh one (the validity bit alone cannot express this)
            _clip01(env.uu_measurement_age_ms,
                    NORMALIZATION["outage_ref_ms"]),                        # 17
            _clip01(env.pc5_measurement_age_ms,
                    NORMALIZATION["outage_ref_ms"]),                        # 18
        ],
        dtype=np.float32,
    )
    if network.shape != (NETWORK_CONTEXT_DIM,):
        raise ValueError(
            f"network context block must be {NETWORK_CONTEXT_DIM}-dim, "
            f"got {network.shape}")

    # ---- c_t: mobility context -------------------------------------------
    # Raw values come straight from the replayed-trace mobility model, sampled
    # by ns-3 before this tick's goal and action are chosen. References are
    # frozen scenario design constants, never fitted to an outcome.
    speed_mps = float(env.ego_speed_mps)
    distance_m = float(env.ego_rsu_distance_m)
    if not (math.isfinite(speed_mps) and math.isfinite(distance_m)):
        raise ValueError(
            f"mobility context is not finite: speed={speed_mps}, "
            f"distance={distance_m}")
    if speed_mps < 0.0 or distance_m < 0.0:
        raise ValueError(
            f"mobility context must be non-negative: speed={speed_mps}, "
            f"distance={distance_m}")
    vehicle = np.array(
        [
            _clip01(speed_mps, NORMALIZATION["speed_ref_mps"]),
            _clip01(distance_m, NORMALIZATION["distance_ref_m"]),
        ],
        dtype=np.float32,
    )
    if vehicle.shape != (VEHICLE_CONTEXT_DIM,):
        raise ValueError(
            f"mobility context must be {VEHICLE_CONTEXT_DIM}-dim, "
            f"got {vehicle.shape}")

    service = build_service_context(
        flow_timeline, float(env.simulation_time_s), service_context_mode)
    frame = np.concatenate((vehicle, service, network)).astype(np.float32)

    if frame.shape != (CONTEXT_FEATURE_DIM,):
        raise ValueError(
            f"context frame must be {CONTEXT_FEATURE_DIM}-dim, got {frame.shape}")
    if not np.all(np.isfinite(frame)):
        raise ValueError(f"context frame is not finite: {frame}")
    if frame.min() < -_FEATURE_TOLERANCE or frame.max() > 1.0 + _FEATURE_TOLERANCE:
        raise ValueError(
            f"context frame must stay in [0, 1]; got range "
            f"[{frame.min()}, {frame.max()}]")
    # Each semantic block must land exactly on its declared slice.
    if frame[C_SLICE].shape != (VEHICLE_CONTEXT_DIM,):
        raise ValueError("c_t block is misplaced in the context frame")
    if frame[V_SLICE].shape != (SERVICE_CONTEXT_DIM,):
        raise ValueError("v_t block is misplaced in the context frame")
    if frame[N_SLICE].shape != (NETWORK_CONTEXT_DIM,):
        raise ValueError("n_t block is misplaced in the context frame")
    return frame


def mobility_context_raw(env) -> Dict[str, float]:
    """Raw and normalized c_t, for logging and auditing.

    Returned so logs can record what the simulator reported alongside what the
    network actually consumed, rather than only anonymous tensor positions.
    """
    speed_mps = float(env.ego_speed_mps)
    distance_m = float(env.ego_rsu_distance_m)
    return {
        "ego_speed_mps": speed_mps,
        "ego_speed_norm": float(_clip01(speed_mps,
                                        NORMALIZATION["speed_ref_mps"])),
        "ego_rsu_distance_m": distance_m,
        "ego_rsu_distance_norm": float(
            _clip01(distance_m, NORMALIZATION["distance_ref_m"])),
    }


def build_radio_state(frame: np.ndarray) -> np.ndarray:
    """Lower-DDQN state, sliced from an already built context frame.

    Slicing (instead of re-reading env) keeps normalization in exactly one
    place, so the planner and the controller can never drift apart.
    """
    values = np.asarray(frame, dtype=np.float32)
    if values.shape != (CONTEXT_FEATURE_DIM,):
        raise ValueError(
            f"radio state needs a {CONTEXT_FEATURE_DIM}-dim frame, "
            f"got {values.shape}")
    radio = values[list(RADIO_SLICE_INDICES)].astype(np.float32).copy()
    if radio.shape != (RADIO_STATE_DIM,):
        raise ValueError(f"radio state must be {RADIO_STATE_DIM}-dim")
    return radio


class ContextHistory:
    """Rolling context window, left zero-padded until it fills up."""

    def __init__(
        self,
        history_len: int = HISTORY_LEN,
        feature_dim: int = CONTEXT_FEATURE_DIM,
    ):
        if history_len < 1 or feature_dim < 1:
            raise ValueError("history_len and feature_dim must be positive")
        self.history_len = int(history_len)
        self.feature_dim = int(feature_dim)
        self._frames: deque = deque(maxlen=self.history_len)

    def append(self, frame: np.ndarray) -> None:
        values = np.asarray(frame, dtype=np.float32)
        if values.shape != (self.feature_dim,):
            raise ValueError(
                f"context frame must be {self.feature_dim}-dim, "
                f"got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("context frame is not finite")
        self._frames.append(values.copy())

    def snapshot(self) -> np.ndarray:
        """[history_len, feature_dim]; newest frame is the last row."""
        window = np.zeros(
            (self.history_len, self.feature_dim), dtype=np.float32)
        if self._frames:
            filled = np.stack(self._frames)
            window[-filled.shape[0]:] = filled
        return window

    def reset(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)


# ============================================================================
# Goal planners
# ============================================================================
def validate_goal(weights) -> None:
    """A goal vector must be a genuine normalized preference mixture."""
    array = (weights.detach().cpu().numpy()
             if isinstance(weights, torch.Tensor) else np.asarray(weights))
    values = np.atleast_2d(np.asarray(array, dtype=np.float64))
    if values.shape[-1] != GOAL_DIM:
        raise ValueError(f"goal must be {GOAL_DIM}-dim, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("goal weights are not finite")
    if values.min() < -GOAL_SUM_TOLERANCE:
        raise ValueError(f"goal weights must be >= 0, got min {values.min()}")
    sums = values.sum(axis=-1)
    if np.any(np.abs(sums - 1.0) > GOAL_SUM_TOLERANCE):
        raise ValueError(f"goal weights must sum to 1, got {sums}")


class TransformerGoalPlanner(nn.Module):
    """Temporal Goal Planner: context history -> normalized QoS goal.

    It reads the whole [T, F] window through a transformer encoder, so it can
    condition on how the situation has been developing, not only on the newest
    frame. It never emits an action.
    """

    kind = "transformer"
    consumes_history = True
    supports_exploration = True

    def __init__(
        self,
        feature_dim: int = CONTEXT_FEATURE_DIM,
        history_len: int = HISTORY_LEN,
        goal_dim: int = GOAL_DIM,
        config: TransformerConfig = DEFAULT_TRANSFORMER,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.history_len = int(history_len)
        self.goal_dim = int(goal_dim)
        self.config = config

        self.input_proj = nn.Linear(self.feature_dim, config.d_model)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.history_len, config.d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.goal_head = nn.Linear(config.d_model, self.goal_dim)

    def raw_logits(self, context_history: torch.Tensor) -> torch.Tensor:
        """Unbounded head output, before centring and bounding."""
        if context_history.dim() != 3:
            raise ValueError(
                f"context history must be [B, T, F], got {tuple(context_history.shape)}")
        if context_history.shape[1] != self.history_len:
            raise ValueError(
                f"context history must have {self.history_len} timesteps")
        if context_history.shape[2] != self.feature_dim:
            raise ValueError(
                f"context history must have {self.feature_dim} features")
        hidden = self.input_proj(context_history) + self.pos_embedding
        encoded = self.encoder(hidden)
        pooled = encoded[:, -1, :]          # newest timestep
        return self.goal_head(pooled)

    def logits(self, context_history: torch.Tensor) -> torch.Tensor:
        """Centred, bounded pre-softmax scores -- the ANTI-SATURATION
        parameterization. Shared with the MLP planner so the two cannot drift
        into different output semantics."""
        return bounded_centred_logits(self.raw_logits(context_history))

    def forward(self, context_history: torch.Tensor) -> torch.Tensor:
        # softmax makes the output a preference mixture, not a mode choice
        return torch.softmax(self.logits(context_history), dim=-1)


class MLPGoalPlanner(nn.Module):
    """Memoryless neural Goal Planner.

    Reads ONLY the current decision-time context frame. The history is
    deliberately not flattened into it: the comparison this baseline exists for
    is "memoryless neural weighting vs temporal Transformer weighting", so
    giving it the whole window in another shape would defeat the purpose.
    Output semantics are identical to the Transformer planner.
    """

    kind = "mlp"
    consumes_history = False
    supports_exploration = True

    def __init__(
        self,
        feature_dim: int = CONTEXT_FEATURE_DIM,
        history_len: int = HISTORY_LEN,
        goal_dim: int = GOAL_DIM,
        hidden_dim: int = PLANNER_HIDDEN_DIM,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.history_len = int(history_len)
        self.goal_dim = int(goal_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.goal_dim),
        )

    def raw_logits(self, context_history: torch.Tensor) -> torch.Tensor:
        """Unbounded head output, before centring and bounding."""
        if context_history.dim() != 3:
            raise ValueError(
                f"context history must be [B, T, F], got {tuple(context_history.shape)}")
        current = context_history[:, -1, :]   # newest frame only
        return self.net(current)

    def logits(self, context_history: torch.Tensor) -> torch.Tensor:
        """Centred, bounded pre-softmax scores -- identical semantics to the
        Transformer planner, by construction."""
        return bounded_centred_logits(self.raw_logits(context_history))

    def forward(self, context_history: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(context_history), dim=-1)


class FixedGoalPlanner(nn.Module):
    """Constant goal vector, independent of context.

    The formal FIXED baseline is [0.25, 0.25, 0.25, 0.25] -- the same balanced
    mixture as the upper objective, which makes it the natural "no planning"
    control. It has no trainable parameters, so the planner gradient is
    structurally zero and only the lower DDQN learns.
    """

    kind = "fixed"
    consumes_history = False
    # No logits and nothing to explore: FIXED is the constant-preference
    # control, and RULE is a deterministic mapping. Adding noise to either
    # would change what the baseline IS.
    supports_exploration = False

    def __init__(self, weights: Sequence[float], goal_dim: int = GOAL_DIM,
                 history_len: int = HISTORY_LEN):
        super().__init__()
        self.history_len = int(history_len)
        if weights is None:
            from tvt_config import FIXED_GOAL_WEIGHTS
            weights = FIXED_GOAL_WEIGHTS
        values = np.asarray(weights, dtype=np.float32)
        if values.shape != (goal_dim,):
            raise ValueError(f"fixed goal must be {goal_dim}-dim")
        validate_goal(values)
        self.goal_dim = int(goal_dim)
        self.register_buffer("weights", torch.as_tensor(values))

    def forward(self, context_history: torch.Tensor) -> torch.Tensor:
        batch = context_history.shape[0]
        return self.weights.unsqueeze(0).expand(batch, self.goal_dim)


class RuleGoalPlanner(nn.Module):
    """Deterministic hand-written Goal Planner.

    Weights each objective by its current normalized deficit (1 - r_i), so the
    objective doing worst gets the most weight. No thresholds, no learned
    parameters, no history, nothing to tune -- see
    connectivity_planner.rule_goal_mapping.
    """

    kind = "rule"
    consumes_history = False
    supports_exploration = False

    def __init__(self, mapping=None, goal_dim: int = GOAL_DIM,
                 thresholds: Optional[Dict] = None,
                 history_len: int = HISTORY_LEN):
        super().__init__()
        self.history_len = int(history_len)
        if mapping is None:
            from connectivity_planner import rule_goal_mapping
            mapping = rule_goal_mapping
        self.mapping = mapping
        self.goal_dim = int(goal_dim)
        self.thresholds = thresholds if thresholds is not None else QOS_THRESHOLDS

    def forward(self, context_history: torch.Tensor) -> torch.Tensor:
        frames = context_history[:, -1, :].detach().cpu().numpy()
        rows = []
        for frame in frames:
            scores = np.asarray(self.mapping(frame), dtype=np.float64)
            if scores.shape != (self.goal_dim,):
                raise ValueError(
                    f"rule mapping must return {self.goal_dim} scores")
            if not np.all(np.isfinite(scores)) or scores.min() < 0.0:
                raise ValueError("rule mapping must return finite scores >= 0")
            total = scores.sum()
            rows.append(scores / total if total > 0.0
                        else np.full(self.goal_dim, 1.0 / self.goal_dim))
        return torch.as_tensor(
            np.stack(rows), dtype=context_history.dtype,
            device=context_history.device)


class ShuffledTransformerGoalPlanner(TransformerGoalPlanner):
    """Ablation: same parameters, same input frames, temporal order destroyed.

    This is the control that separates "temporal modelling" from "more
    parameters": it has identical capacity and identical information content,
    and differs only in that the window is permuted before encoding.
    """

    kind = "transformer_shuffled"

    def raw_logits(self, context_history: torch.Tensor) -> torch.Tensor:
        perm = torch.randperm(context_history.shape[1],
                              device=context_history.device)
        return super().raw_logits(context_history[:, perm, :])


def make_goal_planner(kind: str, **kwargs) -> nn.Module:
    if kind not in GOAL_PLANNERS:
        raise ValueError(f"unknown goal planner: {kind}")
    if kind == "transformer":
        return TransformerGoalPlanner(
            config=kwargs.get("config", DEFAULT_TRANSFORMER),
            history_len=kwargs.get("history_len", HISTORY_LEN))
    if kind == "transformer_shuffled":
        return ShuffledTransformerGoalPlanner(
            config=kwargs.get("config", DEFAULT_TRANSFORMER),
            history_len=kwargs.get("history_len", HISTORY_LEN))
    if kind == "transformer_h1":
        # same architecture family, history length 1
        return TransformerGoalPlanner(
            config=kwargs.get("config", DEFAULT_TRANSFORMER), history_len=1)
    if kind == "mlp":
        return MLPGoalPlanner(history_len=kwargs.get("history_len", HISTORY_LEN))
    if kind == "fixed":
        return FixedGoalPlanner(
            kwargs.get("fixed_weights"),
            history_len=kwargs.get("history_len", HISTORY_LEN))
    return RuleGoalPlanner(
        kwargs.get("rule_mapping"),
        history_len=kwargs.get("history_len", HISTORY_LEN))


# ============================================================================
# Goal-conditioned lower controller
# ============================================================================
class GoalConditionedDDQN(nn.Module):
    """Q(s_t, g_t, a_t) over the five packet-duplication degrees.

    The action is ordinal, but the Q head stays DISCRETE with one output per
    level: discreteness is what makes this a DDQN at all, and a regression head
    over a continuous ratio would be a different algorithm, not a smaller
    change. The ordering is expressed in the STATE, through the scalar
    applied_duplication_ratio, not in the output layer.

    g_t enters as an explicit input concatenated to the radio state. It is not
    hidden inside a latent, which is what makes "goal-conditioned control" a
    property of the code rather than of the prose.
    """

    def __init__(
        self,
        radio_dim: int = RADIO_STATE_DIM,
        goal_dim: int = GOAL_DIM,
        hidden_dim: int = LOWER_HIDDEN_DIM,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self.radio_dim = int(radio_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.radio_dim + self.goal_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

    def forward(self, radio_state: torch.Tensor,
                goal: torch.Tensor) -> torch.Tensor:
        if radio_state.shape[-1] != self.radio_dim:
            raise ValueError(f"radio state must be {self.radio_dim}-dim")
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(f"goal must be {self.goal_dim}-dim")
        return self.net(torch.cat((radio_state, goal), dim=-1))


class GoalCritic(nn.Module):
    """Upper critic  Q_goal(context, goal).

    It deliberately does NOT take the lower action. Its meaning is

        Q_goal(c, g) = E[ macro system return | context c, goal g held for H
                          micro steps, a_micro ~ pi_lower(.|s, g) ]

    so the lower controller's response is folded into the return rather than
    supplied as an input. That is what makes dQ_goal/dg a causal quantity:
    changing g changes which actions the lower policy takes, and the critic
    has learned what that is worth. The previous design conditioned on a
    historical action, which made g causally inert given (s, a).
    """

    def __init__(
        self,
        feature_dim: int = CONTEXT_FEATURE_DIM,
        history_len: int = HISTORY_LEN,
        goal_dim: int = GOAL_DIM,
        hidden_dim: int = LOWER_HIDDEN_DIM,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.history_len = int(history_len)
        self.goal_dim = int(goal_dim)
        # The context is summarised by its newest frame plus a mean over the
        # window: enough to condition on, cheap, and identical for every
        # planner variant so the critic is never a confound.
        self.input_dim = 2 * self.feature_dim + self.goal_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_history: torch.Tensor,
                goal: torch.Tensor) -> torch.Tensor:
        if context_history.dim() != 3:
            raise ValueError("context history must be [B, T, F]")
        if goal.shape[-1] != self.goal_dim:
            raise ValueError(f"goal must be {self.goal_dim}-dim")
        summary = torch.cat(
            (context_history[:, -1, :], context_history.mean(dim=1)), dim=-1)
        return self.net(torch.cat((summary, goal), dim=-1)).squeeze(-1)


class ConnectivityQNetwork(nn.Module):
    """Goal Planner + goal-conditioned DDQN as one differentiable module.

    forward() returns (q_values, goal). When the planner is trainable the goal
    is NOT detached, so the TD loss differentiates through the planner head.
    """

    def __init__(
        self,
        planner_kind: str = DEFAULT_GOAL_PLANNER,
        feature_dim: int = CONTEXT_FEATURE_DIM,
        history_len: int = HISTORY_LEN,
        radio_dim: int = RADIO_STATE_DIM,
        goal_dim: int = GOAL_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = LOWER_HIDDEN_DIM,
        **planner_kwargs,
    ):
        super().__init__()
        self.planner_kind = planner_kind
        self.planner = make_goal_planner(
            planner_kind, history_len=history_len, **planner_kwargs)
        self.lower = GoalConditionedDDQN(
            radio_dim=radio_dim, goal_dim=goal_dim,
            hidden_dim=hidden_dim, action_dim=action_dim)

    @property
    def planner_is_trainable(self) -> bool:
        return any(p.requires_grad for p in self.planner.parameters())

    def forward(
        self, context_history: torch.Tensor, radio_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        goal = self.planner(context_history)
        return self.lower(radio_state, goal), goal

    def parameter_counts(self) -> Dict[str, int]:
        return {
            "planner": sum(p.numel() for p in self.planner.parameters()),
            "planner_trainable": sum(
                p.numel() for p in self.planner.parameters() if p.requires_grad),
            "lower": sum(p.numel() for p in self.lower.parameters()),
            "total_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad),
        }


# ============================================================================
# Decision bookkeeping
# ============================================================================
class PendingPool:
    """decision_id keyed, bounded, age-gated pending decisions.

    Keying by decision_id lets a late APPLIED outcome still be consumed. Age
    gating (not "drop everything older than the reported id") is required
    because applied_decision_id is not monotonic.
    """

    def __init__(
        self,
        max_entries: int = PENDING_MAX_ENTRIES,
        max_age_s: float = PENDING_MAX_AGE_S,
    ):
        if int(max_entries) < 1:
            raise ValueError("max_entries must be positive")
        if float(max_age_s) <= 0.0:
            raise ValueError("max_age_s must be positive")
        self.max_entries = int(max_entries)
        self.max_age_s = float(max_age_s)
        self._items: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self.expired_count = 0
        self.overflow_count = 0
        self.unmatched_count = 0

    def put(self, decision_id: int, entry: Dict[str, Any]) -> None:
        key = int(decision_id)
        if key <= 0:
            raise ValueError("decision_id must be positive (ns-3 ids start at 1)")
        item = dict(entry)
        item["decision_id"] = key
        self._items[key] = item
        self._items.move_to_end(key)
        self._evict(item.get("sim_time_s"))

    def peek(self, decision_id: int) -> Optional[Dict[str, Any]]:
        key = int(decision_id)
        if key <= 0:
            return None
        return self._items.get(key)

    def consume(self, decision_id: int) -> Optional[Dict[str, Any]]:
        key = int(decision_id)
        if key <= 0:
            return None
        entry = self._items.pop(key, None)
        if entry is None:
            self.unmatched_count += 1
        return entry

    def _evict(self, now_s: Optional[float]) -> None:
        if now_s is not None:
            cutoff = float(now_s) - self.max_age_s
            stale = [
                key
                for key, item in self._items.items()
                if item.get("sim_time_s") is not None
                and float(item["sim_time_s"]) < cutoff
            ]
            for key in stale:
                self._items.pop(key, None)
                self.expired_count += 1
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
            self.overflow_count += 1

    def reset(self) -> None:
        self._items.clear()

    def diagnostics(self) -> Dict[str, int]:
        return {
            "pending_size": len(self._items),
            "expired_count": self.expired_count,
            "overflow_count": self.overflow_count,
            "unmatched_count": self.unmatched_count,
        }

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, decision_id: object) -> bool:
        try:
            return int(decision_id) in self._items  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False


# ============================================================================
# Reward
# ============================================================================
@dataclass(frozen=True)
class RewardResult:
    """Everything needed to reproduce and analyse one reward."""

    lower_total: float                 # g_t . components   (planner-shaped)
    # STEP-LEVEL DIAGNOSTIC U_qos on the interval measurement. The macro
    # return is built from matured tick cohorts instead; see
    # cohort_upper_objective.
    upper_total: float
    components: Dict[str, float]       # r_rel / r_lat / r_thr / r_cost
    goal_weights: Dict[str, float]     # w_rel / w_lat / w_thr / w_cost
    raw_kpi: Dict[str, float]
    violations: Dict[str, float]       # BINDING service's per-KPI violation
    service_violations: Dict[str, Dict[str, float]]   # every active service
    binding_service: Optional[str]     # service name, or "TIE"
    binding_services: List[str]        # every service holding the worst V
    binding_kpi: Optional[str]         # and which of its requirements
    v_system: float                    # max_service(max_kpi(violation))
    duplication_rate: float            # constraint metric, NOT in the objective
    lower_version: str
    upper_version: str


def compute_reward_components(env, applied_action: int) -> Optional[Dict[str, float]]:
    """The four normalized components, all 'larger = better'.

    Returns None when the KPI cohort is not measurable yet, matching the
    validity gate the previous architecture used.
    """
    if not 0 <= int(applied_action) < ACTION_DIM:
        raise ValueError(f"invalid duplication action: {applied_action}")
    if not (bool(env.delay_valid) and bool(env.deadline_loss_valid)):
        return None

    delay_ms = float(env.e2e_delay_ms)
    loss_rate = float(env.deadline_loss_rate)
    if (
        not np.isfinite(delay_ms)
        or delay_ms < 0.0
        or not np.isfinite(loss_rate)
        or not 0.0 <= loss_rate <= 1.0
    ):
        return None

    # AUTHORITATIVE useful throughput. Reads ns-3's first-arrival logical
    # payload accounting directly, so a sequence number counts once however
    # many copies arrived and header/probe bytes never enter.
    #
    # RETIRED: max(uu_throughput_bps, pc5_throughput_bps). Those are per-PATH
    # byte rates that include duplicate arrivals, and a logical packet is
    # delivered when EITHER path delivers it first -- a union, which a maximum
    # cannot represent. It understated the real quantity by up to 21.5%, with
    # the bias monotone in the duplication degree. The path rates remain in the
    # observation as PATH-CONDITION features; they are not goodput.
    useful_bps = float(env.logical_goodput_bps)

    return {
        "r_rel": float(1.0 - loss_rate),
        "r_lat": float(1.0 - _clip01(delay_ms, NORMALIZATION["delay_ref_ms"])),
        "r_thr": float(_clip01(useful_bps, NORMALIZATION["throughput_ref_bps"])),
        "r_cost": float(1.0 - duplication_overhead(env)),
    }


def kpi_violations(
    delay_ms: Optional[float],
    pdr: float,
    throughput_bps: float,
    outage_ms: float,
    requirement: Dict[str, float],
) -> Dict[str, float]:
    """Normalized violation of ONE service's requirement, per KPI.

    0 = the requirement is met, 1 = maximal violation. Every form is bounded,
    coefficient-free and defined only by the requirement itself, so no new
    tunable number enters the system objective.

        delay      viol = clip01( 1 - max_delay / delay )        (lower better)
        pdr        viol = clip01( (min_pdr - pdr) / min_pdr )    (higher better)
        throughput viol = clip01( (min_thr - thr) / min_thr )    (higher better)
        outage     viol = clip01( 1 - max_outage / outage )      (lower better)

    delay_ms is None when NOTHING in the cohort was ever delivered. That is a
    real and maximally bad outcome, not a missing sample: it scores 1.0. It
    must never be dropped as "unmeasurable", which would systematically
    discard the worst outcomes and bias the objective upward.
    """
    eps = 1e-9
    max_delay = float(requirement["max_delay_ms"])
    min_pdr = float(requirement["min_pdr"])
    min_thr = float(requirement["min_throughput_bps"])
    max_outage = float(requirement["max_outage_ms"])
    if delay_ms is None or not np.isfinite(delay_ms):
        delay_violation = 1.0
    else:
        delay_violation = float(
            np.clip(1.0 - max_delay / max(float(delay_ms), eps), 0.0, 1.0))
    return {
        "delay": delay_violation,
        "pdr": float(np.clip((min_pdr - float(pdr)) / max(min_pdr, eps),
                             0.0, 1.0)),
        "throughput": float(np.clip(
            (min_thr - float(throughput_bps)) / max(min_thr, eps), 0.0, 1.0)),
        "outage": float(np.clip(
            1.0 - max_outage / max(float(outage_ms), eps), 0.0, 1.0)),
    }


_KPI_ORDER: Tuple[str, ...] = ("delay", "pdr", "throughput", "outage")


def aggregate_kpi(violations: Dict[str, float],
                  mode: str = SERVICE_AGGREGATION_MODE) -> float:
    """Combine one service's four KPI violations into V_service."""
    values = [violations[name] for name in _KPI_ORDER]
    if mode == "worst":
        return float(max(values))
    if mode == "mean":
        return float(np.mean(values))
    raise ValueError(f"unknown service aggregation mode: {mode}")


def service_violation_breakdown(
    delay_ms: Optional[float],
    pdr: float,
    throughput_bps: float,
    outage_ms: float,
    services: Sequence,
    mode: str = SERVICE_AGGREGATION_MODE,
) -> Optional[Dict[str, Any]]:
    """Service-wise system objective, with full attribution.

        V_s      = aggregate_kpi( violations of service s vs ITS OWN spec )
        V_system = max_s V_s                     <- always max across services
        U_qos    = 1 - V_system

    Each active service is evaluated independently against its own declared
    requirement. The synthetic strictest-of-all merge (effective_requirement)
    is deliberately NOT used: it is at best equivalent under a max aggregator
    and it destroys attribution, so no log could then say which service's
    which requirement bound the system.

    Cross-service aggregation is always max, never mean: a met video
    requirement must not dilute a violated safety requirement.

    Returns None only when no service is active.
    """
    active = [ServiceType(int(service)) for service in services]
    if not active:
        return None

    per_service: Dict[str, Dict[str, float]] = {}
    for service in active:
        violations = kpi_violations(delay_ms, pdr, throughput_bps, outage_ms,
                                    _QOS_THRESHOLDS[service])
        entry = dict(violations)
        entry["V_service"] = aggregate_kpi(violations, mode)
        per_service[service.name] = entry

    v_system = max(entry["V_service"] for entry in per_service.values())
    # Preserve the tie instead of silently attributing it to whichever service
    # sits earlier in SERVICE_SLOTS. V_system is untouched: max does not depend
    # on which tied name is reported.
    _slot_order = {service.name: index
                   for index, service in enumerate(SERVICE_SLOTS)}
    binding_services = sorted(
        (name for name, entry in per_service.items()
         if abs(float(entry["V_service"]) - v_system) <= BINDING_TIE_TOLERANCE),
        key=lambda name: _slot_order.get(name, len(_slot_order)))
    tied = len(binding_services) > 1
    binding_service = BINDING_TIE_LABEL if tied else binding_services[0]
    # The per-KPI breakdown has to come from one service. Under a tie the
    # first in slot order is used -- deterministic but arbitrary, and lossless
    # because per_service carries every service's full breakdown.
    binding = per_service[binding_services[0]]
    v_system = float(v_system)
    binding_kpi = max(_KPI_ORDER, key=lambda name: binding[name])
    return {
        "per_service": per_service,
        "binding_service": binding_service,
        "binding_services": list(binding_services),
        "binding_is_tie": bool(tied),
        "binding_kpi": binding_kpi,
        "binding_violations": {name: float(binding[name])
                               for name in _KPI_ORDER},
        "V_system": v_system,
        "u_qos": float(1.0 - v_system),
        "active_services": [service.name for service in active],
        "binding_tie_policy": BINDING_TIE_POLICY,
        "kpi_aggregation": mode,
        "system_aggregation": SYSTEM_AGGREGATION_MODE,
    }


# ============================================================================
# Macro-attributed reliability cohorts
# ============================================================================
@dataclass(frozen=True)
class TickCohort:
    """One decision tick's logical packets, credited to the macro that sent it.

    ns-3 publishes a cohort only once every packet it originated has reached
    its deadline, so the outcome is attributed to the goal that caused it and
    not to whichever goal happened to be in force ~250 ms later. Throughput
    and outage are snapshotted over exactly the window the cohort was
    transmitted in, so all four KPIs describe the same decision.
    """

    macro_id: int
    macro_tick: int
    action: int
    logical_tx: int
    deadline_success: int
    deadline_miss: int
    delay_sum_ms: float
    delay_count: int
    attempted_copies: int
    duplicated_logical_packets: int
    uu_throughput_bps: float      # path-condition observation
    pc5_throughput_bps: float     # path-condition observation
    logical_goodput_bps: float    # the cohort's USEFUL throughput
    time_since_last_rx_ms: float
    backlog: int = 0

    @property
    def matured(self) -> int:
        return int(self.deadline_success) + int(self.deadline_miss)

    @property
    def valid(self) -> bool:
        """A cohort is evaluable as soon as any packet reached its deadline.

        Not "as soon as one was delivered": a cohort in which everything was
        lost is the most informative outcome there is.
        """
        return self.matured > 0

    @property
    def mean_delay_ms(self) -> Optional[float]:
        if self.delay_count <= 0:
            return None            # nothing arrived at all -> maximal violation
        return float(self.delay_sum_ms) / float(self.delay_count)

    @property
    def pdr(self) -> float:
        return (float(self.deadline_success) / float(self.matured)
                if self.matured > 0 else 0.0)

    @property
    def useful_throughput_bps(self) -> float:
        """The cohort's measured logical goodput over its own window.

        Same definition as the dense lower measurement, snapshotted over the
        window the cohort was transmitted in. NOT max(uu, pc5): see the note
        in compute_reward_components.
        """
        return float(self.logical_goodput_bps)

    @property
    def duplication_rate(self) -> float:
        if self.logical_tx <= 0:
            return 0.0
        return float(self.duplicated_logical_packets) / float(self.logical_tx)

    @property
    def attempted_copies_per_logical(self) -> float:
        if self.logical_tx <= 0:
            return float(RESOURCE_COST_BASELINE_COPIES)
        return float(self.attempted_copies) / float(self.logical_tx)


def read_tick_cohort(env) -> Optional[TickCohort]:
    """The matured cohort ns-3 published this step, or None."""
    if not int(env.cohort_ready):
        return None
    return TickCohort(
        macro_id=int(env.cohort_macro_id),
        macro_tick=int(env.cohort_macro_tick),
        action=int(env.cohort_action),
        logical_tx=int(env.cohort_logical_tx),
        deadline_success=int(env.cohort_deadline_success),
        deadline_miss=int(env.cohort_deadline_miss),
        delay_sum_ms=float(env.cohort_delay_sum_ms),
        delay_count=int(env.cohort_delay_count),
        attempted_copies=int(env.cohort_attempted_copies),
        duplicated_logical_packets=int(
            env.cohort_duplicated_logical_packets),
        uu_throughput_bps=float(env.cohort_uu_throughput_bps),
        pc5_throughput_bps=float(env.cohort_pc5_throughput_bps),
        logical_goodput_bps=float(env.cohort_logical_goodput_bps),
        time_since_last_rx_ms=float(env.cohort_time_since_last_rx_ms),
        backlog=int(env.cohort_backlog),
    )


def cohort_upper_objective(
    cohort: TickCohort,
    services: Sequence,
    mode: str = SERVICE_AGGREGATION_MODE,
) -> Optional[Dict[str, Any]]:
    """U_qos for one matured tick cohort, evaluated service by service.

    This is the ONLY quantity that enters the macro return. Returns None when
    the cohort carries no matured packet or no service is active, in which
    case the protocol says the tick contributes 0 to R_macro.
    """
    if not cohort.valid:
        return None
    breakdown = service_violation_breakdown(
        cohort.mean_delay_ms,
        cohort.pdr,
        cohort.useful_throughput_bps,
        max(0.0, float(cohort.time_since_last_rx_ms)),
        services,
        mode,
    )
    if breakdown is None:
        return None
    breakdown["macro_id"] = int(cohort.macro_id)
    breakdown["macro_tick"] = int(cohort.macro_tick)
    breakdown["cohort_logical_tx"] = int(cohort.logical_tx)
    breakdown["cohort_matured"] = int(cohort.matured)
    breakdown["cohort_pdr"] = float(cohort.pdr)
    breakdown["cohort_mean_delay_ms"] = cohort.mean_delay_ms
    breakdown["cohort_duplication_rate"] = float(cohort.duplication_rate)
    return breakdown


# ============================================================================
# Duplication constraint (structure only; the numbers are protocol decisions)
# ============================================================================
class DuplicationConstraint:
    """Sliding-window duplication budget with a learned Lagrange multiplier.

        constraint_violation = max(0, dup_rate_window - B_dup)
        J                    = U_qos - lambda * constraint_violation
        lambda              <- max(0, lambda + eta * (dup_rate_window - B_dup))

    B_dup, the window length and eta are protocol decisions and are NOT set
    here. While any of them is missing the constraint is INERT: the metric is
    still measured and logged, lambda stays at its initial value, and
    J == U_qos. Full training refuses to start in that state.

    This is deliberately a different mechanism from the goal's w_cost. w_cost
    expresses how much the lower controller should care about resource
    efficiency inside the CURRENT goal; the constraint bounds long-run Both
    usage at the system level. Neither replaces the other.
    """

    def __init__(self, budget=None, window=None, eta=None,
                 lambda_init: float = DUPLICATION_LAMBDA_INIT):
        self.budget = None if budget is None else float(budget)
        self.window = None if window is None else int(window)
        self.eta = None if eta is None else float(eta)
        self.lambda_ = float(lambda_init)
        self._samples: deque = deque(
            maxlen=self.window if self.window else 10_000)
        self.dual_updates = 0

    @property
    def enabled(self) -> bool:
        return (self.budget is not None and self.window is not None
                and self.eta is not None)

    def reset(self) -> None:
        self._samples.clear()

    def observe(self, duplication_rate: float) -> None:
        value = float(duplication_rate)
        if np.isfinite(value):
            self._samples.append(float(np.clip(value, 0.0, 1.0)))

    @property
    def window_full(self) -> bool:
        """Whether the sliding window holds its full complement of samples."""
        return self.window is not None and len(self._samples) >= self.window

    def window_rate(self) -> Optional[float]:
        """Mean duplication rate over a FULL window, else None.

        Two distinct reasons to return None, both deliberate:
          * the window length is unset -- an unbounded average is not a
            sliding window, and reporting it as one would let an unresolved
            protocol number look resolved;
          * the window is not yet full -- a partial sample mean is not a duty
            cycle. Reading it as one made a single early Both decision score
            as 100% duplication, so the constraint fired hardest at the start
            of every episode. While the window is filling the constraint is
            INACTIVE (see DUPLICATION_PARTIAL_WINDOW_POLICY); the cost is that
            the first `window` ticks of each episode are unconstrained, and
            that is a stated protocol limitation.
        """
        if not self.window_full:
            return None
        return float(np.mean(self._samples))

    def violation(self) -> float:
        """Raw violation v = max(0, dup_rate_window - B_dup)."""
        if not self.enabled:
            return 0.0
        rate = self.window_rate()
        if rate is None:
            return 0.0
        return float(max(0.0, rate - self.budget))

    def violation_norm(self) -> float:
        """v_norm = v / (1 - B_dup), in [0, 1].

        Used by BOTH the dual step and the objective penalty. Normalising only
        one of them would make lambda's scale depend on B_dup, and the dual
        response horizon T_lambda would stop meaning the same thing across
        budgets. With both normalised, lambda = 1 is "penalty equal to the
        whole objective range" for every B_dup.
        """
        if not self.enabled:
            return 0.0
        span = max(1.0 - float(self.budget), DUPLICATION_NORM_EPS)
        return float(min(1.0, self.violation() / span))

    def penalty(self) -> float:
        return float(self.lambda_ * self.violation_norm()) if self.enabled else 0.0

    def constrained_objective(self, u_qos: float) -> float:
        return float(u_qos) - self.penalty()

    def dual_update(self) -> Optional[float]:
        """One dual step. Called once per macro transition push, never per tick.

        The dual variable lives on the same event stream as the goal critic it
        constrains. Stepping every 100 ms would re-apply an almost identical
        gradient once per tick over a window that changed by a single sample,
        which only inflates the effective step size.
        """
        if not self.enabled:
            return None
        if self.window_rate() is None:
            return None
        self.lambda_ = float(max(0.0, self.lambda_ + self.eta * self.violation_norm()))
        self.dual_updates += 1
        return self.lambda_

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "constraint_version": RESOURCE_CONSTRAINT_VERSION,
            "constraint_metric": RESOURCE_CONSTRAINT_METRIC,
            "enabled": self.enabled,
            "budget": self.budget,
            "window": self.window,
            "eta": self.eta,
            "lambda": float(self.lambda_),
            "window_rate": self.window_rate(),
            "window_full": self.window_full,
            "violation": self.violation(),
            "violation_norm": self.violation_norm(),
            "penalty": self.penalty(),
            "partial_window_policy": DUPLICATION_PARTIAL_WINDOW_POLICY,
            "violation_normalized": DUPLICATION_VIOLATION_NORMALIZED,
            "dual_cadence": DUPLICATION_DUAL_CADENCE,
            "samples": len(self._samples),
            "dual_updates": self.dual_updates,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {"budget": self.budget, "window": self.window, "eta": self.eta,
                "lambda": float(self.lambda_),
                "dual_updates": int(self.dual_updates)}

    def load_state_dict(self, payload: Dict[str, Any]) -> None:
        self.lambda_ = float(payload.get("lambda", self.lambda_))
        self.dual_updates = int(payload.get("dual_updates", 0))


def duplication_rate(env) -> float:
    """Fraction of logical packets that received a SECOND copy this interval.

    Counted from what the accumulator actually did, not from the action label:
    under a partial duplication degree the two differ. This is the constraint
    metric. It is measured, reported and handed to the constraint evaluator; it
    is never folded into the objective with a weight, and the lambda penalty is
    never added to the lower reward.

    Algebraically this equals attempted_copies_per_logical - 1 whenever every
    packet costs one or two copies, which is always. The two quantities are the
    same physical measurement; what differs is their ROLE. r_cost feeds the
    100 ms lower reward through the planner's w_cost, expressing how much
    resource efficiency should matter inside the current goal. This feeds the
    system-level sliding-window dual, expressing a long-run feasibility bound.
    Neither substitutes for the other.
    """
    logical = float(int(env.window_logical_tx))
    if logical <= 0.0:
        return 0.0
    return float(int(env.window_duplicated_logical_packets)) / logical


def compute_reward(
    env,
    applied_action: int,
    goal_weights: Sequence[float],
    active_services: Sequence = (),
    actual_switch: bool = False,
    reverse_switch: bool = False,
    aggregation: str = SERVICE_AGGREGATION_MODE,
) -> Optional[RewardResult]:
    """The LOWER (100 ms) reward, plus a step-level diagnostic U_qos.

    Two quantities, on two timescales, and they must not be confused:

      r_lower   the planner's preference mixture over the four normalized KPI
                components, computed from the freshest interval measurement.
                This is the dense control signal the lower DDQN learns from;
                ordinary TD bootstrapping handles its measurement lag.

      u_qos_step  1 - max_service(max_kpi(violation)) on the SAME interval
                measurement. DIAGNOSTIC ONLY. The macro return never reads it:
                the upper objective is computed from matured tick cohorts
                (cohort_upper_objective), so a deadline outcome is credited to
                the goal that caused it rather than the goal in force when it
                ripened.
    """
    components = compute_reward_components(env, applied_action)
    if components is None:
        return None

    # Same authoritative quantity: a service's min_throughput requirement is a
    # requirement on delivered useful application data, not on a path's byte
    # rate, so it must be evaluated against the same measurement.
    throughput = float(env.logical_goodput_bps)
    breakdown = service_violation_breakdown(
        float(env.e2e_delay_ms),
        1.0 - float(env.deadline_loss_rate),
        throughput,
        max(0.0, float(env.time_since_last_rx_ms)),
        active_services,
        aggregation,
    )
    if breakdown is None:
        return None

    weights = np.asarray(goal_weights, dtype=np.float64)
    validate_goal(weights)
    ordered = [components[name] for name in
               ("r_rel", "r_lat", "r_thr", "r_cost")]
    lower_total = float(np.dot(weights, np.asarray(ordered, dtype=np.float64)))

    return RewardResult(
        lower_total=lower_total,
        upper_total=float(breakdown["u_qos"]),
        components=components,
        goal_weights={name: float(value)
                      for name, value in zip(GOAL_WEIGHT_NAMES, weights)},
        raw_kpi={
            "delay_ms": float(env.e2e_delay_ms),
            "deadline_loss_rate": float(env.deadline_loss_rate),
            "uu_throughput_bps": float(env.uu_throughput_bps),
            "pc5_throughput_bps": float(env.pc5_throughput_bps),
            "outage_ms": max(0.0, float(env.time_since_last_rx_ms)),
            "attempted_copies_per_logical": attempted_copies_per_logical(env),
            "successful_copies_per_logical": successful_copies_per_logical(env),
            "actual_switch": float(bool(actual_switch)),
            "reverse_switch": float(bool(reverse_switch)),
        },
        violations=dict(breakdown["binding_violations"]),
        service_violations={
            name: dict(values)
            for name, values in breakdown["per_service"].items()
        },
        binding_service=str(breakdown["binding_service"]),
        binding_services=list(breakdown["binding_services"]),
        binding_kpi=str(breakdown["binding_kpi"]),
        v_system=float(breakdown["V_system"]),
        duplication_rate=duplication_rate(env),
        lower_version=LOWER_REWARD_VERSION,
        upper_version=UPPER_OBJECTIVE_VERSION,
    )


# ============================================================================
# Diagnostics
# ============================================================================
def goal_statistics(goal: np.ndarray) -> Dict[str, float]:
    """Per-weight mean/std plus entropy, for collapse detection.

    A planner stuck at [.25,.25,.25,.25] has maximum entropy and zero std; a
    planner stuck at a one-hot has zero entropy. Both mean it is not planning.
    """
    values = np.atleast_2d(np.asarray(goal, dtype=np.float64))
    stats: Dict[str, float] = {}
    for index, name in enumerate(GOAL_WEIGHT_NAMES):
        stats[f"{name}_mean"] = float(values[:, index].mean())
        stats[f"{name}_std"] = float(values[:, index].std())
    safe = np.clip(values, 1e-12, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=-1)
    stats["goal_entropy"] = float(entropy.mean())
    stats["min_goal_entropy"] = float(entropy.min())
    stats["max_goal_entropy"] = float(entropy.max())
    stats["goal_entropy_ceiling"] = float(math.log(GOAL_DIM))
    stats["goal_weight_spread"] = float(
        (values.max(axis=-1) - values.min(axis=-1)).mean())
    return stats


class MetricTracker:
    """Episode-level accumulators (diagnostics only)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.action_counts = {index: 0 for index in range(ACTION_DIM)}
        self.switch_count = 0
        self.pingpong_count = 0
        self.goal_rows: List[np.ndarray] = []
        self.component_rows: List[Dict[str, float]] = []
        self.lower_rewards: List[float] = []
        self.train_times_ms: List[float] = []
        self.td_losses: List[float] = []
        self.planner_grad_norms: List[float] = []
        self.critic_grad_norms: List[float] = []
        self.lower_grad_norms: List[float] = []
        self.violation_rows: List[Dict[str, float]] = []
        self.upper_rewards: List[float] = []
        self.clipped_updates = 0
        self.update_count = 0
        self.last_macro: Optional[Dict[str, float]] = None
        # ---- macro / cohort accounting -----------------------------------
        self.cohort_u_qos: List[float] = []
        self.cohort_service_rows: List[Dict[str, Dict[str, float]]] = []
        self.binding_service_counts: Dict[str, int] = {}
        self.binding_kpi_counts: Dict[str, int] = {}
        self.binding_tie_count = 0
        self.macro_returns: List[float] = []
        self.macro_tick_counts: List[int] = []
        self.macro_invalid_ticks = 0
        self.macro_valid_ticks = 0
        self.cohorts_seen = 0
        self.cohorts_dropped_no_macro = 0
        self.macros_truncated = 0
        self.macros_discarded = 0
        # ---- goal exploration / critic support ---------------------------
        self.goal_behaviour_rows: List[np.ndarray] = []
        self.goal_planner_rows: List[np.ndarray] = []
        self.goal_distances: List[float] = []
        # Two separate series -- see ConnectivityAgent.goal_support_distance.
        self.goal_support_distances_macro: List[float] = []
        self.goal_support_distances_update: List[float] = []
        self.q_goal_behaviour: List[float] = []
        self.q_goal_planner: List[float] = []
        self.q_goal_ranges: List[float] = []

    def note_action(self, action: int, switched: bool, reversed_: bool) -> None:
        self.action_counts[int(action)] = self.action_counts.get(int(action), 0) + 1
        if switched:
            self.switch_count += 1
        if reversed_:
            self.pingpong_count += 1

    def note_goal(self, goal: np.ndarray,
                  planner_goal: Optional[np.ndarray] = None) -> None:
        behaviour = np.asarray(goal, dtype=np.float64).reshape(-1)
        self.goal_rows.append(behaviour)
        self.goal_behaviour_rows.append(behaviour)
        if planner_goal is not None:
            wanted = np.asarray(planner_goal, dtype=np.float64).reshape(-1)
            self.goal_planner_rows.append(wanted)
            self.goal_distances.append(
                float(np.linalg.norm(behaviour - wanted)))

    def note_goal_support(self, support_distance: Optional[float],
                          q_behaviour: Optional[float],
                          q_planner: Optional[float]) -> None:
        if support_distance is not None:
            self.goal_support_distances_macro.append(float(support_distance))
        if q_behaviour is not None:
            self.q_goal_behaviour.append(float(q_behaviour))
        if q_planner is not None:
            self.q_goal_planner.append(float(q_planner))

    def note_cohort(self, breakdown: Optional[Dict[str, object]]) -> None:
        """One matured tick cohort, with full service attribution."""
        self.cohorts_seen += 1
        if breakdown is None:
            self.macro_invalid_ticks += 1
            return
        self.macro_valid_ticks += 1
        self.cohort_u_qos.append(float(breakdown["u_qos"]))
        self.cohort_service_rows.append(
            {name: dict(values)
             for name, values in breakdown["per_service"].items()})
        binding = str(breakdown["binding_service"])
        kpi = str(breakdown["binding_kpi"])
        if breakdown.get("binding_is_tie"):
            self.binding_tie_count += 1
        self.binding_service_counts[binding] = (
            self.binding_service_counts.get(binding, 0) + 1)
        self.binding_kpi_counts[kpi] = self.binding_kpi_counts.get(kpi, 0) + 1

    def note_macro_push(self, macro_return: float, ticks: int,
                        truncated: bool = False) -> None:
        self.macro_returns.append(float(macro_return))
        self.macro_tick_counts.append(int(ticks))
        if truncated:
            self.macros_truncated += 1

    def note_reward(self, result: RewardResult) -> None:
        self.component_rows.append(dict(result.components))
        self.violation_rows.append(dict(result.violations))
        self.lower_rewards.append(result.lower_total)
        self.upper_rewards.append(result.upper_total)

    def note_macro(self, stats: Dict[str, float]) -> None:
        """One macro update: critic always, planner on its slower clock."""
        self.last_macro = dict(stats)
        self.critic_grad_norms.append(float(stats["critic_grad_norm"]))
        if stats.get("q_goal_range") is not None:
            self.q_goal_ranges.append(float(stats["q_goal_range"]))
        if stats.get("planner_loss") is not None:
            self.planner_grad_norms.append(float(stats["planner_grad_norm"]))
            if stats.get("goal_support_distance_update") is not None:
                self.goal_support_distances_update.append(
                    float(stats["goal_support_distance_update"]))

    def note_update(self, td_loss: float, lower_grad: float, clipped: bool,
                    train_time_ms: float) -> None:
        """One micro (lower DDQN) update."""
        self.update_count += 1
        self.td_losses.append(float(td_loss))
        self.lower_grad_norms.append(float(lower_grad))
        self.train_times_ms.append(float(train_time_ms))
        if clipped:
            self.clipped_updates += 1

    def summary(self) -> Dict[str, float]:
        def mean(values: Sequence[float]) -> Optional[float]:
            return float(np.mean(values)) if len(values) else None

        out: Dict[str, float] = {
            "switch_count": self.switch_count,
            "pingpong_count": self.pingpong_count,
            "mean_lower_reward": mean(self.lower_rewards),
            "mean_upper_reward": mean(self.upper_rewards),
            "mean_td_loss": mean(self.td_losses),
            "mean_planner_grad_norm": mean(self.planner_grad_norms),
            "mean_critic_grad_norm": mean(self.critic_grad_norms),
            "mean_lower_grad_norm": mean(self.lower_grad_norms),
            "mean_train_time_ms": mean(self.train_times_ms),
            "clipped_update_fraction": (
                self.clipped_updates / self.update_count
                if self.update_count else None),
        }
        total_actions = 0
        ratio_sum = 0.0
        for index in range(ACTION_DIM):
            count = self.action_counts.get(index, 0)
            out[f"action_{ACTION_NAMES[index].lower()}_count"] = count
            total_actions += count
            ratio_sum += count * DUPLICATION_LEVELS[index]
        # Mean COMMANDED duplication degree over the episode. A summary of what
        # the controller asked for; the realised rate is measured separately.
        out["mean_applied_duplication_ratio"] = (
            ratio_sum / total_actions if total_actions else None)
        for name in ("r_rel", "r_lat", "r_thr", "r_cost"):
            values = [row[name] for row in self.component_rows if name in row]
            out[f"mean_{name}"] = mean(values)
        for name in ("delay", "pdr", "throughput", "outage"):
            values = [row[name] for row in self.violation_rows if name in row]
            out[f"mean_viol_{name}"] = mean(values)
        if self.goal_rows:
            out.update(goal_statistics(np.stack(self.goal_rows)))

        # ---- cohort / macro accounting -----------------------------------
        out["cohorts_seen"] = self.cohorts_seen
        out["cohorts_dropped_no_macro"] = self.cohorts_dropped_no_macro
        out["macro_valid_ticks"] = self.macro_valid_ticks
        out["macro_invalid_ticks"] = self.macro_invalid_ticks
        out["macros_pushed"] = len(self.macro_returns)
        out["macros_truncated"] = self.macros_truncated
        out["macros_discarded"] = self.macros_discarded
        out["mean_macro_return"] = mean(self.macro_returns)
        out["mean_macro_ticks"] = mean(self.macro_tick_counts)
        out["mean_cohort_u_qos"] = mean(self.cohort_u_qos)
        # Which service, and which of its requirements, bound the objective.
        # "TIE" is its own bucket, so the counts sum to the cohort count and
        # the tie rate is directly visible. Per-row binding_services lists are
        # logged, so any other attribution is reproducible offline.
        out["binding_service_counts"] = dict(self.binding_service_counts)
        out["binding_tie_count"] = self.binding_tie_count
        out["binding_kpi_counts"] = dict(self.binding_kpi_counts)
        if self.binding_service_counts:
            out["dominant_binding_service"] = max(
                self.binding_service_counts,
                key=self.binding_service_counts.get)
        else:
            out["dominant_binding_service"] = None
        if self.binding_kpi_counts:
            out["dominant_binding_kpi"] = max(
                self.binding_kpi_counts, key=self.binding_kpi_counts.get)
        else:
            out["dominant_binding_kpi"] = None
        per_service_mean: Dict[str, Dict[str, Optional[float]]] = {}
        names = sorted({name for row in self.cohort_service_rows for name in row})
        for name in names:
            entries = [row[name] for row in self.cohort_service_rows
                       if name in row]
            per_service_mean[name] = {
                key: mean([entry[key] for entry in entries])
                for key in ("delay", "pdr", "throughput", "outage", "V_service")
            }
        out["mean_service_violations"] = per_service_mean

        # ---- goal exploration / critic support ---------------------------
        out["mean_goal_distance"] = mean(self.goal_distances)
        out["max_goal_distance"] = (
            float(np.max(self.goal_distances)) if self.goal_distances else None)
        def median(values):
            return float(np.median(values)) if len(values) else None

        # MACRO series -- the only one the formal sigma screen reads.
        out["mean_goal_support_distance_macro"] = mean(
            self.goal_support_distances_macro)
        out["median_goal_support_distance_macro"] = median(
            self.goal_support_distances_macro)
        out["max_goal_support_distance_macro"] = (
            float(np.max(self.goal_support_distances_macro))
            if self.goal_support_distances_macro else None)
        # UPDATE series -- diagnostic only.
        out["mean_goal_support_distance_update"] = mean(
            self.goal_support_distances_update)
        out["median_goal_support_distance_update"] = median(
            self.goal_support_distances_update)
        # C4 (rho) is recorded and never used for selection.
        out["median_goal_distance"] = median(self.goal_distances)
        if self.goal_planner_rows:
            spread = np.std(np.stack(self.goal_planner_rows), axis=0).mean()
            out["planner_goal_context_spread"] = float(spread)
            out["goal_identifiability_rho"] = (
                None if spread <= 0.0 or not self.goal_distances
                else float(median(self.goal_distances) / spread))
        else:
            out["planner_goal_context_spread"] = None
            out["goal_identifiability_rho"] = None
        out["mean_q_goal_behaviour"] = mean(self.q_goal_behaviour)
        out["mean_q_goal_planner"] = mean(self.q_goal_planner)
        out["mean_q_goal_range"] = mean(self.q_goal_ranges)
        return out


# ============================================================================
# Agent
# ============================================================================
@dataclass
class Transition:
    """One micro decision.

    The goal in force is replayed verbatim: the lower controller's job is to
    satisfy whatever goal it was handed, so re-deriving the goal from the
    current planner would change the very quantity the transition is about.
    The goal never carries gradient into the lower update.
    """

    radio: np.ndarray
    goal_at_action: np.ndarray
    action: int
    components: Dict[str, float]
    lower_reward: float
    upper_reward: float          # service-violation system objective
    next_radio: np.ndarray
    done: bool


@dataclass
class MacroTransition:
    """One goal interval.

    No lower action is stored: the upper critic is Q_goal(context, goal) and
    the lower controller's response lives inside macro_return.
    """

    context: np.ndarray
    goal: np.ndarray
    macro_return: float          # sum_{j<H} gamma^j r_upper(t_k + j)
    micro_steps: int             # how many micro rewards actually accrued
    next_context: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._items: deque = deque(maxlen=self.capacity)

    def push(self, item: Transition) -> None:
        self._items.append(item)

    def sample(self, batch_size: int, rng) -> List[Transition]:
        indices = rng.sample(range(len(self._items)), batch_size)
        return [self._items[i] for i in indices]

    def items(self) -> List[Any]:
        """Read-only view, so callers need not touch the private deque."""
        return list(self._items)

    def state_dict(self) -> Dict[str, Any]:
        """Everything needed to reconstruct this buffer exactly.

        Order matters: the deque is FIFO with maxlen, so both the contents and
        their sequence determine which transitions a later index draws. A set
        or a shuffled list would round-trip the contents and still change what
        sample() returns for the same RNG state.
        """
        return {"capacity": self.capacity, "items": list(self._items)}

    def load_state_dict(self, payload: Dict[str, Any]) -> None:
        capacity = int(payload["capacity"])
        if capacity != self.capacity:
            raise ValueError(
                f"replay capacity mismatch: checkpoint {capacity}, "
                f"agent {self.capacity}. Resuming into a different capacity "
                "would silently change the sampling distribution.")
        self._items = deque(payload["items"], maxlen=self.capacity)

    def fingerprint(self) -> str:
        """Order-sensitive digest, for round-trip regressions."""
        import hashlib
        digest = hashlib.sha256()
        digest.update(str(self.capacity).encode())
        digest.update(str(len(self._items)).encode())
        for item in self._items:
            for value in dataclasses.astuple(item):
                if isinstance(value, np.ndarray):
                    digest.update(np.ascontiguousarray(value).tobytes())
                    digest.update(str(value.dtype).encode())
                    digest.update(str(value.shape).encode())
                elif isinstance(value, dict):
                    for key in sorted(value):
                        digest.update(key.encode())
                        digest.update(repr(float(value[key])).encode())
                else:
                    digest.update(repr(value).encode())
        return digest.hexdigest()

    def __len__(self) -> int:
        return len(self._items)


def _as_rng_state(state):
    """random.Random.setstate needs tuples; a torch.save round-trip yields
    lists and setstate rejects those with a bare TypeError."""
    if isinstance(state, (list, tuple)):
        return tuple(_as_rng_state(x) for x in state)
    return state


def _seeded_init(module: nn.Module, seed: int) -> None:
    """Re-initialise a module from its own dedicated RNG stream.

    Without this the lower DDQN's initial weights depend on how many random
    numbers the planner constructor happened to consume, so FIXED / RULE / MLP
    / TRANSFORMER would start their lower controllers from different points
    under the same train seed. That is a confound in a comparison whose whole
    premise is "only the Goal Planner differs".
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            # torch's default Linear init, reproduced against our generator
            fan_in = layer.weight.shape[1]
            bound = 1.0 / math.sqrt(fan_in)
            with torch.no_grad():
                layer.weight.uniform_(-bound, bound, generator=generator)
                if layer.bias is not None:
                    layer.bias.uniform_(-bound, bound, generator=generator)


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().norm(2).item()) ** 2
    return float(math.sqrt(total))


class ConnectivityAgent:
    """Macro/micro hierarchical agent.

    MACRO tier (goal interval H = 5 micro steps = 500 ms)
        pi_goal(context) -> g_k, held for the whole interval
        Q_goal(context, goal)                       <- NO lower action input
        L_critic  = MSE( Q_goal(c_k, g_k), y_goal )
        y_goal    = R_macro_k + gamma^H * Q_goal_target(c_k+1, pi_goal_target(c_k+1))
        R_macro_k = sum_{j<H} gamma^j * r_upper(t_k + j)
        L_planner = -mean Q_goal(c_k, pi_goal(c_k))

        R_macro is discounted within the interval from gamma^0, so the gamma^H
        bootstrap continues the same series. No double discounting.

    MICRO tier (action interval 100 ms)
        Q_lower(radio, g_k.detach()) -> Q over {Uu, PC5, Both}
        L_lower = MSE( Q_lower(s_t, g_k, a_t), r_lower + gamma * Q_lower_target )

    Three independent optimizers with independent gradient clipping, so the
    Goal Planner's parameter count can never change the effective step size of
    the lower controller or the critic. Two-timescale: the planner updates less
    often than the critic.
    """

    def __init__(
        self,
        planner_kind: str = DEFAULT_GOAL_PLANNER,
        capacity: int = 20_000,
        macro_capacity: int = 4_000,
        batch_size: int = 64,
        macro_batch_size: int = 32,
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_update: int = 200,
        grad_clip: float = E2E_GRAD_CLIP,
        history_len: int = HISTORY_LEN,
        macro_horizon: int = MACRO_HORIZON,
        planner_update_k: Optional[int] = None,
        critic_warmup_w: Optional[int] = None,
        beta: Optional[float] = None,
        p_global: Optional[float] = None,
        duplication_budget: Optional[float] = None,
        duplication_window: Optional[int] = None,
        duplication_dual_eta: Optional[float] = None,
        init_seed: int = 0,
        device: str = "cpu",
        **planner_kwargs,
    ):
        self.planner_kind = planner_kind
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.macro_batch_size = int(macro_batch_size)
        self.gamma = float(gamma)
        self.macro_horizon = int(macro_horizon)
        self.target_update = int(target_update)
        self.grad_clip = float(grad_clip)
        self.init_seed = int(init_seed)

        # ---- two-timescale upper tier -------------------------------------
        # No defaults. K and W are protocol decisions; a silent default here
        # would let an unfrozen number become an experimental result.
        if planner_update_k is None or critic_warmup_w is None:
            raise ValueError(
                "planner_update_k (K) and critic_warmup_w (W) must be given "
                "explicitly -- they are unresolved protocol parameters, not "
                "defaults. In a smoke test pass TEST-ONLY values; a full "
                "training run must have them frozen.")
        if int(planner_update_k) < 1 or int(critic_warmup_w) < 0:
            raise ValueError("K must be >= 1 and W must be >= 0")
        self.planner_update_k = int(planner_update_k)
        self.critic_warmup_w = int(critic_warmup_w)
        # Backwards-compatible alias used by existing diagnostics.
        self.planner_update_every = self.planner_update_k

        # ---- goal exploration: simplex-space Dirichlet convex mixing ------
        #     g_beh = (1 - beta) * g_det + beta * u,   u ~ Dir(1,1,1,1)
        # The retired logit-space mechanism perturbed the pre-softmax scores,
        # so its goal-space radius carried the softmax Jacobian and vanished
        # as the deterministic policy saturated. Mixing acts directly on the
        # simplex, so the radius is exactly beta * ||u - g_det|| and cannot
        # collapse.
        self.beta = None if beta is None else float(beta)
        if self.beta is not None and not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        # Dedicated RNG stream, disjoint from torch and from the python RNG,
        # so changing beta or the planner cannot shift the sequence any other
        # component draws from.
        self._mixing_rng = np.random.default_rng(int(init_seed) + 104729)
        self.mixing_draws = 0
        # The runner owns ONE random.Random that drives BOTH replay sampling
        # and epsilon-greedy action choice. Exact continuation needs its
        # state, so the runner hands it over here. Nothing about how it is
        # used changes -- it is still passed into train_lower / train_macro /
        # select_action exactly as before.
        self._action_rng = None
        # U8: probability that a behaviour goal is drawn WITHOUT reference to
        # the planner. Separate from beta, which stays the LOCAL budget.
        self.p_global = 0.0 if p_global is None else float(p_global)
        if not 0.0 <= self.p_global <= 1.0:
            raise ValueError("p_global must lie in [0, 1]")
        self.local_draws = 0
        self.global_draws = 0

        # ---- duplication constraint (structure only) -----------------------
        self.duplication = DuplicationConstraint(
            budget=duplication_budget, window=duplication_window,
            eta=duplication_dual_eta)

        # ---- networks -----------------------------------------------------
        self.planner = make_goal_planner(
            planner_kind, history_len=history_len, **planner_kwargs).to(self.device)
        self.planner_target = make_goal_planner(
            planner_kind, history_len=history_len, **planner_kwargs).to(self.device)
        self.planner_target.load_state_dict(self.planner.state_dict())
        self.planner_target.eval()

        self.lower = GoalConditionedDDQN().to(self.device)
        self.lower_target = GoalConditionedDDQN().to(self.device)
        self.goal_critic = GoalCritic(history_len=history_len).to(self.device)
        self.goal_critic_target = GoalCritic(history_len=history_len).to(self.device)

        # Dedicated init streams: identical across every planner variant.
        _seeded_init(self.lower, self.init_seed)
        _seeded_init(self.goal_critic, self.init_seed + 1)
        self.lower_target.load_state_dict(self.lower.state_dict())
        self.goal_critic_target.load_state_dict(self.goal_critic.state_dict())
        self.lower_target.eval()
        self.goal_critic_target.eval()

        # ---- independent optimizers, independent clipping -----------------
        # RETIRED INVARIANT: lr_planner == lr_critic, with the timescale
        # separation carried by K and W alone. The STEP 4B-3 audit measured
        # that this cannot hold under Adam, which sets the per-parameter step
        # to ~lr regardless of gradient magnitude: K divides the planner's step
        # COUNT and leaves its step SIZE untouched, so the planner still moved
        # 1.09-2.35x the critic in normalized terms and committed 31-45% of its
        # total movement while the critic was below 20% of steady state.
        #
        # lr_planner = lr_critic / K makes K carry the separation in both count
        # and size. It adds no new constant: K is already frozen and lr_critic
        # is unchanged.
        self.lr = float(lr)
        self.lr_planner = (self.lr / float(self.planner_update_k)
                           if (PLANNER_LR_SCALED_BY_K
                               and self.planner_update_k)
                           else self.lr)
        self.opt_lower = optim.Adam(self.lower.parameters(), lr=self.lr)
        self.opt_critic = optim.Adam(self.goal_critic.parameters(), lr=self.lr)
        planner_params = [p for p in self.planner.parameters() if p.requires_grad]
        self.opt_planner = (
            optim.Adam(planner_params, lr=self.lr_planner)
            if planner_params else None)

        self.replay = ReplayBuffer(capacity)               # micro
        self.macro_replay = ReplayBuffer(macro_capacity)   # macro
        self.updates = 0
        self.macro_updates = 0
        self.planner_updates = 0
        self.last_planner_grad = 0.0
        self.last_lower_grad = 0.0
        self.last_critic_grad = 0.0
        # Per-PLANNER-UPDATE support distance (batch mean). Diagnostic only.
        # The per-MACRO value the formal screen uses is computed by
        # goal_support_distance() at goal-emission time and never merged here.
        self.last_goal_support_update = None

        # ---- exploration schedule (filled by attach_bookkeeping) ----------
        # Kept on the agent from construction so a checkpoint always carries a
        # complete, self-describing schedule.
        self.epsilon = 0.0
        self.epsilon_start = float(EPSILON_START)
        self.epsilon_min = float(EPSILON_MIN)
        self.epsilon_decay = 1.0
        self.epsilon_decay_step = 0
        self.epsilon_decay_mode = EPSILON_DECAY_MODE
        self.epsilon_schedule_version = EPSILON_SCHEDULE_VERSION
        self.total_epsilon_decay_steps = None
        self.training_horizon_n = None
        self.sim_time_e = None
        self.n_formal = None

    @property
    def planner_is_trainable(self) -> bool:
        return self.opt_planner is not None

    def parameter_counts(self) -> Dict[str, int]:
        return {
            "planner": sum(p.numel() for p in self.planner.parameters()),
            "planner_trainable": sum(
                p.numel() for p in self.planner.parameters() if p.requires_grad),
            "lower": sum(p.numel() for p in self.lower.parameters()),
            "goal_critic": sum(p.numel() for p in self.goal_critic.parameters()),
        }

    # ---- macro tier: goal generation -------------------------------------
    @property
    def goal_exploration_active(self) -> bool:
        """Whether behaviour goals are mixed at all.

        beta = 0 (or unset) means the mechanism is OFF -- not that it behaves
        like the retired one. FIXED and RULE have no trainable planner, so
        exploration is structurally inert for them, exactly as before.
        """
        return (self.beta is not None and self.beta > 0.0
                and getattr(self.planner, "supports_exploration", False)
                and self.opt_planner is not None)

    def exploration_diagnostics(self) -> Dict[str, Any]:
        """Protocol identity of the exploration mechanism, for provenance."""
        low, high = ((None, None) if self.beta is None
                     else exploration_radius_bracket(self.beta))
        return {
            "exploration_protocol_version": EXPLORATION_PROTOCOL_VERSION,
            "goal_exploration_mechanism": GOAL_EXPLORATION_MECHANISM,
            "beta": self.beta,
            "dirichlet_concentration": DIRICHLET_CONCENTRATION,
            "mixing_draws": int(self.mixing_draws),
            "local_draws": int(self.local_draws),
            "global_draws": int(self.global_draws),
            "p_global": self.p_global,
            "global_draw_fraction": (float(self.global_draws) / self.mixing_draws
                                     if self.mixing_draws else None),
            "planner_parameterization": PLANNER_PARAMETERIZATION,
            "planner_logit_bound": PLANNER_LOGIT_BOUND,
            "radius_bracket_low": low,
            "radius_bracket_high": high,
            "active": self.goal_exploration_active,
        }

    def plan_goal(self, context: np.ndarray,
                  explore: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """One macro decision. Called once per goal interval, not per action.

        Returns (behaviour_goal, planner_goal). They differ only when
        exploration is on, and then through one of two simplex-valued modes
        (LOCAL beta-mixing around pi(c), or a planner-independent GLOBAL
        draw). Either way the behaviour goal is exactly a point on the simplex
        (w_i >= 0, sum w_i == 1) by construction, never a clipped or
        renormalised vector. Both are returned so the critic's
        support can be monitored: the planner is trained on the goal it WANTS,
        the critic is trained on the goal that was actually in force.
        """
        self.planner.eval()
        with torch.no_grad():
            ctx = torch.as_tensor(context, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            planner_goal = self.planner(ctx).squeeze(0).cpu().numpy()
            behaviour = planner_goal
            if explore and self.goal_exploration_active:
                # Two behaviour modes. Both produce an exact simplex point --
                # no projection, no clipping, no renormalisation.
                #
                #   LOCAL   (1 - p_global): the frozen beta-mixing. Dense
                #           support around pi(c), which is where the DPG
                #           planner gradient is evaluated.
                #   GLOBAL  (p_global): a draw that does NOT reference the
                #           planner at all. This is the U8 fix: beta-mixing
                #           alone anchors every behaviour goal to the planner,
                #           so once the planner collapsed the critic never saw
                #           w_thr / w_cost above ~0.3 -- exactly where the
                #           environment's context x goal interaction lives.
                #
                # A GLOBAL goal is BEHAVIOUR ONLY: not a manual reward rule,
                # not a planner target, not a supervision label, and never the
                # deployed policy. The planner still emits g_det = pi(c) and is
                # still trained only through the critic.
                reference = self._mixing_rng.dirichlet(
                    np.full(GOAL_DIM, DIRICHLET_CONCENTRATION))
                if (self.p_global > 0.0
                        and self._mixing_rng.random() < self.p_global):
                    behaviour = reference
                    self.global_draws += 1
                else:
                    behaviour = ((1.0 - self.beta)
                                 * planner_goal.astype(np.float64)
                                 + self.beta * reference)
                    self.local_draws += 1
                self.mixing_draws += 1
        validate_goal(planner_goal)
        validate_goal(behaviour)
        return (np.asarray(behaviour, dtype=np.float32),
                np.asarray(planner_goal, dtype=np.float32))

    def expected_squared_radius(self, planner_goal) -> Optional[float]:
        """Closed-form E||g_beh - g_det||^2 for the goal just emitted.

        Logged next to the realised radius so the analytic identity is
        checkable at run time rather than assumed.
        """
        if self.beta is None:
            return None
        return expected_squared_radius(self.beta, planner_goal)

    def replay_goals(self) -> Optional[np.ndarray]:
        """Behaviour goals currently held in the macro replay, or None."""
        items = list(self.macro_replay.items())
        if not items:
            return None
        return np.stack([np.asarray(t.goal, dtype=np.float64) for t in items])

    def goal_support_distance(self, goal: np.ndarray) -> Optional[float]:
        """goal_support_distance_MACRO: one goal against the replay support.

            min_j || goal - g_j ||_2      over replay behaviour goals g_j

        Evaluated once per macro on the LIVE context, for a SINGLE goal. This
        is the only support metric the formal sigma screen (C1/C1') reads. It
        must never be pooled with the per-planner-update batch mean: the two
        differ in query count (1 vs batch), context source (live vs replay)
        and aggregation (none vs mean).

        A deliberately simple metric -- no density model. It answers the only
        question that matters here: is the planner asking the critic to value a
        goal region the critic has never seen behave?
        """
        stored = self.replay_goals()
        if stored is None:
            return None
        target = np.asarray(goal, dtype=np.float64).reshape(1, -1)
        return float(np.min(np.linalg.norm(stored - target, axis=1)))

    def q_goal(self, context: np.ndarray, goal: np.ndarray) -> float:
        """Q_goal(context, goal) for diagnostics. No gradient, no side effect."""
        self.goal_critic.eval()
        with torch.no_grad():
            ctx = torch.as_tensor(context, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            g = torch.as_tensor(goal, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return float(self.goal_critic(ctx, g).squeeze(0).item())

    # ---- micro tier: action selection ------------------------------------
    def select_action(self, radio, goal, valid_actions, rng):
        """Epsilon-greedy over legal actions, conditioned on the held goal."""
        if not valid_actions:
            raise ValueError("valid_actions must be non-empty")
        if any(a not in range(ACTION_DIM) for a in valid_actions):
            raise ValueError(
                f"valid_actions must be a subset of 0..{ACTION_DIM - 1}")
        self.lower.eval()
        with torch.no_grad():
            rad = torch.as_tensor(radio, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
            g = torch.as_tensor(goal, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            q = self.lower(rad, g).squeeze(0).cpu().numpy()
        if rng.random() < float(self.epsilon):
            action = int(rng.choice(list(valid_actions)))
        else:
            masked = {a: float(q[a]) for a in valid_actions}
            action = int(max(masked, key=masked.get))
        return action, q

    # ---- learning --------------------------------------------------------
    def train_lower(self, rng) -> Optional[Dict[str, float]]:
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size, rng)
        rad = torch.as_tensor(np.stack([t.radio for t in batch]),
                              dtype=torch.float32, device=self.device)
        nrad = torch.as_tensor(np.stack([t.next_radio for t in batch]),
                               dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(np.stack([t.goal_at_action for t in batch]),
                               dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([[t.action] for t in batch],
                                  dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([t.lower_reward for t in batch],
                                  dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(t.done) for t in batch],
                                dtype=torch.float32, device=self.device)

        # The goal in force at the time is replayed verbatim and never carries
        # gradient: the lower controller learns to satisfy whatever goal it is
        # handed, it does not shape the goal.
        with torch.no_grad():
            next_actions = self.lower(nrad, goal).argmax(dim=1, keepdim=True)
            q_next = self.lower_target(nrad, goal).gather(
                1, next_actions).squeeze(1)
            target = rewards + self.gamma * (1.0 - dones) * q_next

        self.lower.train()
        q = self.lower(rad, goal).gather(1, actions).squeeze(1)
        loss = nn.functional.mse_loss(q, target)
        self.opt_lower.zero_grad(set_to_none=True)
        loss.backward()
        grad = _grad_norm(self.lower.parameters())
        nn.utils.clip_grad_norm_(self.lower.parameters(), self.grad_clip)
        self.opt_lower.step()

        self.updates += 1
        if self.updates % self.target_update == 0:
            self.lower_target.load_state_dict(self.lower.state_dict())
        self.last_lower_grad = grad
        return {"td_loss": float(loss.item()), "lower_grad_norm": grad,
                "lower_clipped": float(grad > self.grad_clip)}

    def train_macro(self, rng) -> Optional[Dict[str, float]]:
        """Upper critic every macro transition; planner on a slower clock."""
        if len(self.macro_replay) < self.macro_batch_size:
            return None
        batch = self.macro_replay.sample(self.macro_batch_size, rng)
        ctx = torch.as_tensor(np.stack([t.context for t in batch]),
                              dtype=torch.float32, device=self.device)
        nctx = torch.as_tensor(np.stack([t.next_context for t in batch]),
                               dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(np.stack([t.goal for t in batch]),
                               dtype=torch.float32, device=self.device)
        macro_r = torch.as_tensor([t.macro_return for t in batch],
                                  dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([float(t.done) for t in batch],
                                dtype=torch.float32, device=self.device)

        # y_goal = R_macro + gamma^H * Q_goal_target(c', g)
        #
        # GOAL_CRITIC_OPERATOR = goal_consistent_bootstrap_v1.
        # The bootstrap is evaluated at `goal` -- the EXACT goal stored on this
        # macro transition, the same tensor used on the prediction side below.
        # It is NOT re-sampled, NOT recomputed by self.planner, NOT recomputed
        # by self.planner_target, NOT the next macro's behaviour goal, and NOT
        # averaged with anything. planner_target plays no part in the critic
        # target under this operator; it remains in use for the goal-support
        # diagnostics and is still hard-synced on the target_update clock.
        #
        # WHY: under the retired planner_target_bootstrap_v1 the current goal
        # entered the target only through R_macro (bounded by 4.900995), while
        # the goal-INDEPENDENT bootstrap grew with the value scale. That left
        # the goal 0.89-1.13% of the target scale at ep161, and the critic
        # responded with a ~99% affine goal surface whose slope direction was
        # nearly context-invariant, which specifies a context-INDEPENDENT
        # optimal vertex; the planner collapsed to a constant by ep8
        # (PHASE D-3B). Evaluating the bootstrap at the same g propagates
        # goal-specific credit across the 1/(1-gamma^H) = 20.40-macro horizon
        # instead of confining it to a single macro.
        with torch.no_grad():
            bootstrap = self.goal_critic_target(nctx, goal)
            target = macro_r + (self.gamma ** self.macro_horizon) * (
                1.0 - dones) * bootstrap

        self.goal_critic.train()
        value = self.goal_critic(ctx, goal)
        loss_critic = nn.functional.mse_loss(value, target)
        self.opt_critic.zero_grad(set_to_none=True)
        loss_critic.backward()
        critic_grad = _grad_norm(self.goal_critic.parameters())
        nn.utils.clip_grad_norm_(self.goal_critic.parameters(), self.grad_clip)
        self.opt_critic.step()
        self.macro_updates += 1

        with torch.no_grad():
            q_batch = self.goal_critic(ctx, goal)
            q_min = float(q_batch.min().item())
            q_max = float(q_batch.max().item())
        stats = {
            "critic_loss": float(loss_critic.item()),
            "critic_grad_norm": critic_grad,
            "planner_loss": None,
            "planner_grad_norm": 0.0,
            "q_goal_behaviour_mean": float(q_batch.mean().item()),
            "q_goal_min": q_min,
            "q_goal_max": q_max,
            "q_goal_range": q_max - q_min,
            "critic_warmup_done": bool(self.macro_updates >= self.critic_warmup_w),
            "planner_updated": False,
        }

        # ---- planner: slower clock, own optimizer, own clip ---------------
        # Two-timescale, expressed only through W and K:
        #   W  critic updates complete before the planner's first step, so the
        #      planner never climbs an untrained critic;
        #   K  critic updates per planner update afterwards, so the critic
        #      stays ahead of the policy it is valuing.
        # Learning rates are identical by construction.
        planner_due = (
            self.macro_updates > self.critic_warmup_w
            and (self.macro_updates - self.critic_warmup_w)
            % self.planner_update_k == 0)
        if self.opt_planner is not None and planner_due:
            self.planner.train()
            for param in self.goal_critic.parameters():
                param.requires_grad_(False)
            loss_planner = -self.goal_critic(ctx, self.planner(ctx)).mean()
            self.opt_planner.zero_grad(set_to_none=True)
            loss_planner.backward()
            planner_grad = _grad_norm(self.planner.parameters())
            nn.utils.clip_grad_norm_(self.planner.parameters(), self.grad_clip)
            self.opt_planner.step()
            for param in self.goal_critic.parameters():
                param.requires_grad_(True)
            self.planner_updates += 1
            self.last_planner_grad = planner_grad
            stats["planner_loss"] = float(loss_planner.item())
            stats["planner_grad_norm"] = planner_grad
            stats["planner_updated"] = True
            # Goal-critic support: how far is the goal the planner now wants
            # from the goals the critic has actually seen behave?
            with torch.no_grad():
                wanted = self.planner(ctx)
                stats["q_goal_planner_mean"] = float(
                    self.goal_critic(ctx, wanted).mean().item())
                stored = torch.as_tensor(
                    self.replay_goals(), dtype=torch.float32,
                    device=self.device)
                distances = torch.cdist(wanted, stored)
                # batch mean of per-context min distances -- the UPDATE-cadence
                # series. Never pooled with the macro-cadence series.
                stats["goal_support_distance_update"] = float(
                    distances.min(dim=1).values.mean().item())
                stats["goal_behaviour_distance"] = float(
                    torch.norm(wanted - goal, dim=1).mean().item())
                safe = wanted.clamp_min(1e-12)
                stats["planner_goal_entropy"] = float(
                    (-(safe * safe.log()).sum(dim=-1)).mean().item())
            self.last_goal_support_update = stats["goal_support_distance_update"]

        if self.macro_updates % self.target_update == 0:
            self.goal_critic_target.load_state_dict(self.goal_critic.state_dict())
            self.planner_target.load_state_dict(self.planner.state_dict())
        self.last_critic_grad = critic_grad
        return stats

    # ---- decision bookkeeping / runner API -------------------------------
    def attach_bookkeeping(
        self,
        epsilon_start: float = EPSILON_START,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = 1.0,
        epsilon_decay_mode: str = EPSILON_DECAY_MODE,
        epsilon_schedule_version: str = EPSILON_SCHEDULE_VERSION,
        total_epsilon_decay_steps: Optional[int] = None,
        training_horizon_n: Optional[int] = None,
        sim_time_e: Optional[float] = None,
        n_formal: Optional[int] = None,
        pending_max_entries: int = PENDING_MAX_ENTRIES,
        pending_max_age_s: float = PENDING_MAX_AGE_S,
    ) -> None:
        """Install the exploration schedule and the decision bookkeeping pool.

        `epsilon_decay` is supplied by the caller, which derives it from the
        frozen training budget (tvt_config.derived_epsilon_decay). Nothing here
        invents a decay rate.
        """
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = float(epsilon_start)
        self.epsilon_decay_step = 0
        self.epsilon_decay_mode = str(epsilon_decay_mode)
        self.epsilon_schedule_version = str(epsilon_schedule_version)
        self.total_epsilon_decay_steps = (
            None if total_epsilon_decay_steps is None
            else int(total_epsilon_decay_steps))
        self.training_horizon_n = (
            None if training_horizon_n is None else int(training_horizon_n))
        self.sim_time_e = (
            None if sim_time_e is None else float(sim_time_e))
        self.n_formal = None if n_formal is None else int(n_formal)
        self._pending = PendingPool(pending_max_entries, pending_max_age_s)

    def attach_rng(self, rng) -> None:
        """Register the runner's sampling/action RNG so it can be checkpointed.

        Observation-only: the object is stored, never re-seeded, never consumed
        here, and never substituted for the rng the caller keeps passing in.
        """
        self._action_rng = rng

    def discard_pending(self) -> None:
        self._pending.reset()

    def pending_for(self, decision_id: int) -> Optional[Dict[str, Any]]:
        return self._pending.peek(decision_id)

    def remember_decision(self, decision_id, radio, goal, action,
                          source_action, sim_time_s, epsilon) -> None:
        self._pending.put(int(decision_id), {
            "radio": np.asarray(radio, dtype=np.float32).copy(),
            "goal": np.asarray(goal, dtype=np.float32).copy(),
            "action": int(action),
            "source_action": int(source_action),
            "sim_time_s": float(sim_time_s),
            "epsilon": float(epsilon),
        })

    def consume_outcome(self, decision_id, applied_action, reward_result,
                        next_radio, transition_valid, learn: bool = True) -> bool:
        """Close one micro decision. Returns True if a transition was stored."""
        entry = self._pending.consume(int(decision_id))
        if entry is None or not transition_valid or reward_result is None:
            return False
        # The stored action must be what the environment actually executed.
        if int(applied_action) != int(entry["action"]):
            entry["action"] = int(applied_action)
        if not learn:
            return False
        self.store(Transition(
            radio=entry["radio"],
            goal_at_action=entry["goal"],
            action=int(entry["action"]),
            components=dict(reward_result.components),
            lower_reward=float(reward_result.lower_total),
            upper_reward=float(reward_result.upper_total),
            next_radio=np.asarray(next_radio, dtype=np.float32).copy(),
            done=False,
        ))
        return True

    def store(self, transition) -> None:
        self.replay.push(transition)

    def store_macro(self, transition: MacroTransition) -> None:
        self.macro_replay.push(transition)

    def decay_epsilon(self) -> None:
        """Advance the exploration schedule by exactly one formal action.

        Evaluated in CLOSED FORM from the step counter rather than by repeated
        multiplication. Two consequences that matter:
          * epsilon is exactly a function of epsilon_decay_step, so a resumed
            run reproduces an uninterrupted run bit-for-bit instead of only
            approximately;
          * the identity epsilon == max(eps_min, eps_start * decay**step) is
            true by construction and is checkable as an invariant.

        Reaching epsilon_min is EXPLORATION SCHEDULE COMPLETION, not policy
        convergence.
        """
        self.epsilon_decay_step += 1
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start * (self.epsilon_decay ** self.epsilon_decay_step))

    def epsilon_schedule_state(self) -> Dict[str, Any]:
        """Everything needed to resume the schedule and to verify it."""
        return {
            "epsilon": float(self.epsilon),
            "epsilon_decay": float(self.epsilon_decay),
            "epsilon_decay_mode": self.epsilon_decay_mode,
            "epsilon_schedule_version": self.epsilon_schedule_version,
            "epsilon_start": float(self.epsilon_start),
            "epsilon_min": float(self.epsilon_min),
            "total_epsilon_decay_steps": self.total_epsilon_decay_steps,
            "epsilon_decay_step": int(self.epsilon_decay_step),
            "training_horizon_N": self.training_horizon_n,
            "sim_time_E": self.sim_time_e,
            "n_formal": self.n_formal,
        }

    def diagnostics(self) -> Dict[str, int]:
        return self._pending.diagnostics()

    # ---- checkpoint -------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        from tvt_config import (
            ACTION_REPRESENTATION,
            ACTION_SEMANTICS_VERSION,
            ARCHITECTURE_VERSION,
            DUPLICATION_SCHEDULER,
            E2E_CHECKPOINT_SCHEMA_VERSION,
            GOAL_SEMANTICS,
            LOWER_REWARD_VERSION,
            PRIMARY_PATH_POLICY,
            THROUGHPUT_PRIMITIVE_VERSION,
            UPPER_OBJECTIVE_VERSION,
            VEHICLE_CONTEXT_DIM as _VEH_DIM,
        )
        return {
            "architecture_version": ARCHITECTURE_VERSION,
            "checkpoint_schema_version": E2E_CHECKPOINT_SCHEMA_VERSION,
            # UPPER-tier observation contract. The upper tensors are shaped by
            # CONTEXT_FEATURE_DIM, so a contract change makes them unloadable
            # even though the schema key set is unchanged.
            "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
            "context_feature_dim": CONTEXT_FEATURE_DIM,
            "vehicle_context_dim": _VEH_DIM,
            "planner_kind": self.planner_kind,
            "action_dim": ACTION_DIM,
            "action_semantics": ACTION_SEMANTICS_VERSION,
            "action_representation": ACTION_REPRESENTATION,
            "duplication_levels": list(DUPLICATION_LEVELS),
            "duplication_scheduler": DUPLICATION_SCHEDULER,
            "primary_path_policy": PRIMARY_PATH_POLICY,
            "throughput_primitive_version": THROUGHPUT_PRIMITIVE_VERSION,
            "lower_reward_version": LOWER_REWARD_VERSION,
            "planner_parameterization": PLANNER_PARAMETERIZATION,
            "planner_logit_bound": PLANNER_LOGIT_BOUND,
            # SEMANTIC contract on what Q_goal(c, g) means. Tensor shapes are
            # identical across operators, so without this key an old-operator
            # critic could be resumed silently under the new one.
            "goal_critic_operator": GOAL_CRITIC_OPERATOR,
            "e2e_timescale_protocol": E2E_TIMESCALE_PROTOCOL,
            "planner_lr": self.lr_planner,
            "exploration_protocol_version": EXPLORATION_PROTOCOL_VERSION,
            "actions": list(ACTION_NAMES),
            "goal_dim": GOAL_DIM,
            "goal_components": ["reliability", "latency", "throughput",
                                "resource"],
            "goal_semantics": dict(GOAL_SEMANTICS),
            "mpquic_enabled": False,
            "action_hold_enabled": ACTION_HOLD_ENABLED,
            "macro_horizon": self.macro_horizon,
            "macro_interval_basis": MACRO_INTERVAL_BASIS,
            "macro_return_convention": MACRO_RETURN_CONVENTION,
            "macro_terminal_convention": MACRO_TERMINAL_CONVENTION,
            "reliability_attribution_version": RELIABILITY_ATTRIBUTION_VERSION,
            "goal_exploration_version": GOAL_EXPLORATION_VERSION,
            "exploration_protocol_version": EXPLORATION_PROTOCOL_VERSION,
            "goal_exploration_mechanism": GOAL_EXPLORATION_MECHANISM,
            "beta": self.beta,
            "mixing_draws": int(self.mixing_draws),
            "two_timescale_version": TWO_TIMESCALE_VERSION,
            "planner_update_k": self.planner_update_k,
            "critic_warmup_w": self.critic_warmup_w,
            "duplication_constraint": self.duplication.state_dict(),
            "lr": self.lr,
            "lower_reward_version": LOWER_REWARD_VERSION,
            "upper_objective_version": UPPER_OBJECTIVE_VERSION,
            "resource_cost_version": RESOURCE_COST_VERSION,
            "context_feature_dim": CONTEXT_FEATURE_DIM,
            "history_len": self.planner.history_len,
            "radio_state_dim": RADIO_STATE_DIM,
            "lower_input_dim": LOWER_INPUT_DIM,
            "parameter_counts": self.parameter_counts(),
            "planner": self.planner.state_dict(),
            "planner_target": self.planner_target.state_dict(),
            "lower": self.lower.state_dict(),
            "lower_target": self.lower_target.state_dict(),
            "goal_critic": self.goal_critic.state_dict(),
            "goal_critic_target": self.goal_critic_target.state_dict(),
            "opt_lower": self.opt_lower.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "opt_planner": (None if self.opt_planner is None
                            else self.opt_planner.state_dict()),
            "updates": self.updates,
            "macro_updates": self.macro_updates,
            "planner_updates": self.planner_updates,
            "epsilon_schedule": self.epsilon_schedule_state(),
            # ---- exact-resume state (schema 8) --------------------------
            # Without these a "resume" restores the same weights into a fresh
            # trajectory: empty replays and a rewound exploration stream. The
            # three streams below were audited as the ones that actually feed
            # training:
            #   mixing rng   -> LOCAL/GLOBAL mode and the Dirichlet draw
            #   action rng   -> replay sampling AND epsilon-greedy choice
            #   torch rng    -> Transformer dropout (p=0.1) inside the planner
            #                   forward that produces the planner gradient
            "exact_resume_capable": True,
            "replay": self.replay.state_dict(),
            "macro_replay": self.macro_replay.state_dict(),
            "mixing_rng_state": self._mixing_rng.bit_generator.state,
            "action_rng_state": (None if self._action_rng is None
                                 else self._action_rng.getstate()),
            "torch_rng_state": torch.get_rng_state(),
            "local_draws": int(self.local_draws),
            "global_draws": int(self.global_draws),
        }

    def load_state_dict(self, payload: Dict[str, Any],
                        exact_resume: bool = False) -> None:
        from tvt_config import (
            ACTION_REPRESENTATION,
            ACTION_SEMANTICS_VERSION,
            ARCHITECTURE_VERSION,
            DUPLICATION_SCHEDULER,
            E2E_CHECKPOINT_SCHEMA_VERSION,
            LOWER_REWARD_VERSION,
            PRIMARY_PATH_POLICY,
            THROUGHPUT_PRIMITIVE_VERSION,
            UPPER_OBJECTIVE_VERSION,
        )
        expected = {
            "architecture_version": ARCHITECTURE_VERSION,
            "action_dim": ACTION_DIM,
            "action_semantics": ACTION_SEMANTICS_VERSION,
            "action_representation": ACTION_REPRESENTATION,
            "duplication_scheduler": DUPLICATION_SCHEDULER,
            "primary_path_policy": PRIMARY_PATH_POLICY,
            "throughput_primitive_version": THROUGHPUT_PRIMITIVE_VERSION,
            "lower_reward_version": LOWER_REWARD_VERSION,
            "planner_parameterization": PLANNER_PARAMETERIZATION,
            "goal_dim": GOAL_DIM,
            "planner_kind": self.planner_kind,
            "context_feature_dim": CONTEXT_FEATURE_DIM,
            "radio_state_dim": RADIO_STATE_DIM,
            "upper_objective_version": UPPER_OBJECTIVE_VERSION,
            "macro_horizon": self.macro_horizon,
            "action_hold_enabled": ACTION_HOLD_ENABLED,
            "macro_interval_basis": MACRO_INTERVAL_BASIS,
            "reliability_attribution_version": RELIABILITY_ATTRIBUTION_VERSION,
            "exploration_protocol_version": EXPLORATION_PROTOCOL_VERSION,
            # Absent from every pre-D-3C checkpoint. The loop below skips a
            # missing key ONLY when exact_resume is False, so an old-operator
            # checkpoint stays READABLE for diagnostics and is refused for an
            # exact training resume -- exactly the required asymmetry.
            "goal_critic_operator": GOAL_CRITIC_OPERATOR,
        }
        # The UPPER tier (Transformer planner and GoalCritic) is shaped by
        # CONTEXT_FEATURE_DIM, so a checkpoint written under a different
        # observation contract cannot be loaded into it. Refused by name, with
        # the contract numbers, rather than via an opaque tensor-shape error.
        # The LOWER tier is unaffected by the context width and a lower state
        # dict from such a checkpoint remains structurally loadable on its own.
        _ckpt_obs = payload.get("observation_contract_version")
        if _ckpt_obs is not None and int(_ckpt_obs) != int(
                OBSERVATION_CONTRACT_VERSION):
            raise ValueError(
                "observation contract mismatch: checkpoint was written under "
                f"observation contract {int(_ckpt_obs)} "
                f"(context_feature_dim={payload.get('context_feature_dim')}), "
                f"this code runs observation contract "
                f"{OBSERVATION_CONTRACT_VERSION} "
                f"(context_feature_dim={CONTEXT_FEATURE_DIM}). The upper "
                "planner and GoalCritic are shaped by the context width and "
                "will not load it.")
        if _ckpt_obs is None:
            raise ValueError(
                "observation contract mismatch: checkpoint records no "
                "observation_contract_version, so it predates observation "
                f"contract {OBSERVATION_CONTRACT_VERSION}. The upper planner "
                "and GoalCritic are shaped by the context width and will not "
                "load it.")
        # A checkpoint from the retired {Uu, PC5, Both} action space has a
        # three-output Q head over a categorical path choice. Its weights are
        # not a starting point for a five-level duplication head, so it is
        # refused by name rather than by an opaque shape error.
        if int(payload.get("action_dim", -1)) == 3:
            raise ValueError(
                "checkpoint was produced under the RETIRED {Uu, PC5, Both} "
                "path-selection action space (action_dim=3); this "
                "architecture selects a packet duplication degree over "
                f"{ACTION_DIM} levels and will not load it")
        # A checkpoint whose r_thr was trained against the retired proxy
        # learned a Q function over a systematically biased third component.
        # Refused by name so the failure says what is actually wrong.
        from tvt_config import RETIRED_LOWER_REWARD_VERSIONS
        if payload.get("lower_reward_version") in RETIRED_LOWER_REWARD_VERSIONS:
            raise ValueError(
                "checkpoint was trained with the RETIRED max-path-throughput "
                "proxy as r_thr (lower_reward_version="
                f"{payload.get('lower_reward_version')!r}); this architecture "
                "measures unique logical goodput and will not load it")
        # A checkpoint written by a planner with raw unbounded logits encodes
        # a saturated vertex solution whose gradient is dead. It is not an
        # initialization for the ball-projected parameterization, so it is
        # refused by name rather than by a silent semantic change.
        from tvt_config import RETIRED_PLANNER_PARAMETERIZATIONS
        legacy_param = payload.get("planner_parameterization")
        if legacy_param in RETIRED_PLANNER_PARAMETERIZATIONS:
            raise ValueError(
                f"checkpoint planner_parameterization={legacy_param!r} is "
                "RETIRED; it predates the anti-saturation parameterization "
                "and encodes a collapsed, gradient-free planner")
        legacy_arch = payload.get("architecture_version")
        from tvt_config import RETIRED_ARCHITECTURE_VERSIONS
        if legacy_arch in RETIRED_ARCHITECTURE_VERSIONS:
            # Do not assert a single reason here: the retired list spans
            # several distinct breaks (MP-QUIC era, the path-selection action
            # space, and the unbounded-logit planner). Naming the wrong one
            # would send a reader after the wrong incompatibility.
            raise ValueError(
                f"checkpoint architecture_version={legacy_arch!r} is RETIRED "
                f"and is not an initialization for {ARCHITECTURE_VERSION!r}")
        # A checkpoint written under the retired logit-space mechanism carries
        # goal_noise_* and no beta. Loading it would run with an undefined
        # behaviour policy, so it is refused rather than coerced.
        if any(key in payload for key in ("goal_noise_scale_initial",
                                          "goal_noise_scale", "goal_noise_min")):
            raise ValueError(
                "checkpoint was produced under the RETIRED logit-space "
                "exploration mechanism; screening protocol v2 uses simplex "
                "Dirichlet mixing and will not load it")
        # The schema version is NOT an equality gate. A bump adds resume state;
        # it does not invalidate the evidence an older run produced. An older
        # schema therefore still loads for analysis and evaluation, and is
        # refused only where it actually matters -- an exact training resume,
        # handled in _load_exact_resume_state. Every genuine semantic break
        # (action space, reward, parameterization, ...) has its own entry in
        # `expected` and still fails loudly.
        schema = int(payload.get("checkpoint_schema_version", 0))
        if schema > E2E_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema {schema} is NEWER than this code's "
                f"{E2E_CHECKPOINT_SCHEMA_VERSION}; refusing to guess at it")
        if exact_resume and schema != E2E_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "LEGACY_CHECKPOINT_NOT_EXACT_RESUMABLE: checkpoint schema "
                f"{schema}, this code writes {E2E_CHECKPOINT_SCHEMA_VERSION}. "
                "It loads fine for evaluation and offline analysis "
                "(resume_training=False); it cannot continue a trajectory.")
        for key, want in expected.items():
            if key not in payload and not exact_resume:
                # The key was added after this checkpoint was written. For an
                # ANALYSIS load that is not a break: the run simply predates
                # the field. An exact resume never reaches here, because the
                # schema gate above already refused an older schema.
                continue
            got = payload.get(key)
            if got != want:
                raise ValueError(
                    f"incompatible checkpoint: {key}={got!r}, expected {want!r}")
        if payload.get("mpquic_enabled", False):
            raise ValueError("checkpoint was produced with MP-QUIC enabled")
        self.planner.load_state_dict(payload["planner"])
        self.planner_target.load_state_dict(payload["planner_target"])
        self.lower.load_state_dict(payload["lower"])
        self.lower_target.load_state_dict(payload["lower_target"])
        self.goal_critic.load_state_dict(payload["goal_critic"])
        self.goal_critic_target.load_state_dict(payload["goal_critic_target"])
        self.opt_lower.load_state_dict(payload["opt_lower"])
        self.opt_critic.load_state_dict(payload["opt_critic"])
        if self.opt_planner is not None and payload.get("opt_planner"):
            self.opt_planner.load_state_dict(payload["opt_planner"])
        self.updates = int(payload.get("updates", 0))
        self.macro_updates = int(payload.get("macro_updates", 0))
        self.planner_updates = int(payload.get("planner_updates", 0))
        if payload.get("duplication_constraint"):
            self.duplication.load_state_dict(payload["duplication_constraint"])
        if exact_resume:
            # The derived decay is a function of (E, N), so it is validated and
            # restored only where a trajectory is actually continued. On the
            # evaluation / analysis path (exact_resume=False) the schedule is
            # discarded regardless -- load() sets epsilon_decay_step to 0 and
            # the runner sets epsilon to 0 -- so validating it there would only
            # refuse to read a checkpoint for a run with a different E, which
            # is exactly what an evaluation at another episode length needs.
            self._load_epsilon_schedule(payload)
            self._load_exact_resume_state(payload)

    def _load_exact_resume_state(self, payload: Dict[str, Any]) -> None:
        """Restore the state a bare weight-load leaves behind.

        A checkpoint written before schema 8 has none of it. Refusing that
        case is the point: restoring weights into empty replays and a rewound
        exploration stream starts a NEW trajectory, and calling that a
        continuation is how a diagnostic silently measures the wrong thing.
        """
        from tvt_config import E2E_CHECKPOINT_SCHEMA_VERSION as _schema
        if not payload.get("exact_resume_capable"):
            raise ValueError(
                "LEGACY_CHECKPOINT_NOT_EXACT_RESUMABLE: this checkpoint "
                f"predates schema {_schema} and carries "
                "no replay or RNG state. It can still be loaded for "
                "evaluation or offline analysis (resume_training=False), but "
                "resuming training from it would start a different "
                "trajectory, not continue this one.")
        # The optimizer protocol is deliberately NOT in `expected`: it changes
        # how training proceeds, not what the stored tensors mean, so it must
        # not block reading a completed run. It does block CONTINUING one.
        stored_protocol = payload.get("e2e_timescale_protocol")
        if stored_protocol != E2E_TIMESCALE_PROTOCOL:
            raise ValueError(
                f"cannot exactly resume: checkpoint was trained under "
                f"{stored_protocol!r}, this code runs {E2E_TIMESCALE_PROTOCOL!r}. "
                "It remains loadable for evaluation and offline analysis.")
        self.replay.load_state_dict(payload["replay"])
        _macro_state = payload["macro_replay"]
        for _item in (_macro_state.get("items") or [])[:1]:
            _ctx = getattr(_item, "context", None)
            if _ctx is not None and np.asarray(_ctx).shape[-1] != (
                    CONTEXT_FEATURE_DIM):
                raise ValueError(
                    "observation contract mismatch: macro replay stores "
                    f"context frames of width {np.asarray(_ctx).shape[-1]}, "
                    f"this contract uses {CONTEXT_FEATURE_DIM}")
        self.macro_replay.load_state_dict(_macro_state)
        self._mixing_rng.bit_generator.state = payload["mixing_rng_state"]
        torch.set_rng_state(
            torch.as_tensor(payload["torch_rng_state"], dtype=torch.uint8))
        action_state = payload.get("action_rng_state")
        if action_state is not None:
            if self._action_rng is None:
                raise ValueError(
                    "the checkpoint carries an action/sampling RNG state but "
                    "no RNG was attached: call attach_rng(rng) before an "
                    "exact resume, or replay sampling and the epsilon-greedy "
                    "stream would silently restart")
            self._action_rng.setstate(_as_rng_state(action_state))
        self.local_draws = int(payload.get("local_draws", 0))
        self.global_draws = int(payload.get("global_draws", 0))

    def _load_epsilon_schedule(self, payload: Dict[str, Any]) -> None:
        """Restore the exploration schedule, refusing an incompatible one.

        The derived decay is a function of (E, N), so resuming a run under a
        different horizon would silently change the exploration schedule
        mid-run. Every field that participates in the derivation is therefore
        compared, and a mismatch is a load failure rather than a warning.
        """
        saved = payload.get("epsilon_schedule")
        if not saved:
            return
        current = self.epsilon_schedule_state()
        for key in ("epsilon_decay_mode", "epsilon_schedule_version",
                    "training_horizon_N", "sim_time_E", "n_formal",
                    "total_epsilon_decay_steps"):
            want, got = current.get(key), saved.get(key)
            if want is None or got is None:
                continue
            if want != got:
                raise ValueError(
                    f"incompatible epsilon schedule: {key}={got!r} in the "
                    f"checkpoint, {want!r} requested. The derived decay is a "
                    "function of (E, N); resuming under a different budget "
                    "would change the exploration schedule mid-run.")
        saved_decay = saved.get("epsilon_decay")
        if saved_decay is not None and self.epsilon_decay > 0.0:
            if abs(float(saved_decay) - self.epsilon_decay) > 1e-15:
                raise ValueError(
                    f"incompatible epsilon_decay: checkpoint {saved_decay!r}, "
                    f"requested {self.epsilon_decay!r}")
        self.epsilon_decay_step = int(saved.get("epsilon_decay_step", 0))
        # Closed form, so the resumed trajectory is identical rather than
        # merely close. The stored epsilon is only cross-checked.
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start * (self.epsilon_decay ** self.epsilon_decay_step))

    def save(self, path) -> None:
        payload = self.state_dict()
        payload["epsilon"] = float(getattr(self, "epsilon", 0.0))
        torch.save(payload, str(path))

    def load(self, path, resume_training: bool = True,
             exact_resume: Optional[bool] = None) -> None:
        """Load a checkpoint.

        resume_training=False is the ANALYSIS / EVALUATION path and accepts a
        legacy checkpoint: the historical STEP 4B/4C evidence must stay
        readable. resume_training=True defaults to demanding exact-resume
        state, because that is the only way "continuing a trajectory" means
        what it says.
        """
        payload = torch.load(str(path), map_location=self.device,
                             weights_only=False)
        if exact_resume is None:
            exact_resume = bool(resume_training)
        self.load_state_dict(payload, exact_resume=exact_resume)
        if not resume_training:
            # EVAL: the schedule is not resumed at all. epsilon is set to 0 by
            # the caller and no decay, replay write, optimizer step or
            # checkpoint write happens for the rest of the process.
            self.epsilon_decay_step = 0
