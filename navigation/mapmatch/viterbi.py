"""Strictly causal online fixed-lag sliding window Viterbi decoder (Phase 12).

Characteristics:
1. Strictly causal: at current time step t, only observations up to t are ingested.
2. Fixed lag: with lag W, decision for epoch t - W is committed only after W future
   epochs of evidence have arrived (buffer size W + 1).
3. Bounded memory: older epochs are pruned from the trellis buffer upon commitment.
4. Deterministic: tie-breaking by candidate edge_id ordering.
5. Zero future lookahead beyond the current online processing epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple
from collections import deque
import numpy as np

from navigation.mapmatch.candidates import RoadCandidate
from navigation.mapmatch.transition import TransitionModel, LOG_ZERO


@dataclass(frozen=True)
class ViterbiCommit:
    """A committed map matching candidate decision from the fixed-lag trellis."""
    epoch_idx: int
    timestamp_ns: int
    traj_pos_enu: Tuple[float, float]
    candidate: Optional[RoadCandidate]
    best_score: float
    second_best_score: float
    margin: float
    is_mature: bool
    fallback_reason: Optional[str] = None


@dataclass
class _TrellisEpoch:
    """Internal record for one epoch inside the fixed-lag sliding window."""
    epoch_idx: int
    timestamp_ns: int
    traj_pos_enu: Tuple[float, float]
    candidates: List[RoadCandidate]
    log_emissions: np.ndarray  # (M,)
    viterbi_scores: np.ndarray  # (M,)
    backpointers: np.ndarray  # (M,) int
    best_candidate_idx: int
    second_best_candidate_idx: Optional[int]


class FixedLagViterbi:
    """Causal fixed-lag sliding window Viterbi trellis decoder."""

    def __init__(
        self,
        transition_model: TransitionModel,
        lag_epochs: int = 8,
    ) -> None:
        """Initialize fixed-lag Viterbi decoder.

        Args:
            transition_model: TransitionModel instance for computing graph routing likelihoods.
            lag_epochs: Lag window size W in epochs (default 8 epochs, within [5, 10]).
        """
        if lag_epochs < 1:
            raise ValueError(f"lag_epochs must be at least 1, got {lag_epochs}")

        self.transition_model = transition_model
        self.lag_epochs = int(lag_epochs)
        self._buffer: Deque[_TrellisEpoch] = deque()
        self._current_epoch_idx: int = 0

    def reset(self) -> None:
        """Clear all active trellis history."""
        self._buffer.clear()
        self._current_epoch_idx = 0

    def step(
        self,
        timestamp_ns: int,
        traj_pos_enu: Tuple[float, float],
        candidates: List[RoadCandidate],
        log_emissions: Sequence[float],
        delta_t_s: Optional[float] = None,
    ) -> Optional[ViterbiCommit]:
        """Ingest observation at current time t and return mature committed decision if available.

        Strictly causal: uses observations ONLY up to current time t.

        Args:
            timestamp_ns: Timestamp of current sample in nanoseconds.
            traj_pos_enu: Current trajectory point (east, north) [meters].
            candidates: List of RoadCandidate objects for current epoch.
            log_emissions: Log emission probabilities for each candidate.
            delta_t_s: Elapsed time since previous epoch in seconds.

        Returns:
            ViterbiCommit for mature epoch (t - W) if buffer has filled, else None.
        """
        epoch_idx = self._current_epoch_idx
        self._current_epoch_idx += 1

        num_cands = len(candidates)

        # Handle empty candidate set
        if num_cands == 0:
            commit_out = None
            # If buffer has items, we must flush the oldest mature item
            if len(self._buffer) > self.lag_epochs:
                commit_out = self._commit_oldest()

            # Append empty epoch marker
            empty_epoch = _TrellisEpoch(
                epoch_idx=epoch_idx,
                timestamp_ns=timestamp_ns,
                traj_pos_enu=traj_pos_enu,
                candidates=[],
                log_emissions=np.empty(0, dtype=np.float64),
                viterbi_scores=np.empty(0, dtype=np.float64),
                backpointers=np.empty(0, dtype=np.int64),
                best_candidate_idx=-1,
                second_best_candidate_idx=None,
            )
            self._buffer.append(empty_epoch)
            return commit_out

        emissions_arr = np.asarray(log_emissions, dtype=np.float64)
        viterbi_scores = np.full(num_cands, LOG_ZERO, dtype=np.float64)
        backpointers = np.full(num_cands, -1, dtype=np.int64)

        prev_epoch = self._buffer[-1] if self._buffer else None
        has_valid_prev = prev_epoch is not None and len(prev_epoch.candidates) > 0

        if not has_valid_prev:
            # First epoch or recovery after empty candidates: initialize directly from emissions
            viterbi_scores[:] = emissions_arr
        else:
            assert prev_epoch is not None
            prev_scores = prev_epoch.viterbi_scores
            prev_cands = prev_epoch.candidates
            prev_traj = prev_epoch.traj_pos_enu

            # DP update for each current candidate j
            for j, c_curr in enumerate(candidates):
                best_trans_val = LOG_ZERO
                best_prev_idx = 0

                for i, c_prev in enumerate(prev_cands):
                    if prev_scores[i] <= LOG_ZERO:
                        continue
                    log_trans = self.transition_model.compute_log_transition(
                        c_prev=c_prev,
                        c_curr=c_curr,
                        prev_traj_enu=prev_traj,
                        curr_traj_enu=traj_pos_enu,
                        delta_t_s=delta_t_s,
                    )
                    candidate_total = prev_scores[i] + log_trans

                    if candidate_total > best_trans_val:
                        best_trans_val = candidate_total
                        best_prev_idx = i
                    elif abs(candidate_total - best_trans_val) < 1e-12:
                        # Deterministic tie-breaking by edge ID
                        if prev_cands[i].edge_id < prev_cands[best_prev_idx].edge_id:
                            best_trans_val = candidate_total
                            best_prev_idx = i

                viterbi_scores[j] = best_trans_val + emissions_arr[j]
                backpointers[j] = best_prev_idx

        # Rank current epoch candidates
        sorted_indices = sorted(
            range(num_cands),
            key=lambda k: (-viterbi_scores[k], candidates[k].edge_id),
        )
        best_cand_idx = sorted_indices[0]
        second_best_cand_idx = sorted_indices[1] if num_cands > 1 else None

        current_epoch = _TrellisEpoch(
            epoch_idx=epoch_idx,
            timestamp_ns=timestamp_ns,
            traj_pos_enu=traj_pos_enu,
            candidates=candidates,
            log_emissions=emissions_arr,
            viterbi_scores=viterbi_scores,
            backpointers=backpointers,
            best_candidate_idx=best_cand_idx,
            second_best_candidate_idx=second_best_cand_idx,
        )
        self._buffer.append(current_epoch)

        # If buffer length exceeds lag window W, commit the mature tail epoch
        if len(self._buffer) > self.lag_epochs:
            return self._commit_oldest()

        return None

    def _commit_oldest(self) -> ViterbiCommit:
        """Trace back from current newest epoch to the oldest epoch in buffer and commit it."""
        assert len(self._buffer) > 0

        # Trace back starting from best candidate of the newest epoch
        newest = self._buffer[-1]
        best_idx = newest.best_candidate_idx

        if best_idx < 0 or len(newest.candidates) == 0:
            oldest = self._buffer.popleft()
            return ViterbiCommit(
                epoch_idx=oldest.epoch_idx,
                timestamp_ns=oldest.timestamp_ns,
                traj_pos_enu=oldest.traj_pos_enu,
                candidate=None,
                best_score=LOG_ZERO,
                second_best_score=LOG_ZERO,
                margin=0.0,
                is_mature=True,
                fallback_reason="NO_CANDIDATES",
            )

        # Follow backpointers down to oldest epoch
        path_indices = [best_idx]
        for ep in reversed(list(self._buffer)[1:]):
            prev_idx = ep.backpointers[best_idx] if best_idx >= 0 and best_idx < len(ep.backpointers) else -1
            path_indices.append(prev_idx)
            best_idx = prev_idx

        path_indices.reverse()
        oldest_choice_idx = path_indices[0]

        oldest = self._buffer.popleft()

        if oldest_choice_idx < 0 or oldest_choice_idx >= len(oldest.candidates):
            return ViterbiCommit(
                epoch_idx=oldest.epoch_idx,
                timestamp_ns=oldest.timestamp_ns,
                traj_pos_enu=oldest.traj_pos_enu,
                candidate=None,
                best_score=LOG_ZERO,
                second_best_score=LOG_ZERO,
                margin=0.0,
                is_mature=True,
                fallback_reason="NO_CANDIDATES",
            )

        committed_cand = oldest.candidates[oldest_choice_idx]
        best_score = float(oldest.viterbi_scores[oldest_choice_idx])

        # Compute margin between committed candidate and best alternative in oldest epoch
        other_scores = [float(oldest.viterbi_scores[k]) for k in range(len(oldest.candidates)) if k != oldest_choice_idx]
        if other_scores:
            second_score = max(other_scores)
            margin = max(0.0, best_score - second_score) if second_score > LOG_ZERO else 100.0
        else:
            second_score = LOG_ZERO
            margin = 100.0

        return ViterbiCommit(
            epoch_idx=oldest.epoch_idx,
            timestamp_ns=oldest.timestamp_ns,
            traj_pos_enu=oldest.traj_pos_enu,
            candidate=committed_cand,
            best_score=best_score,
            second_best_score=second_score,
            margin=margin,
            is_mature=True,
            fallback_reason=None,
        )

    def flush_remaining(self) -> List[ViterbiCommit]:
        """Flush and commit all remaining uncommitted epochs in the buffer (at stream completion)."""
        commits: List[ViterbiCommit] = []
        while len(self._buffer) > 0:
            commits.append(self._commit_oldest())
        return commits
