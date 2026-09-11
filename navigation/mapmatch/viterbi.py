"""Fixed-Lag Sliding Window Viterbi Decoder for HMM map matching."""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from navigation.mapmatch.candidates import RoadCandidate
from navigation.mapmatch.transition import TransitionModel, LOG_ZERO

@dataclass(frozen=True)
class ViterbiCommit:
    """A committed map matching result for a specific epoch after the lag window."""
    epoch_idx: int
    timestamp_ns: int
    traj_pos_enu: Tuple[float, float]
    candidate: Optional[RoadCandidate]
    best_score: float
    second_best_score: float
    margin: float
    is_mature: bool
    fallback_reason: Optional[str]
    candidate_count: int
    second_best_candidate: Optional[RoadCandidate] = None

@dataclass
class _TrellisEpoch:
    """Internal Trellis representation for one epoch."""
    epoch_idx: int
    timestamp_ns: int
    traj_pos_enu: Tuple[float, float]
    candidates: List[RoadCandidate]
    log_emissions: np.ndarray
    viterbi_scores: np.ndarray
    backpointers: np.ndarray
    best_candidate_idx: int
    second_best_candidate_idx: Optional[int]

class FixedLagViterbi:
    """Fixed-lag sliding window Viterbi decoding algorithm."""
    
    def __init__(self, transition_model: TransitionModel, lag_epochs: int = 8):
        self.transition_model = transition_model
        self.lag_epochs = max(1, lag_epochs)
        self._buffer: deque[_TrellisEpoch] = deque(maxlen=lag_epochs + 1)
        self._current_epoch_idx = 0

    def reset(self) -> None:
        """Clear internal trellis state."""
        self._buffer.clear()
        self._current_epoch_idx = 0

    def step(
        self,
        timestamp_ns: int,
        traj_pos_enu: Tuple[float, float],
        candidates: List[RoadCandidate],
        log_emissions: List[float],
        delta_t_s: Optional[float],
    ) -> Optional[ViterbiCommit]:
        """
        Step the Viterbi trellis with current observations.
        
        Returns:
            ViterbiCommit for mature epoch (t - W) if buffer filled, else None.
        """
        epoch_idx = self._current_epoch_idx
        self._current_epoch_idx += 1
        
        num_cands = len(candidates)
        
        if num_cands == 0:
            commit_out = None
            if len(self._buffer) >= self.lag_epochs:
                commit_out = self._commit_oldest()
                
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
            viterbi_scores[:] = emissions_arr
        else:
            assert prev_epoch is not None
            prev_scores = prev_epoch.viterbi_scores
            prev_cands = prev_epoch.candidates
            prev_traj = prev_epoch.traj_pos_enu
            
            for j, c_curr in enumerate(candidates):
                best_trans_val = LOG_ZERO
                best_prev_idx = -1
                
                for i, c_prev in enumerate(prev_cands):
                    if prev_scores[i] <= LOG_ZERO:
                        continue
                    log_trans = self.transition_model.compute_log_transition(
                        c_prev=c_prev,
                        c_curr=c_curr,
                        prev_traj_enu=prev_traj,
                        curr_traj_enu=traj_pos_enu,
                        delta_t_s=delta_t_s or 1.0,
                    )
                    score = prev_scores[i] + log_trans
                    if score > best_trans_val:
                        best_trans_val = score
                        best_prev_idx = i
                        
                viterbi_scores[j] = best_trans_val + emissions_arr[j]
                backpointers[j] = best_prev_idx

        # Rank candidates
        sorted_indices = sorted(range(num_cands), key=lambda k: -viterbi_scores[k])
        best_cand_idx = sorted_indices[0]
        second_best_cand_idx = sorted_indices[1] if num_cands > 1 else None
        
        curr_epoch = _TrellisEpoch(
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
        self._buffer.append(curr_epoch)
        
        if len(self._buffer) > self.lag_epochs:
            return self._commit_oldest()
            
        return None
        
    def _commit_oldest(self) -> ViterbiCommit:
        """Trace back and commit the oldest epoch in the sliding window."""
        newest = self._buffer[-1]
        best_idx = newest.best_candidate_idx
        
        if best_idx < 0 or len(newest.candidates) == 0:
            oldest = self._buffer.popleft()
            return self._build_empty_commit(oldest)
            
        path_indices = [best_idx]
        for ep in reversed(list(self._buffer)[1:]):
            prev_idx = ep.backpointers[best_idx] if 0 <= best_idx < len(ep.backpointers) else -1
            path_indices.append(prev_idx)
            best_idx = prev_idx
            
        path_indices.reverse()
        oldest_choice_idx = path_indices[0]
        
        oldest = self._buffer.popleft()
        
        if oldest_choice_idx < 0 or oldest_choice_idx >= len(oldest.candidates):
            return self._build_empty_commit(oldest)
            
        cand = oldest.candidates[oldest_choice_idx]
        best_score = float(oldest.viterbi_scores[oldest_choice_idx])
        
        other_scores = [(k, float(oldest.viterbi_scores[k])) for k in range(len(oldest.candidates)) if k != oldest_choice_idx]
        if other_scores:
            second_idx, second_score = max(other_scores, key=lambda x: x[1])
            second_cand = oldest.candidates[second_idx]
            margin = max(0.0, best_score - second_score) if second_score > LOG_ZERO else 100.0
        else:
            second_score = LOG_ZERO
            second_cand = None
            margin = 100.0
            
        return ViterbiCommit(
            epoch_idx=oldest.epoch_idx,
            timestamp_ns=oldest.timestamp_ns,
            traj_pos_enu=oldest.traj_pos_enu,
            candidate=cand,
            best_score=best_score,
            second_best_score=second_score,
            margin=margin,
            is_mature=True,
            fallback_reason=None,
            candidate_count=len(oldest.candidates),
            second_best_candidate=second_cand,
        )
        
    def _build_empty_commit(self, oldest: _TrellisEpoch) -> ViterbiCommit:
        return ViterbiCommit(
            epoch_idx=oldest.epoch_idx,
            timestamp_ns=oldest.timestamp_ns,
            traj_pos_enu=oldest.traj_pos_enu,
            candidate=None,
            best_score=LOG_ZERO,
            second_best_score=LOG_ZERO,
            margin=0.0,
            is_mature=True,
            fallback_reason="NO_CANDIDATES" if len(oldest.candidates) == 0 else "NO_VALID_PATH",
            candidate_count=len(oldest.candidates),
            second_best_candidate=None,
        )
        
    def flush_remaining(self) -> List[ViterbiCommit]:
        commits = []
        while len(self._buffer) > 0:
            commits.append(self._commit_oldest())
        return commits
