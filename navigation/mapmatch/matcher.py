"""Downstream Map Matching engine and trajectory snapper with safe fallback (Phase 12).

Architectural Invariants:
1. STRICTLY DOWNSTREAM: Map matching never feeds back into ESKF nominal state,
   covariance, velocity, attitude, sensor biases, ML models, GNSS FSM, NHC, or ZUPT.
2. SAFE FALLBACK: Whenever confidence is low, coverage is absent, parallel roads are
   ambiguous, or snap displacement exceeds threshold, returns the EXACT unsnapped
   estimator coordinates.
3. CAUSAL FIXED-LAG: Trellis decisions are committed after fixed lag W epochs.
4. COORDINATE HARMONIZATION: Road graph and trajectory are evaluated in the identical
   session-local ENU frame via the active GeoReference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from maps.extract_osm import RoadNetworkGraph
from navigation.frames.local_geo import GeoReference
from navigation.mapmatch.candidates import CandidateSearch, RoadCandidate
from navigation.mapmatch.emission import EmissionModel
from navigation.mapmatch.transition import TransitionModel, LOG_ZERO
from navigation.mapmatch.viterbi import FixedLagViterbi, ViterbiCommit
from navigation.schemas.mapmatch import MapMatchResult
from navigation.schemas.state import NavigationState


@dataclass(frozen=True)
class MapMatchOutput:
    """Complete downstream map matching output contract."""
    timestamp_ns: int
    estimator_lat_lon: Tuple[float, float]
    estimator_enu: Tuple[float, float, float]
    display_lat_lon: Tuple[float, float]
    display_enu: Tuple[float, float, float]
    snapped: bool
    confidence: float
    matched_edge_id: Optional[str]
    distance_to_road_m: Optional[float]
    road_heading_rad: Optional[float]
    fallback_reason: Optional[str]
    candidate_count: int
    best_score: float
    second_best_score: float
    margin: float
    result_schema: MapMatchResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "estimator_lat_lon": list(self.estimator_lat_lon),
            "estimator_enu": list(self.estimator_enu),
            "display_lat_lon": list(self.display_lat_lon),
            "display_enu": list(self.display_enu),
            "snapped": self.snapped,
            "confidence": self.confidence,
            "matched_edge_id": self.matched_edge_id,
            "distance_to_road_m": self.distance_to_road_m,
            "road_heading_rad": self.road_heading_rad,
            "fallback_reason": self.fallback_reason,
            "candidate_count": self.candidate_count,
            "best_score": self.best_score,
            "second_best_score": self.second_best_score,
            "margin": self.margin,
        }


class MapMatcher:
    """Downstream HMM map matcher and trajectory snapper with safe fallback."""

    def __init__(
        self,
        road_graph: RoadNetworkGraph,
        search_radius_m: float = 35.0,
        max_snap_distance_m: float = 25.0,
        min_confidence: float = 0.45,
        ambiguity_margin: float = 1.0,
        lag_epochs: int = 8,
        sigma_road_m: float = 4.0,
        beta_m: float = 5.0,
    ) -> None:
        """Initialize MapMatcher with frozen, documented configuration parameters.

        Args:
            road_graph: Active RoadNetworkGraph with GeoReference bound.
            search_radius_m: Radius to search for road candidates (default 35.0 m).
            max_snap_distance_m: Hard displacement threshold above which snap is rejected (default 25.0 m).
            min_confidence: Minimum normalized confidence to accept a snap (default 0.45).
            ambiguity_margin: Log-likelihood margin below which parallel roads are declared ambiguous (default 1.0).
            lag_epochs: Number of epochs to lag sliding-window Viterbi (default 8).
            sigma_road_m: Road measurement uncertainty (default 4.0 m).
            beta_m: Scale parameter for transition discrepancy (default 5.0 m).
        """
        self.road_graph = road_graph
        self.search_radius_m = float(search_radius_m)
        self.max_snap_distance_m = float(max_snap_distance_m)
        self.min_confidence = float(min_confidence)
        self.ambiguity_margin = float(ambiguity_margin)
        self.lag_epochs = int(lag_epochs)

        self.candidate_search = CandidateSearch(
            road_graph=road_graph,
            search_radius_m=self.search_radius_m,
        )
        self.emission_model = EmissionModel(
            sigma_road_m=sigma_road_m,
        )
        self.transition_model = TransitionModel(
            road_graph=road_graph,
            beta_m=beta_m,
        )
        self.viterbi = FixedLagViterbi(
            transition_model=self.transition_model,
            lag_epochs=self.lag_epochs,
        )

        self._last_timestamp_ns: Optional[int] = None
        self._last_committed_edge: Optional[str] = None

    def reset(self) -> None:
        """Reset matcher streaming state."""
        self.viterbi.reset()
        self._last_timestamp_ns = None
        self._last_committed_edge = None

    def _evaluate_commit(
        self,
        commit: ViterbiCommit,
        estimator_lat_lon: Tuple[float, float],
        estimator_enu: Tuple[float, float, float],
        candidate_count: Optional[int] = None,
    ) -> MapMatchOutput:
        """Apply anti-catastrophic-snap safeguards and construct output.

        Strictly causal and mature: evaluates the committed mature epoch using
        the committed epoch's own metadata (candidate_count, scores, margin)
        without contamination from current or subsequent ingestion epochs.
        """
        cand_count = commit.candidate_count if candidate_count is None else candidate_count
        cand = commit.candidate

        # Check 1: Empty candidate / no coverage
        if cand is None:
            reason = commit.fallback_reason or "NO_CANDIDATES"
            res_schema = MapMatchResult(snapped=False, confidence=0.0)
            return MapMatchOutput(
                timestamp_ns=commit.timestamp_ns,
                estimator_lat_lon=estimator_lat_lon,
                estimator_enu=estimator_enu,
                display_lat_lon=estimator_lat_lon,
                display_enu=estimator_enu,
                snapped=False,
                confidence=0.0,
                matched_edge_id=None,
                distance_to_road_m=None,
                road_heading_rad=None,
                fallback_reason=reason,
                candidate_count=cand_count,
                best_score=commit.best_score,
                second_best_score=commit.second_best_score,
                margin=commit.margin,
                result_schema=res_schema,
            )

        # Check 2: Maximum snap displacement check
        snap_dist = cand.distance_to_road_m
        if snap_dist > self.max_snap_distance_m:
            res_schema = MapMatchResult(
                snapped=False,
                confidence=0.0,
                distance_to_road_m=snap_dist,
            )
            return MapMatchOutput(
                timestamp_ns=commit.timestamp_ns,
                estimator_lat_lon=estimator_lat_lon,
                estimator_enu=estimator_enu,
                display_lat_lon=estimator_lat_lon,
                display_enu=estimator_enu,
                snapped=False,
                confidence=0.0,
                matched_edge_id=cand.edge_id,
                distance_to_road_m=snap_dist,
                road_heading_rad=cand.edge_azimuth_rad,
                fallback_reason="LARGE_DISPLACEMENT",
                candidate_count=cand_count,
                best_score=commit.best_score,
                second_best_score=commit.second_best_score,
                margin=commit.margin,
                result_schema=res_schema,
            )

        # Check 3: Parallel-road ambiguity margin check
        # Evaluated using the mature epoch's candidate count and margin
        if cand_count > 1 and commit.margin < self.ambiguity_margin:
            res_schema = MapMatchResult(
                snapped=False,
                confidence=0.2,
                matched_road_id=cand.edge_id,
                distance_to_road_m=snap_dist,
            )
            return MapMatchOutput(
                timestamp_ns=commit.timestamp_ns,
                estimator_lat_lon=estimator_lat_lon,
                estimator_enu=estimator_enu,
                display_lat_lon=estimator_lat_lon,
                display_enu=estimator_enu,
                snapped=False,
                confidence=0.2,
                matched_edge_id=cand.edge_id,
                distance_to_road_m=snap_dist,
                road_heading_rad=cand.edge_azimuth_rad,
                fallback_reason="AMBIGUOUS_PARALLEL_ROADS",
                candidate_count=cand_count,
                best_score=commit.best_score,
                second_best_score=commit.second_best_score,
                margin=commit.margin,
                result_schema=res_schema,
            )

        # Check 4: Graph transition disconnection
        if commit.best_score <= LOG_ZERO / 2:
            res_schema = MapMatchResult(snapped=False, confidence=0.0)
            return MapMatchOutput(
                timestamp_ns=commit.timestamp_ns,
                estimator_lat_lon=estimator_lat_lon,
                estimator_enu=estimator_enu,
                display_lat_lon=estimator_lat_lon,
                display_enu=estimator_enu,
                snapped=False,
                confidence=0.0,
                matched_edge_id=cand.edge_id,
                distance_to_road_m=snap_dist,
                road_heading_rad=cand.edge_azimuth_rad,
                fallback_reason="DISCONNECTED_TRANSITION",
                candidate_count=cand_count,
                best_score=commit.best_score,
                second_best_score=commit.second_best_score,
                margin=commit.margin,
                result_schema=res_schema,
            )

        # Compute normalized confidence in [0.0, 1.0]
        sigma_d = max(self.emission_model.sigma_road_m, 2.0)
        c_dist = math.exp(-0.5 * (snap_dist / sigma_d) ** 2)
        c_margin = min(1.0, commit.margin / 3.0)
        confidence = float(np.clip(0.65 * c_dist + 0.35 * c_margin, 0.0, 1.0))

        # Check 5: Minimum confidence threshold
        if confidence < self.min_confidence:
            res_schema = MapMatchResult(
                snapped=False,
                confidence=confidence,
                matched_road_id=cand.edge_id,
                distance_to_road_m=snap_dist,
            )
            return MapMatchOutput(
                timestamp_ns=commit.timestamp_ns,
                estimator_lat_lon=estimator_lat_lon,
                estimator_enu=estimator_enu,
                display_lat_lon=estimator_lat_lon,
                display_enu=estimator_enu,
                snapped=False,
                confidence=confidence,
                matched_edge_id=cand.edge_id,
                distance_to_road_m=snap_dist,
                road_heading_rad=cand.edge_azimuth_rad,
                fallback_reason="LOW_CONFIDENCE",
                candidate_count=cand_count,
                best_score=commit.best_score,
                second_best_score=commit.second_best_score,
                margin=commit.margin,
                result_schema=res_schema,
            )

        # All safeguards passed: successful confident snap
        self._last_committed_edge = cand.edge_id
        snapped_enu = (
            float(cand.projected_point_enu[0]),
            float(cand.projected_point_enu[1]),
            float(estimator_enu[2]),
        )
        snapped_lat_lon = (float(cand.projected_lat_lon[0]), float(cand.projected_lat_lon[1]))

        res_schema = MapMatchResult(
            snapped=True,
            snapped_lat_lon=snapped_lat_lon,
            matched_road_id=cand.edge_id,
            confidence=confidence,
            heading_rad=cand.edge_azimuth_rad,
            distance_to_road_m=snap_dist,
        )

        return MapMatchOutput(
            timestamp_ns=commit.timestamp_ns,
            estimator_lat_lon=estimator_lat_lon,
            estimator_enu=estimator_enu,
            display_lat_lon=snapped_lat_lon,
            display_enu=snapped_enu,
            snapped=True,
            confidence=confidence,
            matched_edge_id=cand.edge_id,
            distance_to_road_m=snap_dist,
            road_heading_rad=cand.edge_azimuth_rad,
            fallback_reason=None,
            candidate_count=cand_count,
            best_score=commit.best_score,
            second_best_score=commit.second_best_score,
            margin=commit.margin,
            result_schema=res_schema,
        )

    def process_state(
        self,
        nav_state: Any,
        geo_ref: GeoReference,
        timestamp_ns: Optional[int] = None,
    ) -> Optional[MapMatchOutput]:
        """Ingest a live NavigationState or ESKFState and return a mature MapMatchOutput if available.

        Non-mutating: nav_state is never modified.
        """
        if hasattr(nav_state, "position_local"):
            # Canonical NavigationState
            t_ns = nav_state.timestamp_ns if timestamp_ns is None else timestamp_ns
            pos_enu = (float(nav_state.position_local[0]), float(nav_state.position_local[1]), float(nav_state.position_local[2]))
            cov_arr = np.asarray(nav_state.covariance, dtype=np.float64)
            cov_2x2 = cov_arr[0:2, 0:2]
            v_horiz = math.hypot(nav_state.velocity_local[0], nav_state.velocity_local[1])
            heading_rad = float(math.atan2(nav_state.velocity_local[0], nav_state.velocity_local[1])) % (2.0 * math.pi) if v_horiz > 1.5 else None
        elif hasattr(nav_state, "nominal"):
            # Internal ESKFState
            t_ns = timestamp_ns if timestamp_ns is not None else getattr(nav_state, "timestamp_ns", 0)
            nom = nav_state.nominal
            pos_enu = (float(nom.position_enu[0]), float(nom.position_enu[1]), float(nom.position_enu[2]))
            cov_arr = np.asarray(nav_state.covariance, dtype=np.float64)
            cov_2x2 = cov_arr[0:2, 0:2]
            v_horiz = math.hypot(nom.velocity_enu[0], nom.velocity_enu[1])
            heading_rad = float(math.atan2(nom.velocity_enu[0], nom.velocity_enu[1])) % (2.0 * math.pi) if v_horiz > 1.5 else None
        else:
            raise TypeError(f"Unsupported state object: {type(nav_state)}")

        delta_t_s = None
        if self._last_timestamp_ns is not None:
            delta_t_s = (t_ns - self._last_timestamp_ns) * 1e-9
        self._last_timestamp_ns = t_ns

        e2d = (pos_enu[0], pos_enu[1])

        # Convert estimator ENU to geodetic lat/lon
        lat_est, lon_est, _ = geo_ref.enu_to_geodetic(pos_enu[0], pos_enu[1], pos_enu[2])
        estimator_lat_lon = (float(lat_est), float(lon_est))

        # Query candidates in session ENU
        candidates = self.candidate_search.search_candidates(e2d)

        emissions: List[float] = []
        for cand in candidates:
            log_e = self.emission_model.compute_log_emission(
                candidate=cand,
                cov_enu_2x2=cov_2x2,
                vehicle_heading_rad=heading_rad,
                vehicle_speed_mps=v_horiz,
            )
            emissions.append(log_e)

        # Step causal fixed-lag Viterbi with effective timestamp t_ns
        commit = self.viterbi.step(
            timestamp_ns=t_ns,
            traj_pos_enu=e2d,
            candidates=candidates,
            log_emissions=emissions,
            delta_t_s=delta_t_s,
        )

        if commit is None:
            return None

        # Convert committed epoch's trajectory ENU to lat/lon
        comm_enu = (commit.traj_pos_enu[0], commit.traj_pos_enu[1], pos_enu[2])
        comm_lat, comm_lon, _ = geo_ref.enu_to_geodetic(comm_enu[0], comm_enu[1], comm_enu[2])
        comm_lat_lon = (float(comm_lat), float(comm_lon))

        return self._evaluate_commit(
            commit=commit,
            estimator_lat_lon=comm_lat_lon,
            estimator_enu=comm_enu,
        )

    def flush_remaining(self, geo_ref: GeoReference) -> List[MapMatchOutput]:
        """Flush remaining buffered epochs at end of run."""
        commits = self.viterbi.flush_remaining()
        outputs: List[MapMatchOutput] = []
        for commit in commits:
            comm_enu = (commit.traj_pos_enu[0], commit.traj_pos_enu[1], 0.0)
            comm_lat, comm_lon, _ = geo_ref.enu_to_geodetic(comm_enu[0], comm_enu[1], 0.0)
            comm_lat_lon = (float(comm_lat), float(comm_lon))
            out = self._evaluate_commit(
                commit=commit,
                estimator_lat_lon=comm_lat_lon,
                estimator_enu=comm_enu,
            )
            outputs.append(out)
        return outputs
