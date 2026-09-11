import math
import numpy as np
from navigation.mapmatch.candidates import RoadCandidate
from navigation.mapmatch.emission import EmissionModel
from navigation.mapmatch.transition import TransitionModel
from navigation.mapmatch.viterbi import FixedLagViterbi

def test_viterbi_deterministic_routing():
    """
    Test the sliding window Viterbi decoder against a deterministic synthetic graph.
    The graph consists of two parallel roads, Road A (y=0) and Road B (y=10).
    The vehicle moves along Road A, but the measurements are slightly noisy.
    """
    transition = TransitionModel(beta_m=5.0)
    viterbi = FixedLagViterbi(transition_model=transition, lag_epochs=3)
    emission = EmissionModel(sigma_road_m=2.0)

    # Road A is correct (y=0), Road B is incorrect parallel (y=10)
    traj_x = [0.0, 5.0, 10.0, 15.0, 20.0]
    noisy_y = [1.0, -1.0, 2.0, 1.0, 0.0]  # Noisy but closer to 0 than 10

    results = []
    
    for i in range(len(traj_x)):
        x = traj_x[i]
        y = noisy_y[i]
        
        # Candidate 1: Road A
        cand_a = RoadCandidate(
            edge_id="RoadA",
            projected_point_enu=(x, 0.0, 0.0),
            distance_to_road_m=abs(y - 0.0),
            edge_azimuth_rad=math.pi/2, # East
            projected_lat_lon=(0.0, 0.0),
            distance_along_edge_m=x,
        )
        
        # Candidate 2: Road B
        cand_b = RoadCandidate(
            edge_id="RoadB",
            projected_point_enu=(x, 10.0, 0.0),
            distance_to_road_m=abs(y - 10.0),
            edge_azimuth_rad=math.pi/2, # East
            projected_lat_lon=(0.0, 0.0),
            distance_along_edge_m=x,
        )
        
        cands = [cand_a, cand_b]
        
        emissions = [
            emission.compute_log_emission(c) for c in cands
        ]
        
        commit = viterbi.step(
            timestamp_ns=i * 1000000000,
            traj_pos_enu=(x, y),
            candidates=cands,
            log_emissions=emissions,
            delta_t_s=1.0,
        )
        if commit is not None:
            results.append(commit)
            
    # Flush
    results.extend(viterbi.flush_remaining())
    
    assert len(results) == 5
    for commit in results:
        assert commit.candidate is not None
        assert commit.candidate.edge_id == "RoadA"
        assert commit.is_mature

def test_viterbi_no_candidates_fallback():
    transition = TransitionModel(beta_m=5.0)
    viterbi = FixedLagViterbi(transition_model=transition, lag_epochs=1)
    
    commit = viterbi.step(0, (0.0, 0.0), [], [], 1.0)
    assert commit is None
    
    # Step 2: also empty, but forces lag window to commit step 1
    commit = viterbi.step(1000000000, (1.0, 0.0), [], [], 1.0)
    assert commit is not None
    assert commit.candidate is None
    assert commit.fallback_reason == "NO_CANDIDATES"
    
    flushed = viterbi.flush_remaining()
    assert len(flushed) == 1
    assert flushed[0].candidate is None
