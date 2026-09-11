"""Map matching module for COMPASS (Phase 12).

Provides strictly downstream Hidden Markov Model (HMM) road matching
with fixed-lag online Viterbi and safe fallback.
"""

from navigation.mapmatch.candidates import CandidateSearch, RoadCandidate
from navigation.mapmatch.emission import EmissionModel
from navigation.mapmatch.matcher import MapMatchOutput, MapMatcher
from navigation.mapmatch.transition import TransitionModel
from navigation.mapmatch.viterbi import FixedLagViterbi, ViterbiCommit

__all__ = [
    "CandidateSearch",
    "RoadCandidate",
    "EmissionModel",
    "TransitionModel",
    "FixedLagViterbi",
    "ViterbiCommit",
    "MapMatcher",
    "MapMatchOutput",
]
