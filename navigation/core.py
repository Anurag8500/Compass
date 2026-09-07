"""NavigationCore: Central Python reference implementation and behavioral oracle for COMPASS.

This module will coordinate sensor ingestion, classical strapdown INS propagation,
ESKF state fusion, ML pseudo-measurements (VelocityNet & BiasNet), NHC kinematics,
and mode FSM transitions.

Implementation begins in Phase 1+.
"""
