"""Algorithms selected independently from the policy and value configuration."""

from flow_rl.algorithms.awac import AWAC
from flow_rl.algorithms.base import Algorithm
from flow_rl.algorithms.bc import BC

__all__ = ["AWAC", "BC", "Algorithm"]
