"""Discriminative magnitude-regressor subsystem for VeriTaS judge calibration.

The LLM judge predicts the integrity *direction* (intact / unknown / compromised)
reliably, but its *certainty magnitude* (certain vs. rather-certain) is miscalibrated.
This package trains a standalone gradient-boosted regressor that predicts the
*magnitude* ``|integrity.score| in [0, 1]`` from execution-trace features. At serving
time the judge owns the sign and the regressor owns the magnitude:

    final_score = sign(judge_direction) * regressor_magnitude

This continuous value is the deliverable; for diagnostics it is binned to a 7-class
label by the canonical VeriTaS rule (``mafc.eval.veritas`` ``THRESHOLDS_7``).

Modules
-------
- ``labels``      shared score<->direction helpers built on ``mafc.eval.veritas``.
- ``sampler``     stratified, boundary-weighted training-set sampler.
- ``features``    trace -> feature-table extractor (joins traces with claims).
- ``train``       gradient-boosted magnitude regressor + CV / learning curve.
- ``evaluate``    7/3-class + regression + calibration (ECE) reporting.
"""

from __future__ import annotations
