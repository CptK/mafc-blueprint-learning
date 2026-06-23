"""Strategy-paper baseline.

A deliberately simple alternative to the structured blueprint pool: instead of
many routed blueprints with verification graphs, the fact-checker is handed a
single free-text playbook (``Strategy.md``) distilled from professional
fact-checks. The document is built by sequentially folding batches of fact-check
article analyses into a running document — one LLM call per batch, single pass
over the corpus by default (cost-driven), with optional multi-epoch passes and
per-epoch checkpointing so a run can be resumed or continued from a file.

The folding engine lives in :mod:`mafc.strategy.synthesizer`; checkpoint/state
serialization in :mod:`mafc.strategy.checkpoint`. The inference-time consumer — a
standalone fact-check agent driven solely by the document plus the web_search and
media tools — lives in :mod:`mafc.strategy.agent`.
"""

from mafc.single_file_strategy.agent import StrategyAgent
from mafc.single_file_strategy.synthesizer import StrategyFoldResult, StrategySynthesizer

__all__ = ["StrategySynthesizer", "StrategyFoldResult", "StrategyAgent"]
