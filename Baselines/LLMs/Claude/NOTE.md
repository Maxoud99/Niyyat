# Claude — not yet implemented

No Claude-based intent-attribution pipeline exists anywhere in the source
codebase as of this reorganization (2026-06-25). This folder is a
placeholder so the LLM baseline lineup matches the intended model roster
(Gemini, Llama, Qwen, Deepseek-R1, Mixtral, GPT, Claude).

To add it, mirror the structure used by the other models, e.g. by adapting:
- Baselines/LLMs/GPT/Adult/code/intent_attribution_pipeline-gpt.py (Adult, row/cell-level)
- Baselines/LLMs/Gemini/TwitterBot/code/*.py (bare-min / few-shot / info prompt variants)
- Baselines/LLMs/Gemini/TabFact/code/*.py
