# context-forge-py

[![PyPI](https://img.shields.io/pypi/v/context-forge-py.svg)](https://pypi.org/project/context-forge-py/)
[![Python](https://img.shields.io/pypi/pyversions/context-forge-py.svg)](https://pypi.org/project/context-forge-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Context engineering toolkit for RAG and agent prompts.** Chunk documents, score relevance (BM25), apply diversity-aware packing (MMR), flag prompt-injection risk, and emit citation-ready context blocks that fit a token budget. Zero runtime dependencies.

Python port of [@mukundakatta/context-forge](https://github.com/MukundaKatta/context-forge).

## Install

```bash
pip install context-forge-py
```

## Quick start

```python
from context_forge import forge

result = forge(
    chunks=[
        {"id": "doc1#0", "text": "Pluto was reclassified as a dwarf planet in 2006.", "source": "wiki/pluto"},
        {"id": "doc2#0", "text": "Mars is the fourth planet from the Sun.", "source": "wiki/mars"},
    ],
    query="What happened to Pluto?",
    budget=200,
)

result.blocks      # list of kept chunks (ranked, diversified, packed)
result.used_tokens # int -- tokens used by kept blocks
result.dropped     # list of {id, reason}
result.risks       # injection findings (e.g. ignore_instructions, exfil_curl)
result.citations   # {block_id: {source, span: [start, end]}}
```

You can also use the lower-level pipeline and the document-level entry:

```python
from context_forge import (
    pack_context,    # full pipeline: chunk -> score -> diversify -> risk-scan -> pack
    chunk_document,  # split a document into token-bounded chunks
    score_chunks,    # BM25 score against a query
    diversify,       # MMR re-rank for diversity
    pack_to_budget,  # greedy budget packer
    scan_injection,  # prompt-injection risk scan
    estimate_tokens, # ceil(len/4) heuristic
)
```

## API

### `forge(chunks, query, budget=1200, *, lambda_=0.7, per_chunk_min=20)`

Compose ranking + diversification + injection-scanning + budget-packing on a list of pre-chunked context. Returns a `ForgedContext` dataclass with `blocks`, `used_tokens`, `dropped`, `risks`, and `citations`.

### `pack_context(query, documents, budget_tokens=1200, max_tokens=200, overlap_tokens=20, lambda_=0.7, per_chunk_min=20)`

Full pipeline starting from raw documents (`{"id": ..., "text": ..., "source": ...}`). Chunks each document, then runs the same score/diversify/scan/pack steps as `forge`.

### Risk classes

`scan_injection(text)` returns findings with `kind`, `severity`, `snippet`, and `index`. Detected kinds include `ignore_instructions`, `system_prefix`, `you_are_now`, `role_tag`, `exfil_curl`, `exfil_wget`, `exfil_base64`, `zero_width_char`, and `suspicious_url`.

## License

MIT
