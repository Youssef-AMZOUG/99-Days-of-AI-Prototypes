# Day 04 — Orthography & Grammar: Fast Spell Corrector

This folder contains a fast, local orthography (spelling) corrector that:
- fixes word typos,
- preserves punctuation and capitalization,
- outputs corrected text and confidence information.

**Files**
- `simple_spell.py` — main script (no heavy models).
- `examples.txt` — demo sentences.

**Install**
We recommend running inside your project environment:

```bash
pip install pyspellchecker textblob
python -m textblob.download_corpora
