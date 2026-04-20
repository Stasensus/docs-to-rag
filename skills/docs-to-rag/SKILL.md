---
name: docs-to-rag
description: Convert a folder of documents (PDFs, DOCX, DOC, scans, images, TXT/MD) into a local searchable RAG database with semantic search. Runs fully offline with GPU-accelerated OCR. Use when the user has a large pile of mixed documents and needs to search them by meaning or point specific questions at them without reading everything. Output lives in .docs-to-rag/ alongside the source folder.
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
---

# /docs-to-rag — Local RAG pipeline for mixed document corpora

## When to use

Invoke this skill when the user has:
- a folder with many documents (>20 files or >100 pages)
- mixed formats (PDF with text, PDF scans, DOCX, DOC, images, TXT)
- a need to search them by meaning ("what did we decide about X?") or point questions at them
- a preference for offline processing (confidential / legal / medical / personal data)

Do **not** invoke for:
- a single short document — just read it directly
- content on the web — use WebFetch
- corpora with >500k pages — this skill scales to ~50k chunks; beyond that escalate to FAISS/Chroma

## What this skill produces

For a given `<path>`, builds a `<path>/.docs-to-rag/` directory containing:
```
.docs-to-rag/
├── ocr/            # one .txt per page, uniform naming: {type}_{slug}_{page:03d}.txt
├── index/
│   ├── emb.npy     # (N, 384) float32 embeddings
│   └── meta.json   # chunk metadata
├── query.py        # CLI: python query.py "question" -k 10
├── grep.py         # CLI: python grep.py "exact word"
├── manifest.json   # file hashes + timestamps for --update
└── README.md       # how to use the output
```

Hidden folder `.docs-to-rag/` keeps the original corpus clean.

## Commands

| Command | What it does |
|---|---|
| `/docs-to-rag <path>` | Full pipeline (detect → convert → OCR → index) |
| `/docs-to-rag <path> --inspect` | Just `detect` — show file inventory, no processing |
| `/docs-to-rag <path> --no-ocr` | Skip OCR, only native-text PDFs / DOCX / TXT |
| `/docs-to-rag <path> --lang ru,en` | OCR languages (default: `ru,en`) |
| `/docs-to-rag <path> --dpi 200` | PDF render DPI for OCR (default 180) |
| `/docs-to-rag <path> --cpu` | Force CPU OCR even if GPU is available |
| `/docs-to-rag <path> --update` | Incremental — re-process only new/changed files |
| `/docs-to-rag <path> --graph` | Chain into `/graphify` after indexing |
| `/docs-to-rag query <path> "question"` | Run a semantic query against a built index |

## What You Must Do When Invoked

If no path was given, ask the user once for a folder path. Then follow the steps in order.

### Step 1 — Locate pipeline.py

The plugin ships `pipeline.py` next to this SKILL.md. Resolve the path via:
```bash
PIPELINE="${CLAUDE_PLUGIN_ROOT}/skills/docs-to-rag/pipeline.py"
```
If `CLAUDE_PLUGIN_ROOT` is unset, fall back to:
```bash
PIPELINE="${HOME}/.claude/plugins/docs-to-rag/skills/docs-to-rag/pipeline.py"
```

### Step 2 — Dependency check

Before any processing, verify dependencies:
```bash
python "$PIPELINE" check
```
If missing, the script prints a single `pip install ...` command. Run it once. Do not install packages the user did not ask for beyond this baseline.

### Step 3 — Detect

```bash
python "$PIPELINE" detect <path>
```
Reads the folder, categorises files (pdf-text, pdf-scan, docx, doc, image, text, skip). Writes `<path>/.docs-to-rag/detect.json`.

Present a clean summary to the user:
```
Corpus: N files · ~K pages
  pdf-text:   A files (C pages)
  pdf-scan:   B files (D pages)    ← OCR needed
  docx/doc:   E files
  images:     F files              ← OCR needed
  text:       G files
Skipped:      H files (unsupported)
```

### Step 4 — Warn if OCR workload is large without GPU

If `detect.json` shows OCR is needed and `python "$PIPELINE" gpu-check` returns no GPU, tell the user:

> "OCR on CPU for N pages will take ~M hours. Options:
> 1. Install CUDA/PyTorch-GPU and re-run (fastest).
> 2. Continue on CPU (set and forget — skill will run in background).
> 3. Use `--no-ocr` to process only native-text PDFs and docs (skips scans/images)."

Wait for the user to choose. Do not silently burn hours on CPU.

### Step 5 — Convert + OCR (background-friendly)

```bash
python "$PIPELINE" build <path> --lang ru,en --dpi 180
```
Steps internally:
1. Convert native-text PDFs → `ocr/pdf_*.txt` per page
2. Convert DOCX/DOC → one file per document
3. Render scan-PDFs and images → OCR via EasyOCR (GPU if available)
4. Write all results to `<path>/.docs-to-rag/ocr/`

For corpora >300 pages, run in background with `run_in_background: true` and poll progress via `<path>/.docs-to-rag/build.log`. Report ETA to the user periodically (~every 200 pages).

### Step 6 — Index

```bash
python "$PIPELINE" index <path>
```
Embeds all chunks with `intfloat/multilingual-e5-small`, saves `index/emb.npy` + `meta.json`. Takes ~1 min per 1000 pages on CPU.

### Step 7 — Install CLIs + write README

`pipeline.py index` auto-writes `query.py`, `grep.py`, and `README.md` inside `.docs-to-rag/`. The user runs them directly with the system Python.

### Step 8 — (Optional) Graph

If `--graph` was given, after indexing:
1. Group `.txt` files into `.md` chunks of ~30 pages each in `.docs-to-rag/graph_input/`.
2. Invoke `/graphify` on that folder.
3. The graph lives in `.docs-to-rag/graphify-out/` — point the user to `graph.html`.

### Step 9 — Final report

Tell the user:
```
Done. Index built in <path>/.docs-to-rag/

  Semantic search:   python .docs-to-rag/query.py "your question" -k 10
  Exact-word grep:   python .docs-to-rag/grep.py "word"
  Source pages:      .docs-to-rag/ocr/
```
Pick one example question relevant to the user's domain and show the command with it. Offer to run the first query.

## Gotchas and workarounds

### A. EasyOCR + Cyrillic path on Windows
EasyOCR's `readtext(path)` fails silently on non-ASCII paths on Windows. `pipeline.py` loads images via `PIL.Image.open()` + `np.array()` and passes the ndarray directly. Never pass raw paths to `readtext` — always go through ndarray.

### B. ChromaDB 1.5.8 HNSW crash on Windows
Intermittent "Error loading hnsw index" after successful insert. The skill uses NumPy cosine instead (full scan on (N, 384) is ~20 ms for N=50k). Do not swap back to Chroma without testing.

### C. PyTorch + CUDA on fresh install
Default `pip install torch` gets the CPU build. To enable GPU:
```bash
pip uninstall -y torch torchvision
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```
For driver CUDA 13.x use cu121; for 12.x also cu121 works (backward compat).

### D. .doc (legacy Word, not .docx)
Needs conversion. On Windows, use Word COM via pywin32. On Linux/Mac, shell out to `soffice --headless --convert-to txt`. `pipeline.py` tries both and reports a clear error if neither is available.

### E. Text layer detection
Some PDFs report a text layer that's just page numbers. `pipeline.py` requires >200 chars across the first 5 pages before trusting the text layer; otherwise it renders and OCRs.

### F. Encoding of legacy Word output
Word COM saves .txt as cp1251 on RU locale. The script tries `utf-8, utf-16, cp1251` in that order and re-encodes to UTF-8.

## What this skill does NOT do

- **No cloud uploads.** All processing is local. If the user insists on cloud OCR for speed, route them to a separate tool.
- **No knowledge graph.** Delegated to `/graphify` via `--graph`.
- **No domain-specific prompting.** This skill prepares the corpus; prompt engineering is up to the user or another skill on top.
- **No watch mode.** `--update` is on-demand only.

## Honest limits

- OCR quality depends on source scan DPI. For handwritten or low-contrast scans, expect 70-90% accuracy. Add `--dpi 300` and accept 2x time.
- `multilingual-e5-small` handles ~100 languages well but peaks in EN/RU/ZH/ES/FR/DE/JP. For rare languages consider `--model intfloat/multilingual-e5-base`.
- Chunk size is 1200 chars. Legal/academic prose: fine. Code/tables: can lose structure — grep remains the fallback.
