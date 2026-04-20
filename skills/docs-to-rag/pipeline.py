#!/usr/bin/env python3
"""docs-to-rag pipeline: detect → convert → OCR → index → query.

Subcommands:
  check               verify dependencies
  gpu-check           return 0 if GPU available, 1 otherwise
  detect <path>       inventory files, write .docs-to-rag/detect.json
  build <path>        convert + OCR → .docs-to-rag/ocr/
  index <path>        embed chunks → .docs-to-rag/index/
  query <path> <q>    semantic search
  grep <path> <w>     literal grep over ocr/

Usage:
  python pipeline.py detect "C:/docs"
  python pipeline.py build "C:/docs" --lang ru,en --dpi 180
  python pipeline.py index "C:/docs"
  python pipeline.py query "C:/docs" "when was the warning delivered"
"""
import argparse, hashlib, io, json, os, re, shutil, sys, time, traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ─── paths ────────────────────────────────────────────────────────────
def outdir(corpus: Path) -> Path:
    return corpus / '.docs-to-rag'

def ocrdir(corpus: Path) -> Path:
    return outdir(corpus) / 'ocr'

def indexdir(corpus: Path) -> Path:
    return outdir(corpus) / 'index'


# ─── helpers ──────────────────────────────────────────────────────────
def slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r'[^\w\d]+', '_', s, flags=re.UNICODE)[:maxlen]
    return s.strip('_').lower() or 'doc'

def log(msg: str, logfile: Path = None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if logfile:
        try:
            logfile.parent.mkdir(parents=True, exist_ok=True)
            with open(logfile, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception:
            pass

def file_hash(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ─── check ────────────────────────────────────────────────────────────
REQUIRED = {
    'pymupdf':              'fitz',
    'easyocr':              'easyocr',
    'sentence-transformers':'sentence_transformers',
    'numpy':                'numpy',
    'Pillow':               'PIL',
}
def cmd_check(args):
    missing = []
    for pkg, mod in REQUIRED.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print('Missing:', ', '.join(missing))
        print('Install with:')
        print('  pip install ' + ' '.join(missing))
        return 1
    print('OK. All required deps present.')
    return 0


# ─── gpu-check ────────────────────────────────────────────────────────
def cmd_gpu_check(args):
    try:
        import torch
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            return 0
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print('GPU: MPS (Apple Silicon)')
            return 0
    except Exception:
        pass
    print('No GPU available (CPU only)')
    return 1


# ─── detect ───────────────────────────────────────────────────────────
PDF_EXT = {'.pdf'}
DOCX_EXT = {'.docx'}
DOC_EXT = {'.doc'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
TEXT_EXT = {'.txt', '.md'}
SKIP_DIRS = {'.git', 'node_modules', '.docs-to-rag', '__pycache__', '.venv', 'venv'}

def walk_files(root: Path):
    for dp, dnames, files in os.walk(root):
        dnames[:] = [d for d in dnames if d not in SKIP_DIRS and not d.startswith('.')]
        for f in files:
            yield Path(dp) / f

def detect_pdf_kind(p: Path):
    """Return ('pdf-text'|'pdf-scan', page_count) — quick probe of first 5 pages."""
    import fitz
    try:
        d = fitz.open(str(p))
    except Exception:
        return None, 0
    n = d.page_count
    sample = ''
    for i in range(min(n, 5)):
        sample += d[i].get_text()
    d.close()
    kind = 'pdf-text' if len(sample.strip()) > 200 else 'pdf-scan'
    return kind, n

def cmd_detect(args):
    corpus = Path(args.path).resolve()
    if not corpus.is_dir():
        print(f'ERROR: {corpus} is not a directory')
        return 1
    result = {
        'path': str(corpus),
        'files': {'pdf-text':[], 'pdf-scan':[], 'docx':[], 'doc':[], 'image':[], 'text':[], 'skip':[]},
        'pages': {'pdf-text':0, 'pdf-scan':0, 'docx':0, 'doc':0, 'image':0, 'text':0},
        'total_files': 0,
    }
    for p in walk_files(corpus):
        # skip our own output
        try:
            if outdir(corpus) in p.parents:
                continue
        except Exception:
            pass
        ext = p.suffix.lower()
        rel = str(p.relative_to(corpus))
        if ext in PDF_EXT:
            kind, n = detect_pdf_kind(p)
            if kind is None:
                result['files']['skip'].append(rel); continue
            result['files'][kind].append(rel)
            result['pages'][kind] += n
        elif ext in DOCX_EXT:
            result['files']['docx'].append(rel); result['pages']['docx'] += 1
        elif ext in DOC_EXT:
            result['files']['doc'].append(rel); result['pages']['doc'] += 1
        elif ext in IMAGE_EXT:
            result['files']['image'].append(rel); result['pages']['image'] += 1
        elif ext in TEXT_EXT:
            result['files']['text'].append(rel); result['pages']['text'] += 1
        else:
            result['files']['skip'].append(rel)
    result['total_files'] = sum(len(v) for v in result['files'].values())
    result['total_pages'] = sum(result['pages'].values())
    result['ocr_pages']   = result['pages']['pdf-scan'] + result['pages']['image']
    out = outdir(corpus); out.mkdir(parents=True, exist_ok=True)
    (out / 'detect.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    # human summary
    print(f'Corpus: {result["total_files"]} files · ~{result["total_pages"]} pages')
    for k in ('pdf-text','pdf-scan','docx','doc','image','text'):
        n = len(result['files'][k]); pg = result['pages'][k]
        if n: print(f'  {k:9s} {n:4d} files  ({pg} pages)')
    if result['files']['skip']:
        print(f'  skipped:  {len(result["files"]["skip"])} files')
    if result['ocr_pages']:
        print(f'  OCR needed for {result["ocr_pages"]} pages')
    return 0


# ─── build (convert + OCR) ────────────────────────────────────────────
def _iter_pdf_pages(pdf_path: Path, out_ocr: Path, slug_prefix: str, render_queue: list, dpi: int):
    """Write native-text pages; push scan pages into render_queue."""
    import fitz
    try:
        d = fitz.open(str(pdf_path))
    except Exception as e:
        log(f'ERROR open pdf {pdf_path.name}: {e}'); return 0
    # probe: text or scan
    sample = ''
    for i in range(min(d.page_count, 5)):
        sample += d[i].get_text()
    has_text = len(sample.strip()) > 200
    written = 0
    for i in range(d.page_count):
        outp = out_ocr / f'{slug_prefix}_{i+1:03d}.txt'
        if outp.exists() and outp.stat().st_size > 0:
            continue
        if has_text:
            txt = d[i].get_text().strip()
            if txt:
                outp.write_text(txt, encoding='utf-8')
                written += 1
        else:
            pix = d[i].get_pixmap(dpi=dpi)
            png_path = out_ocr.parent / 'render' / f'{slug_prefix}_{i+1:03d}.png'
            png_path.parent.mkdir(parents=True, exist_ok=True)
            if not png_path.exists():
                pix.save(str(png_path))
            render_queue.append((str(png_path), str(outp)))
    d.close()
    return written

def _convert_docx(docx_path: Path, out_ocr: Path, slug_prefix: str):
    """DOCX → single .txt (all paragraphs)."""
    import zipfile
    try:
        with zipfile.ZipFile(str(docx_path)) as z:
            xml = z.read('word/document.xml').decode('utf-8', errors='replace')
    except Exception as e:
        log(f'ERROR docx {docx_path.name}: {e}'); return 0
    txt = re.sub(r'</w:p>', '\n', xml)
    txt = re.sub(r'<[^>]+>', '', txt)
    txt = re.sub(r'\n{3,}', '\n\n', txt).strip()
    if not txt: return 0
    outp = out_ocr / f'{slug_prefix}_001.txt'
    outp.write_text(txt, encoding='utf-8')
    return 1

def _convert_doc(doc_path: Path, out_ocr: Path, slug_prefix: str):
    """Legacy .doc via Word COM (Win) or soffice (posix)."""
    outp = out_ocr / f'{slug_prefix}_001.txt'
    if outp.exists(): return 0
    txt = None
    # Try Word COM on Windows
    if sys.platform.startswith('win'):
        try:
            import win32com.client
            word = win32com.client.Dispatch('Word.Application')
            word.Visible = False
            doc = word.Documents.Open(str(doc_path), False, True)
            tmp = out_ocr.parent / f'_tmp_{slug_prefix}.txt'
            doc.SaveAs(str(tmp), 2)   # wdFormatText
            doc.Close(False); word.Quit()
            raw = tmp.read_bytes(); tmp.unlink(missing_ok=True)
            for enc in ('utf-8','utf-16','cp1251','latin-1'):
                try:
                    txt = raw.decode(enc); break
                except Exception:
                    pass
        except Exception as e:
            log(f'Word COM failed on {doc_path.name}: {e}')
    # Fallback: soffice
    if txt is None:
        try:
            import subprocess, tempfile
            with tempfile.TemporaryDirectory() as td:
                subprocess.run(['soffice','--headless','--convert-to','txt','--outdir',td,str(doc_path)],
                               check=True, capture_output=True, timeout=120)
                converted = Path(td) / (doc_path.stem + '.txt')
                if converted.exists():
                    txt = converted.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            log(f'soffice failed on {doc_path.name}: {e}')
    if txt is None:
        log(f'SKIP {doc_path.name}: neither Word COM nor soffice available')
        return 0
    outp.write_text(txt.strip(), encoding='utf-8')
    return 1

def _convert_text(text_path: Path, out_ocr: Path, slug_prefix: str):
    outp = out_ocr / f'{slug_prefix}_001.txt'
    if outp.exists(): return 0
    raw = text_path.read_bytes()
    for enc in ('utf-8','utf-16','cp1251','latin-1'):
        try:
            txt = raw.decode(enc); break
        except Exception:
            txt = None
    if not txt: return 0
    outp.write_text(txt.strip(), encoding='utf-8')
    return 1

def _queue_image(img_path: Path, out_ocr: Path, slug_prefix: str, render_queue: list):
    outp = out_ocr / f'{slug_prefix}_001.txt'
    if outp.exists(): return
    # Copy to render dir so paths are homogeneous
    render_dir = out_ocr.parent / 'render'
    render_dir.mkdir(parents=True, exist_ok=True)
    dst = render_dir / f'{slug_prefix}_001{img_path.suffix.lower()}'
    if not dst.exists():
        shutil.copy(str(img_path), str(dst))
    render_queue.append((str(dst), str(outp)))

def _ocr_run(queue, langs, gpu, logfile):
    if not queue:
        return 0
    # lazy import
    try:
        import easyocr
        import numpy as np
        from PIL import Image
    except ImportError as e:
        log(f'OCR dependencies missing: {e}', logfile)
        return 0
    log(f'Loading EasyOCR (lang={langs}, gpu={gpu})...', logfile)
    reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
    log('EasyOCR loaded', logfile)
    t0 = time.time(); done = 0
    for img_path, out_path in queue:
        if Path(out_path).exists() and Path(out_path).stat().st_size > 0:
            continue
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            res = reader.readtext(img, detail=0, paragraph=True)
            Path(out_path).write_text('\n'.join(res), encoding='utf-8')
            done += 1
            if done % 20 == 0:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(queue) - done) / rate if rate else 0
                log(f'OCR {done}/{len(queue)} rate={rate:.2f}p/s eta={eta/60:.1f}min', logfile)
        except Exception as e:
            log(f'OCR ERR {img_path}: {e}', logfile)
    log(f'OCR finished: {done}/{len(queue)}', logfile)
    return done

def cmd_build(args):
    corpus = Path(args.path).resolve()
    det_json = outdir(corpus) / 'detect.json'
    if not det_json.exists():
        print('Run detect first')
        return 1
    det = json.loads(det_json.read_text(encoding='utf-8'))
    out_ocr = ocrdir(corpus); out_ocr.mkdir(parents=True, exist_ok=True)
    logfile = outdir(corpus) / 'build.log'

    langs = args.lang.split(',') if args.lang else ['ru','en']
    gpu_ok = (cmd_gpu_check(argparse.Namespace()) == 0) if not args.cpu else False

    render_queue = []
    converted_text = 0

    # Native-text conversions first (cheap)
    for rel in det['files']['pdf-text']:
        p = corpus / rel
        slug = 'pdf_' + slugify(p.stem)
        converted_text += _iter_pdf_pages(p, out_ocr, slug, render_queue, args.dpi)

    # Scan PDFs — render to PNG, queue for OCR
    for rel in det['files']['pdf-scan']:
        p = corpus / rel
        slug = 'scan_' + slugify(p.stem)
        _iter_pdf_pages(p, out_ocr, slug, render_queue, args.dpi)

    for rel in det['files']['docx']:
        p = corpus / rel
        slug = 'docx_' + slugify(p.stem)
        converted_text += _convert_docx(p, out_ocr, slug)

    for rel in det['files']['doc']:
        p = corpus / rel
        slug = 'doc_' + slugify(p.stem)
        converted_text += _convert_doc(p, out_ocr, slug)

    for rel in det['files']['text']:
        p = corpus / rel
        slug = 'txt_' + slugify(p.stem)
        converted_text += _convert_text(p, out_ocr, slug)

    for rel in det['files']['image']:
        p = corpus / rel
        slug = 'img_' + slugify(p.stem)
        _queue_image(p, out_ocr, slug, render_queue)

    log(f'Text-extracted: {converted_text}. OCR queue: {len(render_queue)}', logfile)

    if args.no_ocr:
        log('Skipping OCR (--no-ocr)', logfile)
    elif render_queue:
        _ocr_run(render_queue, langs, gpu_ok, logfile)

    # Update manifest
    manifest = {'files': {}, 'built_at': time.time()}
    for bucket in ('pdf-text','pdf-scan','docx','doc','image','text'):
        for rel in det['files'][bucket]:
            p = corpus / rel
            try:
                manifest['files'][rel] = {
                    'hash': file_hash(p),
                    'mtime': p.stat().st_mtime,
                    'kind': bucket,
                }
            except Exception:
                pass
    (outdir(corpus) / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    log('Build complete', logfile)
    return 0


# ─── index ────────────────────────────────────────────────────────────
def _chunk(text, max_chars=1200):
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()] or [text]
    chunks, buf = [], ''
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + '\n' + p).strip()
        else:
            if buf: chunks.append(buf)
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                buf = ''
            else:
                buf = p
    if buf: chunks.append(buf)
    return chunks

def cmd_index(args):
    corpus = Path(args.path).resolve()
    ocr = ocrdir(corpus)
    if not ocr.is_dir():
        print('No ocr/ directory. Run build first.')
        return 1
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model_name = args.model or 'intfloat/multilingual-e5-small'
    print(f'Loading model {model_name} ...')
    model = SentenceTransformer(model_name)

    texts, metas = [], []
    for fn in sorted(os.listdir(ocr)):
        if not fn.endswith('.txt'): continue
        m = re.match(r'([a-z]+)_(.+)_(\d+)\.txt', fn)
        if not m:
            m2 = re.match(r'([a-z]+)_(\d+)\.txt', fn)
            if not m2: continue
            kind, sub, page = m2.group(1), '', int(m2.group(2))
        else:
            kind, sub, page = m.group(1), m.group(2), int(m.group(3))
        text = (ocr / fn).read_text(encoding='utf-8').strip()
        if not text: continue
        for i, ch in enumerate(_chunk(text)):
            texts.append('passage: ' + ch)
            metas.append({
                'kind': kind,
                'source': sub or kind,
                'page': page,
                'chunk': i,
                'text': ch,
                'file': fn,
            })
    if not texts:
        print('No text to index.')
        return 1
    print(f'Encoding {len(texts)} chunks...')
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = emb.astype('float32')

    idx = indexdir(corpus); idx.mkdir(parents=True, exist_ok=True)
    np.save(idx / 'emb.npy', emb)
    (idx / 'meta.json').write_text(json.dumps(metas, ensure_ascii=False), encoding='utf-8')
    (idx / 'model.txt').write_text(model_name, encoding='utf-8')
    print(f'Saved: {emb.shape} → {idx}')

    _write_clis(corpus, model_name)
    return 0

def _write_clis(corpus: Path, model_name: str):
    out = outdir(corpus)
    query_py = f'''#!/usr/bin/env python3
"""Semantic search over .docs-to-rag/index/. Usage: python query.py "вопрос" -k 10 [--full]"""
import argparse, json, os, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np
from sentence_transformers import SentenceTransformer

IDX = Path(__file__).parent / 'index'
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer({model_name!r})
    return _model

def search(query, k=10):
    emb = np.load(IDX / 'emb.npy')
    metas = json.load(open(IDX / 'meta.json', encoding='utf-8'))
    q = get_model().encode(['query: ' + query], normalize_embeddings=True)[0]
    sims = emb @ q
    top = np.argsort(-sims)[:k]
    return [(float(sims[i]), metas[i]) for i in top]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('query', nargs='+')
    ap.add_argument('-k', type=int, default=10)
    ap.add_argument('--full', action='store_true')
    a = ap.parse_args()
    q = ' '.join(a.query)
    print(f'=== {{q}} ===\\n')
    for score, m in search(q, a.k):
        src = m.get('source') or m.get('kind', '')
        print(f'[{{score:.3f}}] {{src}} стр.{{m.get("page","?")}}  ({{m.get("file","")}})')
        txt = m["text"] if a.full else m["text"][:400]
        print(txt.replace("\\n"," | "))
        print("---")

if __name__ == "__main__": main()
'''
    (out / 'query.py').write_text(query_py, encoding='utf-8')

    grep_py = '''#!/usr/bin/env python3
"""Literal-word grep over .docs-to-rag/ocr/. Usage: python grep.py "word" [-C 2]"""
import argparse, os, re, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass

OCR = Path(__file__).parent / 'ocr'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('word', nargs='+')
    ap.add_argument('-C', type=int, default=1, help='context lines')
    ap.add_argument('-i', action='store_true', help='ignore case')
    a = ap.parse_args()
    pat = re.compile(re.escape(' '.join(a.word)), re.IGNORECASE if a.i else 0)
    hits = 0
    for fn in sorted(os.listdir(OCR)):
        if not fn.endswith('.txt'): continue
        text = (OCR / fn).read_text(encoding='utf-8', errors='replace')
        lines = text.split('\\n')
        for i, ln in enumerate(lines):
            if pat.search(ln):
                hits += 1
                start, end = max(0, i-a.C), min(len(lines), i+a.C+1)
                print(f'--- {fn}:{i+1}')
                for j in range(start, end):
                    mark = '>' if j == i else ' '
                    print(f'{mark} {lines[j]}')
    print(f'\\n{hits} matches')

if __name__ == "__main__": main()
'''
    (out / 'grep.py').write_text(grep_py, encoding='utf-8')

    readme = f'''# .docs-to-rag/ — local RAG index

Built by the `docs-to-rag` skill.

## Usage

**Semantic search** (by meaning):
```
python query.py "your question" -k 10
```

**Exact-word search** (like grep):
```
python grep.py "literal word"
```

## Files

- `ocr/`         — plain text per page (`{{kind}}_{{slug}}_{{page:03d}}.txt`)
- `index/emb.npy`— chunk embeddings (float32, cosine-normalized)
- `index/meta.json` — chunk metadata
- `manifest.json`— file hashes for incremental rebuild
- `build.log`    — OCR progress log
- `render/`      — rendered PNG pages (can be deleted after OCR)

## Rebuild

```
# from the corpus parent folder
python /path/to/pipeline.py build <corpus>   --update
python /path/to/pipeline.py index <corpus>
```

## Model

Embeddings: `{model_name}` (384-dim, multilingual).
To use a larger/different model, re-run index with --model.
'''
    (out / 'README.md').write_text(readme, encoding='utf-8')


# ─── query ────────────────────────────────────────────────────────────
def cmd_query(args):
    corpus = Path(args.path).resolve()
    idx = indexdir(corpus)
    if not (idx / 'emb.npy').exists():
        print('No index. Run: pipeline.py index <path>')
        return 1
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model_name = (idx / 'model.txt').read_text(encoding='utf-8').strip()
    model = SentenceTransformer(model_name)
    emb = np.load(idx / 'emb.npy')
    metas = json.loads((idx / 'meta.json').read_text(encoding='utf-8'))
    q = model.encode(['query: ' + args.query], normalize_embeddings=True)[0]
    sims = emb @ q
    top = np.argsort(-sims)[:args.k]
    print(f'=== {args.query} ===\n')
    for i in top:
        m = metas[i]; s = float(sims[i])
        src = m.get('source') or m.get('kind','')
        print(f'[{s:.3f}] {src} стр.{m.get("page","?")} ({m.get("file","")})')
        t = m['text'] if args.full else m['text'][:400]
        print(t.replace('\n',' | '))
        print('---')
    return 0


# ─── grep ─────────────────────────────────────────────────────────────
def cmd_grep(args):
    corpus = Path(args.path).resolve()
    ocr = ocrdir(corpus)
    if not ocr.is_dir(): print('No ocr/'); return 1
    pat = re.compile(re.escape(args.word), re.IGNORECASE if args.i else 0)
    hits = 0
    for fn in sorted(os.listdir(ocr)):
        if not fn.endswith('.txt'): continue
        text = (ocr / fn).read_text(encoding='utf-8', errors='replace')
        for i, ln in enumerate(text.split('\n'), 1):
            if pat.search(ln):
                hits += 1
                print(f'{fn}:{i}: {ln}')
    print(f'\n{hits} matches')
    return 0


# ─── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(prog='docs-to-rag')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('check').set_defaults(func=cmd_check)
    sub.add_parser('gpu-check').set_defaults(func=cmd_gpu_check)

    p = sub.add_parser('detect'); p.add_argument('path'); p.set_defaults(func=cmd_detect)

    p = sub.add_parser('build')
    p.add_argument('path'); p.add_argument('--lang', default='ru,en')
    p.add_argument('--dpi', type=int, default=180); p.add_argument('--no-ocr', action='store_true')
    p.add_argument('--cpu', action='store_true'); p.set_defaults(func=cmd_build)

    p = sub.add_parser('index')
    p.add_argument('path'); p.add_argument('--model', default=None); p.set_defaults(func=cmd_index)

    p = sub.add_parser('query')
    p.add_argument('path'); p.add_argument('query')
    p.add_argument('-k', type=int, default=10); p.add_argument('--full', action='store_true')
    p.set_defaults(func=cmd_query)

    p = sub.add_parser('grep')
    p.add_argument('path'); p.add_argument('word'); p.add_argument('-i', action='store_true')
    p.set_defaults(func=cmd_grep)

    args = ap.parse_args()
    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        print('\ninterrupted'); sys.exit(130)
    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == '__main__':
    main()
