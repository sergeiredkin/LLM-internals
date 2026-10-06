#!/usr/bin/env python3
"""Local browser UI for the petroleum RAG + Ollama pipeline.

Start from anywhere:
    conda run -n gpu-test python -m scripts.petroleum_web

Then open http://localhost:7860.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from llm.rag import BM25Retriever, load_jsonl_chunks
from scripts.ask_petroleum_ollama import ask

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = REPO_ROOT / "data/petroleum/chunks-augmented.jsonl"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Petroleum RAG</title>
<style>
:root { color-scheme: dark; --bg:#101318; --panel:#191e27; --border:#303846; --accent:#63b3ed; --muted:#9aa7b8; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:#e8edf3; font:16px system-ui,-apple-system,Segoe UI,sans-serif; }
main { max-width:1000px; margin:0 auto; padding:28px 18px 60px; }
h1 { margin:0 0 6px; font-size:28px; }
.subtitle { color:var(--muted); margin-bottom:22px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:18px; margin:14px 0; }
textarea { width:100%; min-height:110px; resize:vertical; border:1px solid var(--border); border-radius:8px; background:#0d1117; color:#fff; padding:12px; font:inherit; }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:12px; }
select, button { border:1px solid var(--border); border-radius:8px; padding:10px 14px; background:#222a36; color:#fff; font:inherit; }
button { background:#256da8; cursor:pointer; font-weight:600; }
button:hover { background:#3182ce; } button:disabled { opacity:.5; cursor:wait; }
#answer { white-space:pre-wrap; line-height:1.6; }
.meta { color:var(--muted); font-size:14px; }
.source { border-top:1px solid var(--border); padding:12px 0; }
.source:first-child { border-top:0; }
.source-id { color:var(--accent); font-family:monospace; font-size:13px; }
.source-text { color:#cbd5e1; margin-top:5px; line-height:1.45; }
.error { color:#fc8181; white-space:pre-wrap; }
.hidden { display:none; }
</style>
</head>
<body>
<main>
<h1>Petroleum RAG</h1>
<div class="subtitle">50 verified USGS reports · page citations · local Ollama</div>
<div class="card">
<textarea id="question" placeholder="Ask a question about the petroleum reports..."></textarea>
<div class="controls">
<select id="model"><option>mistral:latest</option><option>qwen3.5:latest</option></select>
<button id="ask">Ask the RAG</button>
<span id="status" class="meta"></span>
</div>
</div>
<div id="result" class="card hidden">
<h2>Answer</h2>
<div id="answer"></div>
<div id="answer-meta" class="meta"></div>
<h2>Retrieved sources</h2>
<div id="sources"></div>
</div>
<div id="error" class="card error hidden"></div>
</main>
<script>
const question = document.getElementById('question');
const button = document.getElementById('ask');
const status = document.getElementById('status');
const result = document.getElementById('result');
const error = document.getElementById('error');
button.onclick = async () => {
  const q = question.value.trim();
  if (!q) return;
  button.disabled = true; status.textContent = 'Searching reports and asking Ollama...';
  result.classList.add('hidden'); error.classList.add('hidden');
  try {
    const response = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q, model:document.getElementById('model').value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    document.getElementById('answer').textContent = data.reply;
    document.getElementById('answer-meta').textContent = `${data.latency_seconds}s · citations: ${data.citations.length ? data.citations.join(', ') : 'none'} · grounded: ${data.checks.grounded}`;
    document.getElementById('sources').innerHTML = data.retrieved_ids.map((id, i) => `<div class="source"><div class="source-id">[${i+1}] ${escapeHtml(id)}</div><div class="source-text">${escapeHtml(data.source_texts[i] || '')}</div></div>`).join('');
    result.classList.remove('hidden'); status.textContent = 'Done';
  } catch (e) { error.textContent = e.message; error.classList.remove('hidden'); status.textContent = 'Failed'; }
  finally { button.disabled = false; }
};
question.addEventListener('keydown', e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') button.click(); });
function escapeHtml(s) { return String(s).replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    retriever: BM25Retriever
    default_model: str
    ollama_url: str
    timeout: int

    def send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "chunks": len(self.retriever.chunks)})
            return
        if self.path != "/":
            self.send_json(404, {"error": "not found"})
            return
        data = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ask":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("question is empty")
            args = SimpleNamespace(
                model=str(payload.get("model") or self.default_model),
                url=self.ollama_url,
                top_k=5,
                timeout=self.timeout,
            )
            result = ask(self.retriever, question, args)
            result["source_texts"] = [
                " ".join(chunk.text.split())[:1400] for chunk in self.retriever.chunks
                if chunk.document_id in result["retrieved_ids"]
            ]
            self.send_json(200, result)
        except Exception as exc:  # return readable UI errors
            self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args) -> None:
        print(f"[petroleum-web] {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default="mistral:latest")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    chunks = load_jsonl_chunks(args.documents)
    Handler.retriever = BM25Retriever(chunks)
    Handler.default_model = args.model
    Handler.ollama_url = args.ollama_url
    Handler.timeout = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Petroleum RAG ready: http://{args.host}:{args.port}", flush=True)
    print(f"Corpus: {len(chunks)} chunks | Ollama: {args.ollama_url} | model: {args.model}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Petroleum RAG", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
