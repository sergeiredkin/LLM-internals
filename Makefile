# Petroleum RAG release commands.
# For the CUDA environment use:
#   make rag-build RAG_PYTHON="conda run -n gpu-test python"

RAG_PYTHON ?= python
RAG_MODEL ?= mistral:latest
RAG_HOST ?= 127.0.0.1
RAG_PORT ?= 7860

.PHONY: rag-help rag-build rag-check rag-test rag-benchmark rag-ollama rag-web

rag-help:
	@echo "make rag-build    Download/checksum sources and build the local RAG corpus"
	@echo "make rag-check    Verify the 50-source release and generated artifacts"
	@echo "make rag-test     Run the repository test suite"
	@echo "make rag-benchmark Run the frozen 50-question retrieval benchmark"
	@echo "make rag-ollama   Test all 50 questions through local Ollama"
	@echo "make rag-web      Start the browser UI at http://localhost:$(RAG_PORT)"

rag-build:
	$(RAG_PYTHON) -m scripts.build_petroleum_corpus

rag-check:
	$(RAG_PYTHON) -m scripts.check_petroleum_release --require-raw

rag-test:
	$(RAG_PYTHON) -m unittest discover -s tests -q

rag-benchmark:
	$(RAG_PYTHON) -m scripts.run_petroleum_production_benchmark \
		--documents data/petroleum/chunks-augmented.jsonl \
		--queries data/petroleum/queries-50.jsonl \
		--output reports/results/petroleum-benchmark-release.json

rag-ollama:
	$(RAG_PYTHON) -m scripts.ask_petroleum_ollama \
		--model $(RAG_MODEL) \
		--questions data/petroleum/gold-answers-50.jsonl \
		--output reports/results/petroleum-ollama-50q-release.jsonl

rag-web:
	$(RAG_PYTHON) -m scripts.petroleum_web --host $(RAG_HOST) --port $(RAG_PORT) --model $(RAG_MODEL)
