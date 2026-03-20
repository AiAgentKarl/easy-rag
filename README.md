# easy-rag

**RAG in 3 lines of code.** Zero dependencies. Load documents, ask questions, get answers.

```python
from easy_rag import EasyRAG

rag = EasyRAG("./my_docs/")
answer = rag.ask("What does the contract say about termination?")
print(answer)
```

That's it. No API keys, no vector databases, no configuration.

## Why easy-rag?

| | easy-rag | LangChain | LlamaIndex |
|---|---|---|---|
| Lines to get started | **3** | 50-100+ | 30-80+ |
| Dependencies | **0** | 50+ | 40+ |
| Setup time | **Instant** | Minutes | Minutes |
| Requires API key | **No** | Usually | Usually |
| Install size | **< 50 KB** | 200+ MB | 150+ MB |

## Installation

```bash
pip install easy-rag
```

### Optional extras

```bash
pip install easy-rag[pdf]      # PDF support (pdfplumber)
pip install easy-rag[openai]   # OpenAI embeddings
pip install easy-rag[local]    # Local embeddings (sentence-transformers)
pip install easy-rag[all]      # Everything
```

## Quick Start

### Search your documents

```python
from easy_rag import EasyRAG

# Load an entire folder of documents
rag = EasyRAG("./contracts/")

# Ask a question
result = rag.ask("What is the payment schedule?")
print(result)
# [contract_v2.pdf | Score: 0.87]
# Payment is due within 30 days of invoice date...
```

### Add documents on the fly

```python
rag = EasyRAG()

# Add individual files
rag.add_document("report.md")
rag.add_document("notes.txt")

# Add raw text directly
rag.add_text("The meeting is scheduled for Friday at 3pm.", source="email")

# Search
results = rag.search("meeting schedule", top_k=3)
for r in results:
    print(f"[{r.score:.2f}] {r.chunk.text[:100]}")
```

### CLI usage

```bash
# Search documents from the command line
easy-rag ./docs/ "What are the key findings?"

# Show index statistics
easy-rag ./docs/ --stats

# Customize chunk size
easy-rag ./docs/ "query" --chunk-size 1000 --top-k 5
```

### Check index stats

```python
rag = EasyRAG("./docs/")
print(rag.stats())
# {
#   'num_chunks': 142,
#   'num_sources': 12,
#   'num_unique_terms': 3847,
#   'chunk_size': 500,
#   'overlap': 50
# }
```

## Supported File Formats

| Format | Support | Notes |
|--------|---------|-------|
| `.txt` | Built-in | Plain text files |
| `.md` | Built-in | Markdown files |
| `.json` | Built-in | Extracts all string values |
| `.pdf` | Optional | Requires `pip install easy-rag[pdf]` |

## How It Works

1. **Load** — Reads all supported files from a directory
2. **Chunk** — Splits text into overlapping pieces at smart boundaries (paragraphs, sentences)
3. **Index** — Builds a TF-IDF index using only Python's standard library
4. **Search** — Finds the most relevant chunks using cosine similarity
5. **Return** — Gives you the best matching text with source and score

No neural networks, no API calls, no GPU needed. Just fast, reliable text search.

## API Reference

### `EasyRAG(path=None, chunk_size=500, overlap=50)`

Create a new RAG index.

- `path` — Directory or file to load (optional, can add later)
- `chunk_size` — Max characters per chunk (default: 500)
- `overlap` — Character overlap between chunks (default: 50)

### `.ask(question, top_k=3) -> RAGResult`

Ask a question and get a formatted answer with sources.

### `.search(query, top_k=5) -> List[SearchResult]`

Search for relevant chunks (raw results).

### `.add_document(filepath) -> int`

Add a single file to the index. Returns number of chunks created.

### `.add_text(text, source="inline") -> int`

Add raw text to the index. Returns number of chunks created.

### `.stats() -> dict`

Get index statistics.

## License

MIT
