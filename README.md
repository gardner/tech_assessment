# Bickford

*TL;DR*

```bash
uv run python -m bickford.chat

uv run python -m bickford.ingest
uv run python -m bickford.bm25.ingest
./eval.sh
uv run travel_agent
# Or: uv run -m bickford.travel
uv run code_agent --task "Write a function to calculate the Fibonacci sequence in python."
# Or: uv run -m bickford.code_agent --task "..."
```

## Chat

### Task 3.1 Conversational Core (Streaming & Cost Telemetry)
<ol type="a">
  <li>Implement a chat loop against MODEL_NAME with token-level streaming. Show incremental tokens in the UI/CLI.</li>
  <li>Persist the last N = 10 messages in memory or a lightweight DB (Chroma, SQLite, etc.).</li>
  <li>Log and display for each turn: prompt tokens, completion tokens, total cost (USD) and round-trip latency.</li>
</ol>

*Acceptance test:* running `uv run python -m bickford.chat` then typing "Hello" prints a streamed assistant response plus a metrics line such as:

```
[stats] prompt=8 completion=23 cost=$0.000146 latency=623 ms
```

#### Notes

This task was accomlished using the vanilla OpenAI SDK.
I included cached tokens in the cost calculated .

### Task 3.2 High-Performance Retrieval-Augmented QA

<ol type="a">
  <li>Ingest at least 50 MB of text (PDFs, docs or crawled pages). Chunk and embed with a model of your choice.</li>
  <li>Store vectors in the DB of your choice (Chroma, FAISS, pgvector, ...).</li>
  <li>Implement a QA endpoint that:
    <ol type="i">
      <li>returns an answer with inline citations, and</li>
      <li>responds in ≤ 300 ms1 median retrieval time.</li>
    </ol>
  </li>
  <li>Provide an automated script that reports top-5 retrieval accuracy on (≥ 20) graded questions (you create the answer key).</li>
</ol>

# Evaluation

## Top-5 Retrieval Accuracy

The primary metric for Task 3.2(d) is **top-5 retrieval accuracy**, which measures whether the correct source document is retrieved in the top-5 results.

**Recommended: Using promptfoo (no custom code):**

```bash
./eval.sh
```

This uses:
- `src/bickford/rag/naive/top_k.py` and `src/bickford/rag/bm25/top_k.py` - A promptfoo provider that returns top-5 retrieved file names
- `tests/top-k.csv` - Test cases with expected file names
- `promptfooconfig.yaml` - Configuration using promptfoo's built-in `contains` assertion
