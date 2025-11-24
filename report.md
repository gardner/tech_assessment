# Technical Assessment Report - Bickford AI Platform

## Overview

This implementation delivers all four core tasks plus the stretch goal, demonstrating a production-grade AI platform with RAG pipelines, autonomous agents, and comprehensive observability.

## Architecture Decisions

### Task 3.1: Conversational Core

**Implementation:** Vanilla OpenAI SDK with streaming support and Phoenix telemetry.

**Key Decisions:**
- **Token-level streaming:** Utilized OpenAI's native `stream=True` with `stream_options={"include_usage": True}` to capture token metrics while streaming responses
- **Message persistence:** In-memory list maintaining last 10 messages via array slicing
- **Cost tracking:** Azure GPT-4o pricing ($2.50/1M input, $1.25/1M cached, $10.00/1M output) with detailed breakdown including cached tokens
- **Telemetry:** Phoenix OTEL tracing for distributed observability without custom logging infrastructure

**Trade-offs:**
- Chose in-memory storage over SQLite/Chroma for simplicity; sufficient for N=10 requirement but not suitable for production persistence
- Included cached tokens in cost calculation for accurate billing transparency

### Task 3.2: High-Performance RAG

**Implementation:** Dual-pipeline architecture with naive (dense embeddings) and BM25 hybrid retrieval.

**Key Decisions:**
- **Embeddings:** Qwen3-Embedding-0.6B via Text Embeddings Inference server for high-throughput batch processing
- **Vector store:** ChromaDB persistent client with separate collections for naive/BM25 pipelines
- **Hybrid retrieval:** QueryFusionRetriever combining dense embeddings + BM25 with reciprocal rank fusion (RRF)
- **Citations:** Inline file references `[file=Source.md]` extracted from node metadata
- **Evaluation:** Promptfoo framework with custom Python providers measuring top-5 retrieval accuracy on 141 graded questions

**Performance:**
- **Naive pipeline:** 82ms mean retrieval latency (meets ≤300ms requirement)
- **BM25 pipeline:** 166ms mean retrieval latency (meets ≤300ms requirement)
- **Accuracy:** BM25 shows higher pass rates in evaluation results

**Trade-offs:**
- Disabled Cohere reranking to meet latency requirement; reranking added 1000+ms per query
- Used SimpleDocumentStore for BM25 (required by llama-index BM25Retriever) despite some redundancy with ChromaVectorStore
- TEI service runs externally (not in docker-compose by default) due to GPU requirements

### Task 3.3: Autonomous Planning Agent

**Implementation:** smolagents multi-agent system with CodeAgent orchestrator and specialized sub-agents.

**Key Decisions:**
- **Framework:** smolagents for native tool calling and multi-agent orchestration
- **External tools:**
  1. Flight booking via `fast-flights` library (Google Flights scraper) with mock booking.
  2. Event search via Event Finda API with mock booking.
- **Constraint handling:** Natural language parsing in agent prompt with budget/date validation in tool logic
- **Reasoning visibility:** smolagents built-in step logging exposes tool calls and observations
- **Output format:** JSON itinerary written to `itinerary.json` with structured flight and event data

**Trade-offs:**
- Used `fast-flights` (web scraping) instead of commercial APIs to avoid requiring API keys during assessment
- CodeAgent chosen over ToolCallingAgent for richer Python code generation capabilities within tools
- No explicit scratch-pad file; reasoning logged to stdout via smolagents' verbose mode

### Task 3.4: Self-Healing Code Assistant

**Implementation:** ToolCallingAgent with filesystem tools and test execution loop.

**Key Decisions:**
- **Agent framework:** smolagents ToolCallingAgent for structured tool calling
- **Tools:** `list_dir`, `read_file`, `write_file`, `run_test_cmd` - minimal filesystem abstraction
- **Test execution:** Shell subprocess running pytest/cargo test with stdout/stderr capture
- **Retry logic:** Max 3 test runs enforced in agent prompt; agent decides when to stop
- **Workspace isolation:** UUID-based `/workspace/{session_id}` directories prevent cross-session conflicts
- **Success detection:** Regex parsing for "RESULT: success/failure" in final agent output

**Trade-offs:**
- Simple prompt-based retry limit vs. programmatic step counting; more flexible but requires agent compliance
- Shell-based test execution vs. programmatic pytest API; shell approach supports arbitrary test commands (cargo, npm, etc.)
- Max 40 agent steps to prevent infinite loops while allowing complex multi-file edits

### Stretch Goal: Evaluation Dashboard

**Implementation:** Streamlit dashboard with Altair visualizations consuming promptfoo evaluation results.

**Key Features:**
- Pass/fail comparison across naive vs BM25 providers
- Latency distribution (box plots, histograms, scatter plots)
- Pass rate by latency bucket analysis
- Detailed results table with filtering

**Trade-offs:**
- Read-only dashboard (no live evaluation triggers) for simplicity
- JSON parsing from promptfoo output vs. direct database integration

## Deliverables Status

✅ **Git repository:** Clean commit history with feature-focused commits
✅ **README.md:** Setup instructions, architecture overview, and run commands
✅ **docker-compose.yml:** Multi-service orchestration (ingest, evaluate, dashboard, Phoenix observability)
✅ **tests/:** Unit tests for pure functions (`test_travel_data.py`, `test_code_agent_prompt.py`) and integration tests
✅ **report.md:** This document

## Testing & Validation

**Unit tests:**
- Travel data transformation logic
- Code agent prompt building

**Integration tests:**
- Full chat flow (mocked OpenAI client)

**Evaluation:**
- 141 graded RAG questions with top-5 retrieval accuracy metrics
- Automated via `./eval.sh` using promptfoo framework

**Coverage:** pytest configured with coverage reporting; execution via `pytest -q` as specified

## Key Technical Highlights

1. **Production-grade observability:** Phoenix OTEL integration across all LLM interactions (OpenAI SDK + smolagents)
2. **Performance optimization:** Disabled reranking to meet 300ms latency SLA while maintaining accuracy
3. **Hybrid retrieval:** BM25 + dense embeddings with RRF fusion outperforms naive approach in evaluation
4. **Containerization:** Single `docker-compose up` deploys entire stack including observability
5. **Framework selection:** smolagents for agent tasks (native tool calling, multi-agent support) vs. llama-index for RAG (mature retrieval abstractions)

## Challenges & Solutions

**Challenge:** Cohere reranker retrieval latency exceeding 300ms
**Solution:** Removed Cohere reranking post-processor; hybrid fusion still maintains higher accuracy than naive

**Challenge:** SimpleDocumentStore requirement for BM25 despite ChromaDB already storing vectors
**Solution:** Accepted minor redundancy; llama-index BM25Retriever requires docstore for full-text indexing separate from embeddings

**Challenge:** Test execution in isolated workspaces for code assistant
**Solution:** UUID-based workspace directories with subprocess execution; allows parallel sessions without conflicts

## Recommendations for Production

1. **Persistence:** Migrate chat message storage from in-memory to PostgreSQL/Redis for multi-user support
2. **Caching:** Add Redis layer for BM25 retrieval results (queries often repeat in production)
3. **Reranking:** Re-enable Cohere rerank for accuracy-critical use cases with relaxed latency requirements
4. **Rate limiting:** Implement token bucket for API cost control

