# Bickford

A production-ready multi-agent RAG system demonstrating modern AI engineering patterns including retrieval-augmented generation, autonomous agents, and self-improving code generation.

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI-compatible endpoint and API key

# Run the streaming chat interface
uv run python -m bickford.chat

# Ingest documents and run RAG evaluation
uv run python -m bickford.ingest
uv run python -m bickford.bm25.ingest
./eval.sh

# Use the travel planning agent
uv run travel_agent

# Use the self-healing code generator
uv run code_agent --task "Write a function to calculate the Fibonacci sequence in Python"
```

## 📦 Project Structure

```
bickford/
├── src/bickford/
│   ├── chat.py              # Streaming chat with telemetry
│   ├── telemetry.py         # Cost and performance tracking
│   ├── rag/                 # RAG pipeline
│   │   ├── naive/           # Basic semantic search
│   │   └── bm25/            # Hybrid BM25 + dense search
│   ├── travel/              # Travel planning agent
│   ├── code_agent/          # Self-healing code generator
│   └── dashboard/           # Streamlit analytics dashboard
├── tests/                   # Unit and integration tests
├── data/                    # Document corpus (50+ MB)
└── docker-compose.yml       # Containerized deployment
```

## 🎯 Features

### 1. Streaming Chat Interface with Observability

A conversational AI with comprehensive telemetry:
- **Token-level streaming** for real-time response feedback
- **Message history** (last 10 exchanges) with in-memory persistence
- **Performance metrics** including token counts, cost (USD), and latency

Example output:
```
User: Hello
Assistant: Hi! How can I help you today?
[stats] prompt=8 completion=23 cost=$0.000146 latency=623ms
```

**Implementation**: Built with vanilla OpenAI SDK, including cached token cost calculation.

### 2. High-Performance RAG Pipeline

Production-grade retrieval-augmented generation optimized for speed and accuracy:

- **Document Ingestion**: 50+ MB text corpus with semantic chunking
- **Vector Storage**: ChromaDB with persistent storage
- **Query Endpoint**: Sub-300ms median retrieval latency (warm cache, LLM inference excluded)
- **Inline Citations**: Source attribution for all generated answers
- **Automated Evaluation**: 20+ test questions with ground-truth answers

#### RAG Variants

**Naive RAG**: Basic dense embedding search
- **Accuracy**: 90% top-5 retrieval

**Hybrid BM25**: Combined sparse + dense retrieval with Reciprocal Rank Fusion
- **Accuracy**: 95% top-5 retrieval
- **Improvement**: 5% over naive approach

### 3. Autonomous Travel Planning Agent

Multi-tool agentic system with:
- **Tool Orchestration**: Coordinates flights, weather, and attractions APIs
- **Constraint Enforcement**: Budget limits, date windows
- **Reasoning Traces**: Visible decision-making process
- **Structured Output**: JSON schema validation

Example: Plan a 2-day Auckland trip under NZ$500

### 4. Self-Healing Code Generator

Iterative refinement loop for code generation:
- **Natural Language Tasks**: "Implement binary search in Python"
- **Automatic Test Execution**: pytest, cargo test, etc.
- **Error Feedback**: Captures diagnostics and self-corrects
- **Retry Logic**: Maximum 3 attempts with streaming progress
- **Success Reporting**: Final outcome to console

### 5. Analytics Dashboard (Optional)

Streamlit-based observability:
- Latency and cost metrics over time
- RAG accuracy comparison charts
- Agent success/failure breakdown

## 🧪 Evaluation

### Top-5 Retrieval Accuracy

Automated evaluation using [promptfoo](https://www.promptfoo.dev/):

```bash
./eval.sh
```

**Components**:
- `src/bickford/rag/naive/top_k.py` - Naive RAG provider (90% accuracy)
- `src/bickford/rag/bm25/top_k.py` - Hybrid BM25 provider (95% accuracy)
- `tests/top-k.csv` - Test cases with expected file names
- `promptfooconfig.yaml` - Evaluation configuration

**Metric**: Percentage of queries where the correct source document appears in top-5 results.

## 🧰 Technology Stack

- **Language**: Python 3.12
- **Package Manager**: `uv`
- **LLM Framework**: OpenAI SDK, llama-index, smolagents
- **Vector DB**: ChromaDB
- **Evaluation**: promptfoo
- **Testing**: pytest with fixtures
- **Dashboard**: Streamlit + Altair
- **Deployment**: Docker + docker-compose

## 🐳 Docker Deployment

Single-command containerized deployment:

```bash
docker compose up
```

Includes:
- ChromaDB vector database
- Application services
- Volume persistence for data
- Streamlit dashboard (optional)

## ✅ Testing

Run the test suite:

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest -m unit

# Integration tests only
uv run pytest -m integration

# With coverage
uv run pytest --cov=src/bickford --cov-report=html
```

## 📄 Documentation

- **`report.md`**: Technical report on architecture, design decisions, and trade-offs
- **`docs/Tech_Questions_AI_1.md`**: Project specification

## ⚙️ Configuration

Required environment variables (see `.env.example`):

```bash
OPENAI_BASE_URL=<your-openai-compatible-endpoint>
OPENAI_API_KEY=<your-api-key>
MODEL_NAME=<model-identifier>
```

## 🎓 Skills Demonstrated

- Production-grade LLM application development
- RAG pipeline optimization (hybrid search, reranking)
- Agentic system design with tool orchestration
- Self-improving AI loops with error feedback
- Comprehensive testing and evaluation frameworks
- Observability and cost tracking
- Containerization and deployment best practices
