# Multi-Agent RAG System - Technical Specification

## 1 Overview

This project demonstrates production-ready AI engineering capabilities including retrieval-augmented generation (RAG), autonomous agents, and self-improving systems. The implementation showcases real-world patterns for building LLM-powered applications with performance monitoring and automated evaluation.

## 2 Configuration

Environment setup:

```bash
OPENAI_BASE_URL=<your-openai-compatible-endpoint>
OPENAI_API_KEY=<your-api-key>
MODEL_NAME=<model-identifier>
```

**Technology Stack**: Python-based implementation

## 3 Core Components

The system consists of four integrated components, each demonstrating key AI engineering patterns.

### 3.1 Streaming Chat Interface with Observability

**Objective**: Build a conversational interface with comprehensive telemetry

**Requirements**:
- Implement token-by-token streaming to provide real-time response feedback
- Maintain conversation history (retain last 10 exchanges) using in-memory storage or lightweight database (SQLite, ChromaDB, etc.)
- Track and display usage metrics for each interaction:
  - Input token count
  - Output token count
  - Estimated cost in USD
  - End-to-end response time (milliseconds)

**Example Output**:
```
User: Hello
Assistant: [streaming response...]
[metrics] in=8 out=23 cost=$0.000146 latency=623ms
```

### 3.2 High-Performance Document QA System

**Objective**: Create a production-grade RAG pipeline optimized for speed and accuracy

**Implementation Requirements**:
- Ingest substantial text corpus (minimum 50MB) from PDFs, documents, or web sources
- Process documents into semantic chunks with embeddings
- Store vector representations in database (ChromaDB, FAISS, pgvector, or similar)
- Build query endpoint with these characteristics:
  - Generate answers with source attribution
  - Achieve sub-300ms median retrieval latency (excluding LLM inference time, warm cache)
- Develop automated evaluation framework:
  - Create test dataset with 20+ questions and ground-truth answers
  - Calculate top-5 retrieval accuracy (percentage of queries where correct document is in top 5 results)

**Suggested Datasets** (or use your own):
- Fantasy literature corpus (~50MB)
- Science fiction screenplay collection

### 3.3 Multi-Tool Planning Agent

**Objective**: Demonstrate agentic reasoning with external tool integration

**Scenario**: Process requests like "Plan a 2-day trip to Auckland under NZ$500"

**Capabilities**:
- Orchestrate multiple API calls or tools (e.g., travel search, weather data, points of interest)
- Maintain visible reasoning trace showing decision-making process
- Enforce hard constraints (budget limits, time windows, etc.)
- Generate structured output conforming to defined JSON schema

### 3.4 Self-Correcting Code Generator

**Objective**: Build an iterative refinement loop for code generation

**System Behavior**:
- Accept natural language programming tasks (e.g., "implement binary search in Python")
- Generate code using LLM and write to filesystem
- Execute appropriate test harness (pytest, cargo test, etc.)
- On test failure:
  - Capture error messages and stack traces
  - Feed diagnostics back to model for correction
  - Retry generation (maximum 3 attempts)
- Stream execution progress and final outcome to console

### 3.5 Optional: Analytics Dashboard

**Objective**: Containerize the system with monitoring UI

**Features**:
- Single-command deployment via Docker Compose
- Streamlit-based dashboard displaying:
  - Historical latency and cost trends
  - RAG accuracy metrics over time
  - Agent task success rates

## 4 Project Deliverables

**Repository Structure**:
1. Version-controlled codebase with meaningful commit history
2. Comprehensive README covering:
   - Installation instructions
   - System architecture overview
   - Test execution guide
3. Docker Compose configuration for service orchestration (vector DB, APIs, UIs)
4. Test suite in `tests/` directory (runnable via `pytest -q`)
5. Technical report (`report.md`, <500 words) documenting:
   - Architectural decisions
   - Performance trade-offs
   - Optimization strategies

## 5 Development Principles

**Code Attribution**: When leveraging open-source libraries or code examples, provide clear citations. Original implementation is expected with acknowledged dependencies.

**Security**: Credentials and sensitive configuration should never be committed to version control. Use environment variables and provide `.env.example` templates.

---

*This project specification is based on production AI platform requirements and demonstrates full-stack LLM application development.*


