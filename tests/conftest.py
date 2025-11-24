"""Pytest configuration and shared fixtures for all tests."""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path
import tempfile
import os


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for all tests to avoid requiring real credentials."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test-openai-api.local")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "test-model")
    monkeypatch.setenv("TEXT_EMBEDDINGS_INFERENCE_BASE_URL", "http://test-embeddings.local")
    monkeypatch.setenv("EVENTFINDA_USER", "test-user")
    monkeypatch.setenv("EVENTFINDA_PASS", "test-pass")


# ============================================================================
# OpenAI Mocks
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing chat functionality.

    Returns a mock client with configurable streaming responses.
    """
    client = Mock()
    return client


@pytest.fixture
def mock_streaming_response():
    """Mock OpenAI streaming response with sample chunks.

    Returns an iterator of chunks simulating a streaming response.
    """
    chunk1 = Mock()
    chunk1.choices = [Mock(delta=Mock(content="Hello"))]
    chunk1.usage = None

    chunk2 = Mock()
    chunk2.choices = [Mock(delta=Mock(content=" "), finish_reason=None)]
    chunk2.usage = None

    chunk3 = Mock()
    chunk3.choices = [Mock(delta=Mock(content="world!"), finish_reason=None)]
    chunk3.usage = None

    # Final chunk with usage stats
    chunk4 = Mock()
    chunk4.choices = []
    chunk4.usage = Mock(
        prompt_tokens=10,
        completion_tokens=5,
        prompt_tokens_details=Mock(cached_tokens=0)
    )

    return [chunk1, chunk2, chunk3, chunk4]


@pytest.fixture
def mock_usage_with_cache():
    """Mock usage object with cached tokens."""
    usage = Mock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.prompt_tokens_details = Mock(cached_tokens=80)
    return usage


@pytest.fixture
def mock_usage_no_cache():
    """Mock usage object without cached tokens."""
    usage = Mock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.prompt_tokens_details = Mock(cached_tokens=0)
    return usage


# ============================================================================
# LLM & RAG Mocks
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM for RAG testing.

    Returns a mock LLM with a complete() method that returns answers with citations.
    """
    llm = Mock()
    llm.complete.return_value = Mock(
        text="The answer is 42 [file=test-document.md]"
    )
    return llm


@pytest.fixture
def mock_node_with_score():
    """Create a mock NodeWithScore for RAG testing."""
    node = Mock()
    node.get_content.return_value = "This is sample content from the document."
    node.metadata = {
        "file_name": "test-doc.md",
        "title": "Test Document"
    }

    node_with_score = Mock()
    node_with_score.node = node
    node_with_score.score = 0.95

    return node_with_score


@pytest.fixture
def mock_retriever(mock_node_with_score):
    """Mock retriever for RAG testing.

    Returns a mock retriever that returns sample nodes.
    """
    retriever = Mock()
    retriever.retrieve.return_value = [mock_node_with_score]
    return retriever


# ============================================================================
# File System Mocks
# ============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for file system tests.

    Yields the path to the temp directory, then cleans it up.
    """
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Test Data
# ============================================================================

@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "What is Python?"},
    ]


@pytest.fixture
def sample_flight_data():
    """Sample flight data for testing travel agent transformations."""
    from dataclasses import dataclass

    @dataclass
    class MockFlight:
        airline: str
        departure_time: str
        arrival_time: str
        price: float
        is_best: bool

    return MockFlight(
        airline="Air New Zealand",
        departure_time="10:00",
        arrival_time="13:00",
        price=299.99,
        is_best=True
    )
