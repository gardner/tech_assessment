"""Integration tests for chat with mocked OpenAI client."""

import pytest
from unittest.mock import Mock, patch, call
from bickford.chat import generate_response


class TestChatGenerateResponse:
    """Test generate_response with mock OpenAI streaming."""

    def test_generate_response_basic_streaming(self, mock_streaming_response):
        """Test basic streaming response generation."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter(mock_streaming_response)

        messages = [{"role": "user", "content": "Hello"}]

        # Collect all chunks
        chunks = list(generate_response(mock_client, messages))

        # Should have 4 chunks (3 content + 1 usage)
        assert len(chunks) == 4

        # Verify client was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == messages
        assert call_kwargs["stream"] is True

    def test_generate_response_reconstructs_full_text(self, mock_streaming_response):
        """Test that streaming chunks can be reconstructed into full text."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter(mock_streaming_response)

        messages = [{"role": "user", "content": "Test message"}]

        # Reconstruct full response
        full_text = ""
        for chunk in generate_response(mock_client, messages):
            if chunk.choices and chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content

        assert full_text == "Hello world!"

    def test_generate_response_includes_usage_stats(self, mock_streaming_response):
        """Test that final chunk includes usage statistics."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = iter(mock_streaming_response)

        messages = [{"role": "user", "content": "Test"}]

        # Get last chunk
        chunks = list(generate_response(mock_client, messages))
        last_chunk = chunks[-1]

        assert last_chunk.usage is not None
        assert last_chunk.usage.prompt_tokens == 10
        assert last_chunk.usage.completion_tokens == 5
        assert last_chunk.usage.prompt_tokens_details.cached_tokens == 0

    def test_generate_response_with_system_message(self):
        """Test generation with system message in conversation."""
        mock_client = Mock()

        chunk = Mock()
        chunk.choices = [Mock(delta=Mock(content="Response"))]
        chunk.usage = Mock(
            prompt_tokens=5,
            completion_tokens=3,
            prompt_tokens_details=Mock(cached_tokens=0)
        )

        mock_client.chat.completions.create.return_value = iter([chunk])

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]

        list(generate_response(mock_client, messages))

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"


class TestCostCalculation:
    """Test cost calculation logic."""

    def test_cost_with_no_cache(self, mock_usage_no_cache):
        """Test cost calculation without cached tokens."""
        # This will need to be updated when we extract calculate_cost()
        # For now, we test the cost formula manually

        usage = mock_usage_no_cache

        # Cost formula from chat.py lines 75-83
        input_tokens = usage.prompt_tokens - usage.prompt_tokens_details.cached_tokens
        cached_tokens = usage.prompt_tokens_details.cached_tokens
        output_tokens = usage.completion_tokens

        input_cost = (input_tokens * 2.50) / 1_000_000
        cache_cost = (cached_tokens * 1.25) / 1_000_000
        output_cost = (output_tokens * 10.00) / 1_000_000
        total_cost = input_cost + cache_cost + output_cost

        # 100 * 2.50/1M + 0 * 1.25/1M + 50 * 10.00/1M
        expected_cost = (100 * 2.50 + 0 * 1.25 + 50 * 10.00) / 1_000_000
        assert abs(total_cost - expected_cost) < 1e-10
        assert abs(total_cost - 0.00075) < 1e-10

    def test_cost_with_cache(self, mock_usage_with_cache):
        """Test cost calculation with cached tokens."""
        usage = mock_usage_with_cache

        # Cost formula
        input_tokens = usage.prompt_tokens - usage.prompt_tokens_details.cached_tokens
        cached_tokens = usage.prompt_tokens_details.cached_tokens
        output_tokens = usage.completion_tokens

        input_cost = (input_tokens * 2.50) / 1_000_000
        cache_cost = (cached_tokens * 1.25) / 1_000_000
        output_cost = (output_tokens * 10.00) / 1_000_000
        total_cost = input_cost + cache_cost + output_cost

        # 20 * 2.50/1M + 80 * 1.25/1M + 50 * 10.00/1M
        expected_cost = (20 * 2.50 + 80 * 1.25 + 50 * 10.00) / 1_000_000
        assert abs(total_cost - expected_cost) < 1e-10
