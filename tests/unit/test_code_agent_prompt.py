"""Unit tests for code agent prompt building functionality."""

import pytest
from unittest.mock import patch
from pathlib import Path
import tempfile
import sys

# Mock WORKSPACE_ROOT before importing code_agent to prevent /workspace creation
temp_workspace = Path(tempfile.mkdtemp())

# Patch the module-level workspace creation
with patch('os.makedirs'):
    from bickford.code_agent.code_agent import build_prompt


class TestBuildPrompt:
    """Test prompt building for code agent."""

    def test_build_prompt_basic(self):
        """Test basic prompt building with simple task."""
        task = "Write a fibonacci function in Python"
        test_command = "pytest"
        max_runs = 3

        prompt = build_prompt(task, test_command, max_runs)

        # Verify all components are included
        assert task in prompt
        assert test_command in prompt
        assert "RESULT: success" in prompt
        assert "RESULT: failure" in prompt

    def test_build_prompt_includes_workspace_info(self):
        """Test that prompt includes workspace path information."""
        task = "Create a hello world script"
        test_command = "python test.py"
        max_runs = 5

        prompt = build_prompt(task, test_command, max_runs)

        assert "/workspace/" in prompt

    def test_build_prompt_includes_tool_descriptions(self):
        """Test that prompt includes descriptions of available tools."""
        task = "Write a sorting algorithm"
        test_command = "pytest -q"
        max_runs = 3

        prompt = build_prompt(task, test_command, max_runs)

        # Check for tool descriptions
        assert "list_dir" in prompt
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "run_test_cmd" in prompt

    def test_build_prompt_includes_max_runs_limit(self):
        """Test that prompt specifies the max test runs limit."""
        task = "Implement quicksort"
        test_command = "cargo test"
        max_runs = 5

        prompt = build_prompt(task, test_command, max_runs)

        # Should mention the limit
        assert "5" in prompt or "five" in prompt.lower()

    def test_build_prompt_custom_test_command(self):
        """Test prompt building with custom test command."""
        task = "Write a Rust function"
        test_command = "cargo test --release"
        max_runs = 3

        prompt = build_prompt(task, test_command, max_runs)

        assert "cargo test --release" in prompt

    def test_build_prompt_output_format_instructions(self):
        """Test that prompt includes clear output format instructions."""
        task = "Any task"
        test_command = "pytest"
        max_runs = 3

        prompt = build_prompt(task, test_command, max_runs)

        # Check for output format guidance
        assert "exit_code == 0" in prompt
        assert "RESULT:" in prompt

    def test_build_prompt_special_characters_in_task(self):
        """Test that special characters in task are handled correctly."""
        task = 'Write a function that returns "Hello, World!"'
        test_command = "pytest"
        max_runs = 3

        prompt = build_prompt(task, test_command, max_runs)

        # Should include the task with quotes intact
        assert '"Hello, World!"' in prompt or "Hello, World!" in prompt
