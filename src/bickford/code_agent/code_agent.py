
# Task 3.4 Self-Healing Code Assistant with ToolCallingAgent
# a) accepts a natural-language coding task
# b) uses tools to read/write files and run tests
# c) uses test failures to iteratively fix the code (up to a limit)
# d) streams progress and final success/failure to the console.

import subprocess
import sys
import os
import uuid
from pathlib import Path

import click
from smolagents import ToolCallingAgent, tool
from bickford.code_agent.config import model
from bickford.telemetry import setup_tracing

setup_tracing()

# Workspace root (repo root). You can tweak this if needed.
session_dir = str(uuid.uuid4())
WORKSPACE_ROOT = Path(f"/workspace/{session_dir}").resolve()

os.makedirs(WORKSPACE_ROOT, exist_ok=True)

@tool
def list_dir(path: str = ".") -> str:
    """
    List files and directories relative to the workspace root.

    Args:
        path: Relative path from workspace root.

    Returns:
        A newline-separated string of entries.
    """
    base = WORKSPACE_ROOT / path
    if not base.exists():
        return f"{base} does not exist."

    entries = []
    for p in sorted(base.iterdir()):
        kind = "dir " if p.is_dir() else "file"
        entries.append(f"{kind:4} {p.relative_to(WORKSPACE_ROOT)}")
    if len(entries) == 0:
        return "No files or directories found. Use the `write_file` tool to create a new file."
    return "\n".join(entries)


@tool
def read_file(path: str) -> str:
    """
    Read a text file relative to the workspace root.

    Args:
        path: Relative path to file.

    Returns:
        File contents as a string, or an error message.
    """
    file_path = WORKSPACE_ROOT / path
    if not file_path.exists():
        return f"ERROR: {path} does not exist."
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"ERROR reading {path}: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """
    Overwrite or create a text file relative to the workspace root.

    Args:
        path: Relative path to file.
        content: Full file contents.

    Returns:
        Short status string.
    """
    file_path = WORKSPACE_ROOT / path
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


@tool
def run_test_cmd(test_command: str = "uv run pytest -q") -> dict:
    """
    Run the test command in the workspace and return structured results.

    Args:
        test_command: Shell command to run tests,
                      e.g. 'uv run pytest -q' or 'cargo test'.

    Returns:
        dict with keys: exit_code (int), stdout (str), stderr (str).
    """
    proc = subprocess.run(
        test_command,
        shell=True,
        cwd=WORKSPACE_ROOT,
        text=True,
        capture_output=True,
    )

    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout.replace("  ", " ").strip(),
        "stderr": proc.stderr,
    }


def build_agent() -> ToolCallingAgent:
    """Create the smolagents ToolCallingAgent wired to our tools."""
    agent = ToolCallingAgent(
        tools=[list_dir, read_file, write_file, run_test_cmd],
        model=model,
        stream_outputs=True,
        max_steps=40,
        name="self_healing_tool_agent",
        description=(
            "Uses tools to inspect and edit the workspace and run tests until "
            "the test suite passes or the attempt limit is reached."
        ),
    )
    return agent


def build_prompt(task: str, test_command: str, max_test_runs: int) -> str:
    """
    Build a single high-level instruction for the ToolCallingAgent.

    The agent will:
    - inspect files using list_dir/read_file
    - edit code using write_file
    - run tests using run_test_cmd(test_command=...)
    - repeat as needed, up to max_test_runs times
    - finally emit RESULT: success or RESULT: failure.
    """
    return f"""
You are a senior software engineer working in a local repository at:

    {WORKSPACE_ROOT}

Your job is to modify or create code so that the test command:

    {test_command}

passes successfully (exit_code == 0).

Task (natural language request from the user):
    {task}

You interact with the filesystem ONLY via these tools:

- list_dir(path="."): list files and directories relative to the repo root.
- read_file(path): read a text file.
- write_file(path, content): fully overwrite or create a file.
- run_test_cmd(test_command): run the test suite and get exit_code, stdout, stderr.

IMPORTANT RULES:

1. You MUST use list_dir/read_file to understand the existing project before making large changes.
2. You MUST use write_file to create or modify code and/or tests, especially if no files exist.
3. You MUST use run_test_cmd(test_command="{test_command}") to actually run the tests.
4. You are allowed to call run_test_cmd at most {max_test_runs} times.
5. After each failing test run, carefully inspect exit_code, stdout, stderr, update the code, and try again.
6. Stop early if you get exit_code == 0 from run_test_cmd (tests passing).


If you see successful test runs and don't get any errors then the test command was successful.

Output format:

- Throughout the process, you may explain what you're doing.
- At the VERY END, append a single line on its own:

    RESULT: success

  if the last test run you executed had exit_code == 0.

- Or, if you used all allowed test runs and could not get exit_code == 0,
  append a single line:

    RESULT: failure

Do NOT invent test results. You must base your final RESULT on the actual exit_code from the last run_test_cmd call you made.
Please include the number of tests run and the number of tests passed in the final answer output.

IMPORTANT: The last line of your output must be "RESULT: failure" or "RESULT: success".
""".strip()


def run_self_healing_task(
    task: str,
    test_command: str = "pytest -q",
    max_test_runs: int = 3,
) -> bool:
    """
    Run the self-healing assistant as a single ToolCallingAgent session.

    Returns True if the agent reports RESULT: success, else False.
    """
    agent = build_agent()
    prompt = build_prompt(task, test_command, max_test_runs)

    last_output = None

    print("\n============================")
    print("🚀 Starting self-healing session (ToolCallingAgent)")
    print("============================\n")

    # Stream steps so you see tool calls and outputs in real time
    for step in agent.run(prompt, stream=True):
        # For smolagents ToolCallingAgent, steps expose tool calls / observations.
        if getattr(step, "step_number", None) is not None:
            print(f"\n[step {step.step_number}]")

        if getattr(step, "tool_calls", None):
            print("\n[tool_calls]")
            print(step.tool_calls)

        if getattr(step, "observations", None):
            print("\n[observations]")
            obs_str = str(step.observations)
            print(obs_str[:800])  # avoid spamming

        if getattr(step, "output", None) is not None:
            last_output = step.output

    if last_output is None:
        print("\n💀 Agent finished without a final output. Treating as failure.")
        return False

    print("\n[final agent output]")
    print(last_output)

    # Decide success / failure based on RESULT line.
    # We only care about the final RESULT tag.
    success = "RESULT: success" in last_output.splitlines()[-1]
    if success:
        print("\n✅ Agent reports: RESULT: success")
    else:
        print("\n❌ Agent reports: RESULT: failure", last_output)

    return success


@click.command()
@click.argument("task")
@click.option(
    "--test-command",
    default="pytest -q",
    help='Shell command to run tests (e.g. "pytest -q" or "cargo test").',
)
@click.option(
    "--max-test-runs",
    type=int,
    default=3,
    help="Maximum number of test runs the agent is allowed to perform.",
)
def main(task: str, test_command: str, max_test_runs: int) -> None:
    """
    Self-healing code assistant using smolagents ToolCallingAgent.

    TASK: Natural language coding task, e.g. "write quicksort in Rust in src/lib.rs"
    """
    success = run_self_healing_task(
        task=task,
        test_command=test_command,
        max_test_runs=max_test_runs,
    )
    print(f"Workspace root: {WORKSPACE_ROOT}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()