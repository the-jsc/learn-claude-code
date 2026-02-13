"""
v3_subagent.py - Mini Claude Code: Subagent Mechanism (~450 lines)

v2 adds planning. But for large tasks like "explore the codebase then refactor auth", a single agent hits problems:
v2 增加了规划能力。但面对“先探索代码库再重构 auth”这类大任务，单一 agent 会遇到新问题：

The Problem - Context Pollution:
    Single-Agent History:
      [exploring...] cat file1.py -> 500 lines
      [exploring...] cat file2.py -> 300 lines
      ... 15 more files ...
      [now refactoring...] "Wait, what did file1 contain?"

The model's context fills with exploration details, leaving little room for the actual task. This is "context pollution".
模型上下文被探索输出填满，留给“真正任务”的空间就变少了。这就是“上下文污染”。

The Solution - Subagents with Isolated Context:
    Main Agent History:
      [Task: explore codebase]
        -> Subagent explores 20 files (in its own context)
        -> Returns ONLY: "Auth in src/auth/, DB in src/models/"
      [now refactoring with clean context]

Each subagent has:
  1. Its own fresh message history
  2. Filtered tools
  3. Specialized system prompt
  4. Returns only final summary to parent

The Key Insight:
---------------
    Process isolation = Context isolation
    进程隔离= 上下文隔离

By spawning subtasks, we get:
  - Clean context for the main agent
  - Parallel exploration possible
  - Natural task decomposition
  - Same agent loop, different contexts

Agent Type Registry:
-------------------
    | Type    | Tools               | Purpose                     |
    |---------|---------------------|---------------------------- |
    | explore | bash, read_file     | Read-only exploration       |
    | plan    | bash, read_file     | Design without modifying    |
    | code    | all tools           | Full implementation access  |

Typical Flow:
-------------
    User: "Refactor auth to use JWT"
    Main Agent:
      1. Task(explore): "Find all auth-related files"
         -> Subagent reads 10 files
         -> Returns: "Auth in src/auth/login.py..."
      2. Task(plan): "Design JWT migration"
         -> Subagent analyzes structure
         -> Returns: "1. Add jwt lib 2. Create utils..."
      3. Task(code): "Implement JWT tokens"
         -> Subagent writes code
         -> Returns: "Created jwt_utils.py, updated login.py"
      4. Summarize changes to user

Usage:
    python v3_subagent.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
WORKDIR = Path.cwd()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=os.getenv("ANTHROPIC_BASE_URL"))
model = os.getenv("MODEL")
max_tokens = int(os.getenv("MAX_TOKENS"))

# =============================================================================
# Agent Type Registry - The core of subagent mechanism
# =============================================================================
AGENT_TYPES = {
    # Explore: Read-only agent for searching and analyzing
    # Cannot modify files - safe for broad exploration
    "explore": {
        "description": "Read-only agent for exploring code, finding files, searching",
        "tools": ["bash", "read_file"],
        "prompt": "You are an exploration agent. Search and analyze, but never modify files. Return a concise summary.",
    },
    # Plan: Analysis agent for design work
    # Read-only, focused on producing plans and strategies
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["bash", "read_file"],
        "prompt": "You are a planning agent. Analyze and output a numbered implementation plan. Do NOT make changes.",
    },
    # Code: Full-powered agent for implementation
    # Has all tools - use for actual coding work
    "code": {
        "description": "Full agent for implementing features and fixing bugs",
        "tools": "*",
        "prompt": "You are a coding agent. Implement the requested changes efficiently.",
    },
}


def get_agent_descriptions() -> str:
    """
    Generate agent type descriptions for the Task tool.
    """
    return "\n".join(f"- {name}: {cfg['description']}" for name, cfg in AGENT_TYPES.items())


# =============================================================================
# TodoManager (from v2, unchanged)
# =============================================================================
class TodoManager:
    def __init__(self):
        self.tasks = []

    def update(self, tasks: list) -> str:
        validated = []
        in_progress_count = 0

        for i, task in enumerate(tasks):
            content = str(task.get("content", "")).strip()
            status = str(task.get("status", "pending")).lower()
            active_form = str(task.get("activeForm", "")).strip()

            if not content:
                raise ValueError(f"Task {i}: content required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Task {i}: invalid status '{status}'")
            if not active_form:
                raise ValueError(f"Task {i}: activeForm required")

            if status == "in_progress":
                in_progress_count += 1

            validated.append({"content": content, "status": status, "activeForm": active_form})

        if len(validated) > 20:
            raise ValueError("Max 20 todos allowed")
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.tasks = validated
        return self.render()

    def render(self) -> str:
        if not self.tasks:
            return "No todos."

        lines = []
        for task in self.tasks:
            if task["status"] == "completed":
                lines.append(f"[x] {task['content']}")
            elif task["status"] == "in_progress":
                lines.append(f"[>] {task['content']} <- {task['activeForm']}")
            else:
                lines.append(f"[ ] {task['content']}")

        completed = sum(1 for task in self.tasks if task["status"] == "completed")
        lines.append(f"\n({completed}/{len(self.tasks)} completed)")
        return "\n".join(lines)


todo = TodoManager()


# =============================================================================
# System Prompt
# =============================================================================
system = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

You can spawn subagents for complex subtasks:
{get_agent_descriptions()}

Rules:
- Use Task tool for subtasks that need focused exploration or implementation
- Use TodoWrite to track multi-step work
- Prefer tools over text. Act, don't just explain.
- After finishing, summarize what changed."""


# =============================================================================
# Base Tool Definitions
# =============================================================================
BASE_TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command. Use for: ls, find, grep, git, npm, python, etc.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to execute"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents. Returns UTF-8 text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "limit": {"type": "integer", "description": "Max lines to read (default: all)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path for the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in a file. Use for surgical edits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "old_text": {"type": "string", "description": "Exact text to find (must match precisely)"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "TodoWrite",
        "description": "Update the task list. Use to plan and track progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "Complete list of tasks (replaces existing)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Task description"},
                            "status": {
                                "type": "string",
                                "description": "Task status",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Present tense action, e.g. 'Reading files'",
                            },
                        },
                        "required": ["content", "status", "activeForm"],
                    },
                }
            },
            "required": ["tasks"],
        },
    },
]


# =============================================================================
# Task Tool - The core addition in v3
# =============================================================================
TASK_TOOL = {
    "name": "Task",
    "description": f"""Spawn a subagent for a focused subtask.
Subagents run in ISOLATED context - they don't see parent's history.
Use this to keep the main conversation clean.

Agent types:
{get_agent_descriptions()}

Example uses:
- Task(explore): "Find all files using the auth module"
- Task(plan): "Design a migration strategy for the database"
- Task(code): "Implement the user registration form"
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short task name (3-5 words) for progress display"},
            "prompt": {"type": "string", "description": "Detailed instructions for the subagent"},
            "agent_type": {"type": "string", "enum": list(AGENT_TYPES.keys()), "description": "Type of agent to spawn"},
        },
        "required": ["description", "prompt", "agent_type"],
    },
}

# Main agent gets all tools including Task
ALL_TOOLS = BASE_TOOLS + [TASK_TOOL]


def get_tools_for_agent(agent_type: str) -> list:
    """
    Filter tools based on agent type.

    Each agent type has a whitelist of allowed tools.
    '*' means all tools (but subagents don't get Task to prevent infinite recursion).
    每种 agent_type 都有一个工具白名单。
    '*' 表示允许全部基础工具（但 subagent 不会获得 Task，以避免演示中出现无限递归）。
    """
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")
    if allowed == "*":
        return BASE_TOOLS
    return [tool for tool in BASE_TOOLS if tool["name"] in allowed]


# =============================================================================
# Tool Implementations
# =============================================================================
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=60)
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit]
            lines.append(f"... ({len(text.splitlines()) - limit} more lines)")
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"

        new_content = content.replace(old_text, new_text, 1)
        fp.write_text(new_content)
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_todo(tasks: list) -> str:
    try:
        return todo.update(tasks)
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Subagent Execution - The heart of v3
# =============================================================================
def run_task(description: str, prompt: str, agent_type: str) -> str:
    """
    Execute a subagent task with isolated context.

    This is the core of the subagent mechanism:
    1. Create isolated message history (KEY: no parent context!)
    2. Use agent-specific system prompt
    3. Filter available tools based on agent type
    4. Run the same query loop as main agent
    5. Return ONLY the final text (not intermediate details)

    The parent agent sees just the summary, keeping its context clean.

    Progress Display:
    ----------------
    While running, we show:
      [explore] find auth files ... 5 tools, 3.2s

    This gives visibility without polluting the main conversation.
    """
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    config = AGENT_TYPES[agent_type]

    # Agent-specific system prompt
    sub_system = f"""You are a {agent_type} subagent at {WORKDIR}.
{config["prompt"]}
Complete the task and return a clear, concise summary."""

    # Filtered tools for this agent type
    sub_tools = get_tools_for_agent(agent_type)

    # ISOLATED message history - this is the key!
    # The subagent starts fresh, doesn't see parent's conversation
    sub_messages = [{"role": "user", "content": prompt}]

    # Progress tracking
    print(f"  [{agent_type}] {description}")
    start = time.time()
    tool_count = 0

    # Run the same agent loop (silently - don't print to main chat)
    while True:
        response = client.messages.create(
            model=model, system=sub_system, messages=sub_messages, tools=sub_tools, max_tokens=max_tokens
        )

        if response.stop_reason != "tool_use":
            break

        tool_calls = [block for block in response.content if block.type == "tool_use"]
        results = []

        for tc in tool_calls:
            tool_count += 1
            output = execute_tool(tc.name, tc.input)
            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

            # Update progress line (in-place)
            elapsed = time.time() - start
            sys.stdout.write(f"\r  [{agent_type}] {description} ... {tool_count} tools, {elapsed:.1f}s")
            sys.stdout.flush()

        sub_messages.append({"role": "assistant", "content": response.content})
        sub_messages.append({"role": "user", "content": results})

    # Final progress update
    elapsed = time.time() - start
    sys.stdout.write(f"\r  [{agent_type}] {description} - done ({tool_count} tools, {elapsed:.1f}s)\n")

    # Extract and return only the final text
    # This is what the parent agent sees - a clean summary
    for block in response.content:
        if block.type == "text":
            return block.text

    return "(subagent returned no text)"


def execute_tool(name: str, args: dict) -> str:
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read(args["path"], args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "TodoWrite":
        return run_todo(args["tasks"])
    if name == "Task":
        return run_task(args["description"], args["prompt"], args["agent_type"])
    return f"Unknown tool: {name}"


# =============================================================================
# Main Agent Loop
# =============================================================================
def agent_loop(messages: list) -> list:
    """
    Main agent loop with subagent support.

    Same pattern as v1/v2, but now includes the Task tool.
    When model calls Task, it spawns a subagent with isolated context.
    """
    while True:
        response = client.messages.create(
            model=model, system=system, messages=messages, tools=ALL_TOOLS, max_tokens=max_tokens
        )

        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
            if block.type == "tool_use":
                tool_calls.append(block)

        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            return messages

        results = []
        for tc in tool_calls:
            # Task tool has special display handling
            if tc.name == "Task":
                print(f"\n> Task: {tc.input.get('description', 'subtask')}")
            else:
                print(f"\n> {tc.name}")

            output = execute_tool(tc.name, tc.input)

            # Don't print full Task output (it manages its own display)
            if tc.name != "Task":
                preview = output[:200] + "..." if len(output) > 200 else output
                print(f"  {preview}")

            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# =============================================================================
def main():
    print(f"Mini Claude Code v3 (with Subagents) - {WORKDIR}")
    print(f"Agent types: {', '.join(AGENT_TYPES.keys())}")

    history = []

    while True:
        try:
            user_prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_prompt or user_prompt.lower() in ("exit", "quit"):
            break

        history.append({"role": "user", "content": user_prompt})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
