#!/usr/bin/env python3
"""
v3_subagent.py - Mini Claude Code: Subagent Mechanism (~450 lines)
v3_subagent.py - 迷你 Claude Code：Subagent Mechanism（子代理机制，约 450 行）

Core Philosophy: "Divide and Conquer with Context Isolation"
=============================================================
核心理念："Divide and Conquer with Context Isolation（通过上下文隔离来分而治之）"

v2 adds planning. But for large tasks like "explore the codebase then
refactor auth", a single agent hits problems:
v2 增加了规划能力。但面对“先探索代码库再重构 auth”这类大任务，单一 agent 会遇到新问题：

The Problem - Context Pollution:
-------------------------------
问题：Context Pollution（上下文污染）

    Single-Agent History:
      [exploring...] cat file1.py -> 500 lines
      [exploring...] cat file2.py -> 300 lines
      ... 15 more files ...
      [now refactoring...] "Wait, what did file1 contain?"
（单 agent 历史里塞满探索细节，真正要改代码时反而记不清关键内容。）

The model's context fills with exploration details, leaving little room
for the actual task. This is "context pollution".
模型上下文被探索输出填满，留给“真正任务”的空间就变少了——这就是“上下文污染”。

The Solution - Subagents with Isolated Context:
----------------------------------------------
解决方案：带上下文隔离（isolated context）的 subagent（子代理）

    Main Agent History:
      [Task: explore codebase]
        -> Subagent explores 20 files (in its own context)
        -> Returns ONLY: "Auth in src/auth/, DB in src/models/"
      [now refactoring with clean context]
（主 agent 只拿到摘要，保持上下文干净，从而更适合后续实现与重构。）

Each subagent has:
  1. Its own fresh message history
  2. Filtered tools (explore can't write)
  3. Specialized system prompt
  4. Returns only final summary to parent
每个 subagent 都具备：
  1. 独立且全新的消息历史
  2. 经过筛选的工具（例如 explore 不能写文件）
  3. 专用的 system prompt
  4. 只把最终摘要返回给父 agent

The Key Insight:
---------------
    Process isolation = Context isolation
关键洞察：
    进程隔离（process isolation）= 上下文隔离（context isolation）

By spawning subtasks, we get:
  - Clean context for the main agent
  - Parallel exploration possible
  - Natural task decomposition
  - Same agent loop, different contexts
通过把子任务拆出去，我们获得：
  - 主 agent 更干净的上下文
  - 具备并行探索的可能性
  - 更自然的任务拆解
  - 同一套 agent loop，不同的上下文与能力配置

Agent Type Registry:
-------------------
    | Type    | Tools               | Purpose                     |
    |---------|---------------------|---------------------------- |
    | explore | bash, read_file     | Read-only exploration       |
    | code    | all tools           | Full implementation access  |
    | plan    | bash, read_file     | Design without modifying    |
（上表保持原样；核心是：不同 agent_type 对应不同工具白名单与职责。）

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
用法：
    python v3_subagent.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)


# =============================================================================
# Configuration
# 配置
# =============================================================================

WORKDIR = Path.cwd()

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")


# =============================================================================
# Agent Type Registry - The core of subagent mechanism
# Agent Type Registry（代理类型注册表）——subagent 机制的核心
# =============================================================================

AGENT_TYPES = {
    # Explore: Read-only agent for searching and analyzing
    # Explore：只读 agent，用于搜索与分析
    # Cannot modify files - safe for broad exploration
    # 不能修改文件，适合大范围探索（更安全）
    "explore": {
        "description": "Read-only agent for exploring code, finding files, searching",
        "tools": ["bash", "read_file"],
        # No write access
        # 无写入权限
        "prompt": "You are an exploration agent. Search and analyze, but never modify files. Return a concise summary.",
    },

    # Code: Full-powered agent for implementation
    # Code：全功能 agent，用于实现与修复
    # Has all tools - use for actual coding work
    # 拥有所有工具——用于真正的 coding 工作
    "code": {
        "description": "Full agent for implementing features and fixing bugs",
        "tools": "*",
        # All tools
        # 全部工具
        "prompt": "You are a coding agent. Implement the requested changes efficiently.",
    },

    # Plan: Analysis agent for design work
    # Plan：分析/设计用 agent
    # Read-only, focused on producing plans and strategies
    # 只读，专注输出计划与策略
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["bash", "read_file"],
        # Read-only
        # 只读
        "prompt": "You are a planning agent. Analyze the codebase and output a numbered implementation plan. Do NOT make changes.",
    },
}


def get_agent_descriptions() -> str:
    """Generate agent type descriptions for the Task tool.
    为 Task 工具生成 agent_type 描述列表。
    """
    return "\n".join(
        f"- {name}: {cfg['description']}"
        for name, cfg in AGENT_TYPES.items()
    )


# =============================================================================
# TodoManager (from v2, unchanged)
# TodoManager（继承自 v2，保持不变）
# =============================================================================

class TodoManager:
    """Task list manager with constraints. See v2 for details.
    带约束的任务列表管理器（细节见 v2）。
    """

    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        validated = []
        in_progress = 0

        for i, item in enumerate(items):
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            active = str(item.get("activeForm", "")).strip()

            if not content or not active:
                raise ValueError(f"Item {i}: content and activeForm required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status")
            if status == "in_progress":
                in_progress += 1

            validated.append({
                "content": content,
                "status": status,
                "activeForm": active
            })

        if in_progress > 1:
            raise ValueError("Only one task can be in_progress")

        self.items = validated[:20]
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for t in self.items:
            mark = "[x]" if t["status"] == "completed" else \
                   "[>]" if t["status"] == "in_progress" else "[ ]"
            lines.append(f"{mark} {t['content']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        return "\n".join(lines) + f"\n({done}/{len(self.items)} done)"


TODO = TodoManager()


# =============================================================================
# System Prompt
# System Prompt（系统提示词）
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

You can spawn subagents for complex subtasks:
{get_agent_descriptions()}

Rules:
- Use Task tool for subtasks that need focused exploration or implementation
- Use TodoWrite to track multi-step work
- Prefer tools over prose. Act, don't just explain.
- After finishing, summarize what changed."""


# =============================================================================
# Base Tool Definitions
# 基础工具定义
# =============================================================================

BASE_TOOLS = [
    {
        "name": "bash",
        "description": "Run shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "TodoWrite",
        "description": "Update task list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            },
                            "activeForm": {"type": "string"},
                        },
                        "required": ["content", "status", "activeForm"],
                    },
                }
            },
            "required": ["items"],
        },
    },
]


# =============================================================================
# Task Tool - The core addition in v3
# Task 工具——v3 的核心增量
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
            "description": {
                "type": "string",
                "description": "Short task name (3-5 words) for progress display"
            },
            "prompt": {
                "type": "string",
                "description": "Detailed instructions for the subagent"
            },
            "agent_type": {
                "type": "string",
                "enum": list(AGENT_TYPES.keys()),
                "description": "Type of agent to spawn"
            },
        },
        "required": ["description", "prompt", "agent_type"],
    },
}

# Main agent gets all tools including Task
# 主 agent 拥有所有工具（包含 Task）
ALL_TOOLS = BASE_TOOLS + [TASK_TOOL]


def get_tools_for_agent(agent_type: str) -> list:
    """
    Filter tools based on agent type.
    根据 agent_type 过滤可用工具。

    Each agent type has a whitelist of allowed tools.
    '*' means all tools (but subagents don't get Task to prevent infinite recursion).
    每种 agent_type 都有一个工具白名单。
    '*' 表示允许全部基础工具（但 subagent 不会获得 Task，以避免演示中出现无限递归）。
    """
    allowed = AGENT_TYPES.get(agent_type, {}).get("tools", "*")

    if allowed == "*":
        return BASE_TOOLS  # All base tools, but NOT Task (no recursion in demo)
        # 返回全部基础工具，但不包含 Task（演示版避免递归）

    return [t for t in BASE_TOOLS if t["name"] in allowed]


# =============================================================================
# Tool Implementations
# 工具实现
# =============================================================================

def safe_path(p: str) -> Path:
    """Ensure path stays within workspace.
    确保路径保持在工作区内。
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(cmd: str) -> str:
    """Execute shell command with safety checks.
    执行 shell 命令并做基础安全检查。
    """
    if any(d in cmd for d in ["rm -rf /", "sudo", "shutdown"]):
        return "Error: Dangerous command"
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=60
        )
        return ((r.stdout + r.stderr).strip() or "(no output)")[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """Read file contents.
    读取文件内容。
    """
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit:
            lines = lines[:limit]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """Write content to file.
    写入文件内容。
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file.
    替换文件中的精确文本片段。
    """
    try:
        fp = safe_path(path)
        text = fp.read_text()
        if old_text not in text:
            return f"Error: Text not found in {path}"
        fp.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_todo(items: list) -> str:
    """Update the todo list.
    更新 todo 列表。
    """
    try:
        return TODO.update(items)
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Subagent Execution - The heart of v3
# Subagent 执行——v3 的核心
# =============================================================================

def run_task(description: str, prompt: str, agent_type: str) -> str:
    """
    Execute a subagent task with isolated context.
    在隔离上下文中执行一个 subagent 任务。

    This is the core of the subagent mechanism:

    1. Create isolated message history (KEY: no parent context!)
    2. Use agent-specific system prompt
    3. Filter available tools based on agent type
    4. Run the same query loop as main agent
    5. Return ONLY the final text (not intermediate details)
    这就是 subagent 机制的核心：
    1. 创建隔离的消息历史（关键：不继承父上下文）
    2. 使用该 agent_type 专属的 system prompt
    3. 按 agent_type 过滤可用工具
    4. 运行与主 agent 相同的 query loop
    5. 只返回最终文本（不返回中间细节）

    The parent agent sees just the summary, keeping its context clean.
    父 agent 只看到摘要，从而保持上下文干净。

    Progress Display:
    ----------------
    While running, we show:
      [explore] find auth files ... 5 tools, 3.2s

    This gives visibility without polluting the main conversation.
    进度展示：
    运行时会输出进度行，让你可见进展，但不会污染主对话上下文。
    """
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    config = AGENT_TYPES[agent_type]

    # Agent-specific system prompt
    # agent_type 专属的 system prompt
    sub_system = f"""You are a {agent_type} subagent at {WORKDIR}.

{config["prompt"]}

Complete the task and return a clear, concise summary."""

    # Filtered tools for this agent type
    # 按 agent_type 过滤工具
    sub_tools = get_tools_for_agent(agent_type)

    # ISOLATED message history - this is the key!
    # 隔离的消息历史——这是关键！
    # The subagent starts fresh, doesn't see parent's conversation
    # subagent 从零开始，不会看到父对话
    sub_messages = [{"role": "user", "content": prompt}]

    # Progress tracking
    # 进度追踪
    print(f"  [{agent_type}] {description}")
    start = time.time()
    tool_count = 0

    # Run the same agent loop (silently - don't print to main chat)
    # 运行同样的 agent loop（静默，不往主聊天里打印中间文本）
    while True:
        response = client.messages.create(
            model=MODEL,
            system=sub_system,
            messages=sub_messages,
            tools=sub_tools,
            max_tokens=8000,
        )

        if response.stop_reason != "tool_use":
            break

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        results = []

        for tc in tool_calls:
            tool_count += 1
            output = execute_tool(tc.name, tc.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": output
            })

            # Update progress line (in-place)
            # 原地更新进度行
            elapsed = time.time() - start
            sys.stdout.write(
                f"\r  [{agent_type}] {description} ... {tool_count} tools, {elapsed:.1f}s"
            )
            sys.stdout.flush()

        sub_messages.append({"role": "assistant", "content": response.content})
        sub_messages.append({"role": "user", "content": results})

    # Final progress update
    # 最终进度更新
    elapsed = time.time() - start
    sys.stdout.write(
        f"\r  [{agent_type}] {description} - done ({tool_count} tools, {elapsed:.1f}s)\n"
    )

    # Extract and return only the final text
    # 只提取并返回最终文本
    # This is what the parent agent sees - a clean summary
    # 父 agent 看到的是“干净摘要”
    for block in response.content:
        if hasattr(block, "text"):
            return block.text

    return "(subagent returned no text)"


def execute_tool(name: str, args: dict) -> str:
    """Dispatch tool call to implementation.
    将工具调用分发到对应实现。
    """
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read(args["path"], args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "TodoWrite":
        return run_todo(args["items"])
    if name == "Task":
        return run_task(args["description"], args["prompt"], args["agent_type"])
    return f"Unknown tool: {name}"


# =============================================================================
# Main Agent Loop
# 主 agent loop
# =============================================================================

def agent_loop(messages: list) -> list:
    """
    Main agent loop with subagent support.
    支持 subagent 的主 agent loop。

    Same pattern as v1/v2, but now includes the Task tool.
    When model calls Task, it spawns a subagent with isolated context.
    模式与 v1/v2 相同，但新增了 Task 工具：当模型调用 Task 时，会以隔离上下文 spawn 一个 subagent。
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=ALL_TOOLS,
            max_tokens=8000,
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
            # Task 工具有特殊展示逻辑
            if tc.name == "Task":
                print(f"\n> Task: {tc.input.get('description', 'subtask')}")
            else:
                print(f"\n> {tc.name}")

            output = execute_tool(tc.name, tc.input)

            # Don't print full Task output (it manages its own display)
            # 不打印 Task 的完整输出（Task 自己管理进度展示）
            if tc.name != "Task":
                preview = output[:200] + "..." if len(output) > 200 else output
                print(f"  {preview}")

            results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": output
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# 主 REPL（交互式入口）
# =============================================================================

def main():
    print(f"Mini Claude Code v3 (with Subagents) - {WORKDIR}")
    print(f"Agent types: {', '.join(AGENT_TYPES.keys())}")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        history.append({"role": "user", "content": user_input})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
