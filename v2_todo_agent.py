"""
v2_todo_agent.py - Mini Claude Code: Structured Planning (~300 lines)

v1 works great for simple tasks. But ask it to "refactor auth, add tests, update docs" and watch what happens.
Without explicit planning, the model:
  - Jumps between tasks randomly
  - Forgets completed steps
  - Loses focus mid-way
v1 对简单任务很有效。但如果你让它“重构认证、加测试、更新文档”，你会看到，没有显式规划时的模型容易：
  - 在任务之间随机跳转
  - 忘记已完成的步骤
  - 做到一半失去焦点

The Problem - "Context Fade":
----------------------------
In v1, plans exist only in the model's "head":
    v1: "I'll do A, then B, then C"  (invisible)
        After 10 tool calls: "Wait, what was I doing?"

The Solution - TodoWrite Tool:
-----------------------------
v2 adds ONE new tool that fundamentally changes how the agent works:
    v2:
      [ ] Refactor auth module
      [>] Add unit tests         <- Currently working on this
      [ ] Update documentation
（todo 列表中只能有一个 in_progress，用于强制聚焦）

Now both YOU and the MODEL can see the plan. The model can:
  - Update status as it works
  - See what's done and what's next
  - Stay focused on one task at a time

Key Constraints (not arbitrary - these are guardrails):
------------------------------------------------------
关键约束（不是随意的——它们是护栏）：
    | Rule              | Why                              |
    |-------------------|----------------------------------|
    | Max 20 items      | Prevents infinite task lists     |
    | One in_progress   | Forces focus on one thing        |
    | Required fields   | Ensures structured output        |
（用“上限 + 单一进行 + 必填字段”来约束模型，换取可控的计划与进度。）

The Deep Insight:
----------------
> "Structure constrains AND enables."
> “结构既约束，也赋能。”

This pattern appears everywhere in agent design:
  - max_tokens constrains -> enables manageable responses
  - Tool schemas constrain -> enable structured calls
  - Todos constrain -> enable complex task completion
这种模式在 agent 设计中到处可见：
  - max_tokens 的约束 -> 换来可控的回复规模
  - 工具 schema 的约束 -> 换来结构化的工具调用
  - todo 约束 -> 换来复杂任务的可靠完成

Good constraints aren't limitations. They're scaffolding.
好的约束不是限制，而是脚手架。

Usage:
    python v2_todo_agent.py
"""

import os
import subprocess
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
# TodoManager - The core addition in v2
# =============================================================================
class TodoManager:
    """
    Manages a structured task list with enforced constraints.
    管理一个“有结构且受约束”的任务列表。

    Key Design Decisions:
    --------------------
    1. Max 20 items: Prevents the model from creating endless lists
    2. One in_progress: Forces focus - can only work on ONE thing at a time
    3. Required fields: Each item needs content, status, and activeForm
    关键设计决策：
    1. 最多 20 项：防止模型生成无限任务清单
    2. 只能 1 个 in_progress：强制聚焦——同一时间只做一件事
    3. 必填字段：每项都必须有 content/status/activeForm

    The activeForm field deserves explanation:
    - It's the PRESENT TENSE form of what's happening
    - Shown when status is "in_progress"
    - Example: content="Add tests", activeForm="Adding unit tests..."
    activeForm 字段需要解释一下：
    - 表示“正在做什么”的现在进行时
    - 当 status 为 in_progress 时展示
    - 例：content="Add tests"；activeForm="Adding unit tests..."

    This gives real-time visibility into what the agent is doing.
    """

    def __init__(self):
        self.tasks = []

    def update(self, tasks: list) -> str:
        """
        Validate and update the todo list.

        The model sends a complete new list each time. We validate it, store it, and return a rendered view that the model will see.
        模型每次会发来一份“完整新列表”。我们校验后保存，并返回渲染后的视图供模型查看。

        Validation Rules:
        - Each item must have: content, status, activeForm
        - Status must be: pending | in_progress | completed
        - Only ONE item can be in_progress at a time
        - Maximum 20 items allowed

        Returns:
            Rendered text view of the todo list
        """
        validated = []
        in_progress_count = 0

        for i, task in enumerate(tasks):
            # Extract and validate fields
            content = str(task.get("content", "")).strip()
            status = str(task.get("status", "pending")).lower()
            active_form = str(task.get("activeForm", "")).strip()

            # Validation checks
            if not content:
                raise ValueError(f"Task {i}: content required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Task {i}: invalid status '{status}'")
            if not active_form:
                raise ValueError(f"Task {i}: activeForm required")

            if status == "in_progress":
                in_progress_count += 1

            validated.append({"content": content, "status": status, "activeForm": active_form})

        # Enforce constraints
        if len(validated) > 20:
            raise ValueError("Max 20 todos allowed")
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.tasks = validated
        return self.render()

    def render(self) -> str:
        """
        Render the todo list as human-readable text.
        将 todo 列表渲染为人类可读的文本。

        Format:
            [x] Completed task
            [>] In progress task <- Doing something...
            [ ] Pending task

            (2/3 completed)

        This rendered text is what the model sees as the tool result.
        It can then update the list based on its current state.
        这个渲染后的文本会作为 tool_result 返回给模型，模型再根据当前状态继续更新列表。
        """
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


# Global todo manager instance
todo = TodoManager()

# =============================================================================
# System Prompt - Updated for v2
# =============================================================================
system = f"""You are a coding agent at {WORKDIR}.
Loop: plan -> act with tools -> update todos -> report.
Rules:
- Use TodoWrite to track multi-step tasks
- Mark tasks in_progress before starting, completed when done
- Prefer tools over text. Act, don't just explain.
- After finishing, summarize what changed."""

# =============================================================================
# System Reminders - Soft prompts to encourage todo usage
# 用“软提示”鼓励使用 todo
# =============================================================================
# Shown at the start of conversation
initial_reminder = "<reminder>Use TodoWrite for multi-step tasks.</reminder>"
# Shown if model hasn't updated todos in a while
# 当模型一段时间没更新 todo 时展示
nag_reminder = "<reminder>10+ turns without todo update. Please update todos.</reminder>"

# =============================================================================
# Tool Definitions (v1 tools + TodoWrite)
# =============================================================================
tools = [
    # v1 tools (unchanged)
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
    # NEW in v2: TodoWrite
    # This is the key addition that enables structured planning
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
# Tool Implementations (v1 + TodoWrite)
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
    """
    Update the todo list.

    The model sends a complete new list (not a diff).
    We validate it and return the rendered view.
    模型发送的是“完整新列表”（不是 diff）。我们校验后返回渲染结果。
    """
    try:
        return todo.update(tasks)
    except Exception as e:
        return f"Error: {e}"


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
    return f"Unknown tool: {name}"


# =============================================================================
# Agent Loop (with todo tracking)
# =============================================================================
# Track how many rounds since last todo update
# 距离上次 todo 更新已经过去多少轮
rounds_without_todo = 0


def agent_loop(messages: list) -> list:
    """
    Agent loop with todo usage tracking.

    Same core loop as v1, but now we track whether the model is using todos.
    If it goes too long without updating, we inject a reminder into the next user message (tool results).
    核心循环与 v1 相同，但这里会追踪模型是否在使用 todo。
    如果太久没更新，就把提醒注入到下一条 user message（tool results）里。
    """
    global rounds_without_todo

    while True:
        response = client.messages.create(
            model=model, system=system, messages=messages, tools=tools, max_tokens=max_tokens
        )

        tool_calls = []
        for block in response.content:
            if block.type == "text":
                print(block.text)
            if block.type == "tool_use":
                tool_calls.append(block)

        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            return messages

        results = []
        used_todo = False
        for tc in tool_calls:
            print(f"\n> {tc.name}")
            output = execute_tool(tc.name, tc.input)
            preview = output[:300] + "..." if len(output) > 300 else output
            print(f"  {preview}")

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": output,
                }
            )

            # Track todo usage
            if tc.name == "TodoWrite":
                used_todo = True

        # Update counter: reset if used todo, increment otherwise
        if used_todo:
            rounds_without_todo = 0
        else:
            rounds_without_todo += 1

        messages.append({"role": "assistant", "content": response.content})

        # Inject NAG_REMINDER into user message if model hasn't used todos
        # 如果模型很久没用 todo，就把 NAG_REMINDER 注入到 user message 中
        # This happens INSIDE the agent loop, so model sees it during task execution
        # 这发生在 agent loop 内部，因此模型在执行任务时能看到提醒
        if rounds_without_todo > 10:
            results.insert(0, {"type": "text", "text": nag_reminder})

        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# =============================================================================
def main():
    """
    REPL with reminder injection.

    Key v2 addition: We inject "reminder" messages to encourage todo usage without forcing it. This is a soft constraint.
    v2 的关键增量：注入 reminder 文本来鼓励使用 todo，但不强制，这是“软约束”。
    """
    global rounds_without_todo

    print(f"Mini Claude Code v2 (with Todos) - {WORKDIR}")

    history = []
    first_message = True

    while True:
        try:
            user_prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_prompt or user_prompt.lower() in ("exit", "quit"):
            break

        # Build user message content
        content = []

        if first_message:
            # Gentle reminder at start of conversation
            content.append({"type": "text", "text": initial_reminder})
            first_message = False

        content.append({"type": "text", "text": user_prompt})
        history.append({"role": "user", "content": content})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
