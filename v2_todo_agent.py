"""
v2_todo_agent.py - Mini Claude Code: Structured Planning (~300 lines)
v2_todo_agent.py - 迷你 Claude Code：Structured Planning（结构化规划，约 300 行）

Core Philosophy: "Make Plans Visible"
=====================================
核心理念："Make Plans Visible（让计划可见）"

v1 works great for simple tasks. But ask it to "refactor auth, add tests,
update docs" and watch what happens. Without explicit planning, the model:
  - Jumps between tasks randomly
  - Forgets completed steps
  - Loses focus mid-way
v1 对简单任务很有效。但如果你让它“重构 auth、加测试、更新文档”，你会看到：没有显式规划时，模型容易：
  - 在任务之间随机跳转
  - 忘记已完成的步骤
  - 做到一半失去焦点

The Problem - "Context Fade":
----------------------------
问题：“Context Fade（上下文褪色）”

In v1, plans exist only in the model's "head":
在 v1 里，计划只存在于模型的“脑海”里：

    v1: "I'll do A, then B, then C"  (invisible)
        After 10 tool calls: "Wait, what was I doing?"
（计划不可见；经过多次工具调用后，很容易忘记自己在做什么。）

The Solution - TodoWrite Tool:
-----------------------------
解决方案：TodoWrite 工具

v2 adds ONE new tool that fundamentally changes how the agent works:
v2 新增一个工具（TodoWrite），但它会从根本上改变 agent 的工作方式：

    v2:
      [ ] Refactor auth module
      [>] Add unit tests         <- Currently working on this
      [ ] Update documentation
（示例：todo 列表中只能有一个 in_progress，用于强制聚焦）

Now both YOU and the MODEL can see the plan. The model can:
  - Update status as it works
  - See what's done and what's next
  - Stay focused on one task at a time
现在你和模型都能看到计划，模型可以：
  - 工作时实时更新状态
  - 清楚已完成与下一步
  - 始终聚焦在同一件事上

Key Constraints (not arbitrary - these are guardrails):
------------------------------------------------------
关键约束（不是随意的——它们是护栏）：

    | Rule              | Why                              |
    |-------------------|----------------------------------|
    | Max 20 items      | Prevents infinite task lists     |
    | One in_progress   | Forces focus on one thing        |
    | Required fields   | Ensures structured output        |
（上表保持原样；核心意思是：用“上限 + 单一进行中 + 必填字段”来约束模型，从而换取可控的计划与进度。）

The Deep Insight:
----------------
> "Structure constrains AND enables."
深层洞察：
> “结构既约束，也赋能。”

Todo constraints (max items, one in_progress) ENABLE (visible plan, tracked progress).
Todo 的约束（数量上限、唯一 in_progress）反而“赋能”了：计划可见、进度可追踪。

This pattern appears everywhere in agent design:
  - max_tokens constrains -> enables manageable responses
  - Tool schemas constrain -> enable structured calls
  - Todos constrain -> enable complex task completion
这种模式在 agent 设计中到处可见：
  - max_tokens 的约束 -> 换来可控的回复规模
  - 工具 schema 的约束 -> 换来结构化的工具调用
  - todo 约束 -> 换来复杂任务的可靠完成

Good constraints aren't limitations. They're scaffolding.
好的约束不是限制，而是脚手架（scaffolding）。

Usage:
    python v2_todo_agent.py
用法：
    python v2_todo_agent.py
"""

import os
import subprocess
import sys
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
# TodoManager - The core addition in v2
# TodoManager——v2 的核心增量
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
    这样就能实时看到 agent 在做什么。
    """

    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        """
        Validate and update the todo list.
        校验并更新 todo 列表。

        The model sends a complete new list each time. We validate it,
        store it, and return a rendered view that the model will see.
        模型每次会发来一份“完整新列表”（不是 diff）。我们校验后保存，并返回渲染后的视图供模型查看。

        Validation Rules:
        - Each item must have: content, status, activeForm
        - Status must be: pending | in_progress | completed
        - Only ONE item can be in_progress at a time
        - Maximum 20 items allowed
        校验规则：
        - 每项必须包含：content/status/activeForm
        - status 只能是：pending | in_progress | completed
        - 同一时间只能有 1 项是 in_progress
        - 总数最多 20 项

        Returns:
            Rendered text view of the todo list
            返回：todo 列表的渲染文本
        """
        validated = []
        in_progress_count = 0

        for i, item in enumerate(items):
            # Extract and validate fields
            # 提取并校验字段
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            active_form = str(item.get("activeForm", "")).strip()

            # Validation checks
            # 校验检查
            if not content:
                raise ValueError(f"Item {i}: content required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status '{status}'")
            if not active_form:
                raise ValueError(f"Item {i}: activeForm required")

            if status == "in_progress":
                in_progress_count += 1

            validated.append({"content": content, "status": status, "activeForm": active_form})

        # Enforce constraints
        # 强制约束
        if len(validated) > 20:
            raise ValueError("Max 20 todos allowed")
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
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
        格式：
            [x] 已完成
            [>] 进行中 <- 正在做什么...
            [ ] 待办

            (2/3 completed)

        This rendered text is what the model sees as the tool result.
        It can then update the list based on its current state.
        这个渲染后的文本会作为 tool_result 返回给模型，模型再根据当前状态继续更新列表。
        """
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            if item["status"] == "completed":
                lines.append(f"[x] {item['content']}")
            elif item["status"] == "in_progress":
                lines.append(f"[>] {item['content']} <- {item['activeForm']}")
            else:
                lines.append(f"[ ] {item['content']}")

        completed = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({completed}/{len(self.items)} completed)")

        return "\n".join(lines)


# Global todo manager instance
# 全局 TodoManager 实例
TODO = TodoManager()


# =============================================================================
# System Prompt - Updated for v2
# System Prompt——v2 版本（包含 todo 的工作流）
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> update todos -> report.

Rules:
- Use TodoWrite to track multi-step tasks
- Mark tasks in_progress before starting, completed when done
- Prefer tools over prose. Act, don't just explain.
- After finishing, summarize what changed."""


# =============================================================================
# System Reminders - Soft prompts to encourage todo usage
# 系统提醒：用“软提示”鼓励使用 todo
# =============================================================================

# Shown at the start of conversation
# 在对话开始时展示
INITIAL_REMINDER = "<reminder>Use TodoWrite for multi-step tasks.</reminder>"

# Shown if model hasn't updated todos in a while
# 当模型一段时间没更新 todo 时展示
NAG_REMINDER = "<reminder>10+ turns without todo update. Please update todos.</reminder>"


# =============================================================================
# Tool Definitions (v1 tools + TodoWrite)
# 工具定义（继承 v1 工具 + TodoWrite）
# =============================================================================

TOOLS = [
    # v1 tools (unchanged)
    # v1 工具（保持不变）
    {
        "name": "bash",
        "description": "Run a shell command.",
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
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
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
    # NEW in v2: TodoWrite
    # v2 新增：TodoWrite
    # This is the key addition that enables structured planning
    # 这是实现“结构化规划”的关键增量
    {
        "name": "TodoWrite",
        "description": "Update the task list. Use to plan and track progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Complete list of tasks (replaces existing)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Task description"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Task status",
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
            "required": ["items"],
        },
    },
]


# =============================================================================
# Tool Implementations (v1 + TodoWrite)
# 工具实现（v1 + TodoWrite）
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
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in cmd for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(cmd, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=60)
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """Read file contents.
    读取文件内容。
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(text.splitlines()) - limit} more)"]
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
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_todo(items: list) -> str:
    """
    Update the todo list.
    更新 todo 列表。

    The model sends a complete new list (not a diff).
    We validate it and return the rendered view.
    模型发送的是“完整新列表”（不是 diff）。我们校验后返回渲染结果。
    """
    try:
        return TODO.update(items)
    except Exception as e:
        return f"Error: {e}"


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
    return f"Unknown tool: {name}"


# =============================================================================
# Agent Loop (with todo tracking)
# Agent Loop（带 todo 使用追踪）
# =============================================================================

# Track how many rounds since last todo update
# 距离上次 todo 更新已经过去多少轮
rounds_without_todo = 0


def agent_loop(messages: list) -> list:
    """
    Agent loop with todo usage tracking.
    带 todo 使用追踪的 agent loop。

    Same core loop as v1, but now we track whether the model
    is using todos. If it goes too long without updating,
    we inject a reminder into the next user message (tool results).
    核心循环与 v1 相同，但这里会追踪模型是否在使用 todo。
    如果太久没更新，就把提醒注入到下一条 user message（tool results）里。
    """
    global rounds_without_todo

    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
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
            # 追踪是否使用了 TodoWrite
            if tc.name == "TodoWrite":
                used_todo = True

        # Update counter: reset if used todo, increment otherwise
        # 更新计数器：用了 todo 就重置，否则递增
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
            results.insert(0, {"type": "text", "text": NAG_REMINDER})

        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# 主 REPL（交互式入口）
# =============================================================================


def main():
    """
    REPL with reminder injection.
    带提醒注入的 REPL。

    Key v2 addition: We inject "reminder" messages to encourage
    todo usage without forcing it. This is a soft constraint.
    v2 的关键增量：注入 reminder 文本来鼓励使用 todo，但不强制，这是“软约束”。

    - INITIAL_REMINDER: injected at conversation start
    - NAG_REMINDER: injected inside agent_loop when 10+ rounds without todo
    - INITIAL_REMINDER：对话开始时注入
    - NAG_REMINDER：在 agent_loop 内部，当 10+ 轮没更新 todo 时注入
    """
    global rounds_without_todo

    print(f"Mini Claude Code v2 (with Todos) - {WORKDIR}")
    print("Type 'exit' to quit.\n")

    history = []
    first_message = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        # Build user message content
        # 构造 user message 内容（支持注入 reminder）
        content = []

        if first_message:
            # Gentle reminder at start of conversation
            # 对话开始时的温和提醒
            content.append({"type": "text", "text": INITIAL_REMINDER})
            first_message = False

        content.append({"type": "text", "text": user_input})
        history.append({"role": "user", "content": content})

        try:
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
