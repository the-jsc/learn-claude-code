"""
v1_basic_agent.py - Mini Claude Code: Model as Agent (~200 lines)
v1_basic_agent.py - 迷你 Claude Code：Model as Agent（模型即代理，约 200 行）

Core Philosophy: "The Model IS the Agent"
=========================================
核心理念："The Model IS the Agent（模型就是代理）"

The secret of Claude Code, Cursor Agent, Codex CLI? There is no secret.
Claude Code、Cursor Agent、Codex CLI 的“秘密”？其实没有秘密。

Strip away the CLI polish, progress bars, permission systems. What remains
is surprisingly simple: a LOOP that lets the model call tools until done.
把 CLI 的外壳、进度条、权限系统这些都剥离掉，剩下的东西出奇地简单：一个循环（LOOP），让模型反复调用工具直到完成任务。

Traditional Assistant:
    User -> Model -> Text Response
（传统助手：用户 -> 模型 -> 纯文本回复）

Agent System:
    User -> Model -> [Tool -> Result]* -> Response
                          ^________|
（Agent 系统：用户 -> 模型 -> [工具 -> 结果]* -> 回复；星号表示可重复多次）

The asterisk (*) matters! The model calls tools REPEATEDLY until it decides
the task is complete. This transforms a chatbot into an autonomous agent.
这个星号（*）很关键：模型会反复调用工具，直到它认为任务完成。这样就把聊天机器人变成了“自治”的 agent。

KEY INSIGHT: The model is the decision-maker. Code just provides tools and
runs the loop. The model decides:
  - Which tools to call
  - In what order
  - When to stop
关键洞察（KEY INSIGHT）：模型才是决策者。代码只负责提供工具并运行循环；模型决定：
  - 调用哪些工具
  - 以什么顺序调用
  - 何时停止

The Four Essential Tools:
------------------------
四个核心工具：

Claude Code has ~20 tools. But these 4 cover 90% of use cases:
Claude Code 大约有 20 个工具，但下面这 4 个就能覆盖 90% 的场景：

    | Tool       | Purpose              | Example                    |
    |------------|----------------------|----------------------------|
    | bash       | Run any command      | npm install, git status    |
    | read_file  | Read file contents   | View src/index.ts          |
    | write_file | Create/overwrite     | Create README.md           |
    | edit_file  | Surgical changes     | Replace a function         |

With just these 4 tools, the model can:
  - Explore codebases (bash: find, grep, ls)
  - Understand code (read_file)
  - Make changes (write_file, edit_file)
  - Run anything (bash: python, npm, make)
只靠这 4 个工具，模型就能：
  - 探索代码库（bash：find/grep/ls）
  - 理解代码（read_file）
  - 修改代码（write_file / edit_file）
  - 运行任何命令（bash：python/npm/make）

Usage:
    python v1_basic_agent.py
用法：
    python v1_basic_agent.py
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
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))


# =============================================================================
# System Prompt - The only "configuration" the model needs
# System Prompt（系统提示词）——模型唯一需要的“配置”
# =============================================================================

SYSTEM = f"""You are a coding agent at {WORKDIR}.

Loop: think briefly -> use tools -> report results.

Rules:
- Prefer tools over prose. Act, don't just explain.
- Never invent file paths. Use bash ls/find first if unsure.
- Make minimal changes. Don't over-engineer.
- After finishing, summarize what changed."""


# =============================================================================
# Tool Definitions - 4 tools cover 90% of coding tasks
# 工具定义：4 个工具覆盖 90% 的 coding 任务
# =============================================================================

TOOLS = [
    # Tool 1: Bash - The gateway to everything
    # 工具 1：Bash——通往一切的入口
    # Can run any command: git, npm, python, curl, etc.
    # 可以运行任何命令：git/npm/python/curl 等
    {
        "name": "bash",
        "description": "Run a shell command. Use for: ls, find, grep, git, npm, python, etc.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to execute"}},
            "required": ["command"],
        },
    },
    # Tool 2: Read File - For understanding existing code
    # 工具 2：Read File——用于理解现有代码
    # Returns file content with optional line limit for large files
    # 支持大文件行数限制（可选）
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
    # Tool 3: Write File - For creating new files or complete rewrites
    # 工具 3：Write File——用于创建新文件或整体重写
    # Creates parent directories automatically
    # 自动创建父目录
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
    # Tool 4: Edit File - For surgical changes to existing code
    # 工具 4：Edit File——对现有代码做“手术式”的精确修改
    # Uses exact string matching for precise edits
    # 使用精确字符串匹配，确保修改可控
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
]


# =============================================================================
# Tool Implementations
# 工具实现
# =============================================================================


def safe_path(p: str) -> Path:
    """
    Ensure path stays within workspace (security measure).
    确保路径不会逃逸出工作区（安全措施）。

    Prevents the model from accessing files outside the project directory.
    Resolves relative paths and checks they don't escape via '../'.
    防止模型访问项目目录之外的文件：解析相对路径，并检查不会通过 `../` 逃逸。
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    Execute shell command with safety checks.
    执行 shell 命令，并做基础安全检查。

    Security: Blocks obviously dangerous commands.
    Timeout: 60 seconds to prevent hanging.
    Output: Truncated to 50KB to prevent context overflow.
    - 安全：拦截明显危险的命令
    - 超时：60 秒，避免卡死
    - 输出：截断到 50KB，避免上下文溢出
    """
    # Basic safety - block dangerous patterns
    # 基础安全：拦截危险模式
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
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
    """
    Read file contents with optional line limit.
    读取文件内容（可选行数限制）。

    For large files, use limit to read just the first N lines.
    Output truncated to 50KB to prevent context overflow.
    对大文件可用 limit 只读取前 N 行；输出截断到 50KB，避免上下文溢出。
    """
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
    """
    Write content to file, creating parent directories if needed.
    写入文件内容（必要时自动创建父目录）。

    This is for complete file creation/overwrite.
    For partial edits, use edit_file instead.
    用于创建/覆盖整个文件；局部修改请使用 edit_file。
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    Replace exact text in a file (surgical edit).
    在文件中替换一段精确匹配的文本（手术式编辑）。

    Uses exact string matching - the old_text must appear verbatim.
    Only replaces the first occurrence to prevent accidental mass changes.
    使用精确字符串匹配（old_text 必须原样出现）；只替换第一次出现，避免误伤大范围内容。
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()

        if old_text not in content:
            return f"Error: Text not found in {path}"

        # Replace only first occurrence for safety
        # 为安全起见，只替换第一次出现
        new_content = content.replace(old_text, new_text, 1)
        fp.write_text(new_content)
        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


def execute_tool(name: str, args: dict) -> str:
    """
    Dispatch tool call to the appropriate implementation.
    将工具调用分发到对应的实现函数。

    This is the bridge between the model's tool calls and actual execution.
    Each tool returns a string result that goes back to the model.
    这是模型“调用工具”和真实执行之间的桥梁；每个工具返回字符串结果并回传给模型。
    """
    if name == "bash":
        return run_bash(args["command"])
    if name == "read_file":
        return run_read(args["path"], args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    return f"Unknown tool: {name}"


# =============================================================================
# The Agent Loop - This is the CORE of everything
# Agent Loop（代理循环）——一切的核心
# =============================================================================


def agent_loop(messages: list) -> list:
    """
    The complete agent in one function.
    用一个函数实现完整的 agent。

    This is the pattern that ALL coding agents share:

        while True:
            response = model(messages, tools)
            if no tool calls: return
            execute tools, append results, continue

    The model controls the loop:
      - Keeps calling tools until stop_reason != "tool_use"
      - Results become context (fed back as "user" messages)
      - Memory is automatic (messages list accumulates history)

    Why this works:
      1. Model decides which tools, in what order, when to stop
      2. Tool results provide feedback for next decision
      3. Conversation history maintains context across turns
    所有 coding agent 共享的模式就是这样：在循环里不断“模型 -> 工具 -> 结果”，直到模型停止调用工具。

    为什么它能工作：
      1. 模型负责决策：调用什么工具、什么顺序、何时停止
      2. 工具结果为下一步决策提供反馈
      3. messages 历史自然累积，形成跨轮次记忆
    """
    while True:
        # Step 1: Call the model
        # 步骤 1：调用模型
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )

        # Step 2: Collect any tool calls and print text output
        # 步骤 2：收集工具调用，并打印文本输出
        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
            if block.type == "tool_use":
                tool_calls.append(block)

        # Step 3: If no tool calls, task is complete
        # 步骤 3：如果没有工具调用，任务完成
        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            return messages

        # Step 4: Execute each tool and collect results
        # 步骤 4：执行工具并收集结果
        results = []
        for tc in tool_calls:
            # Display what's being executed
            # 显示正在执行的内容
            print(f"\n> {tc.name}: {tc.input}")

            # Execute and show result preview
            # 执行并展示结果预览
            output = execute_tool(tc.name, tc.input)
            preview = output[:200] + "..." if len(output) > 200 else output
            print(f"  {preview}")

            # Collect result for the model
            # 收集结果并回传给模型
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": output,
                }
            )

        # Step 5: Append to conversation and continue
        # 步骤 5：追加到对话历史并继续
        # Note: We append assistant's response, then user's tool results
        # 注意：先追加 assistant 的响应，再追加 user 的 tool_result
        # This maintains the alternating user/assistant pattern
        # 这样可以保持 user/assistant 交替结构
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# 主 REPL（交互式入口）
# =============================================================================


def main():
    """
    Simple Read-Eval-Print Loop for interactive use.
    简单的交互式 Read-Eval-Print Loop（REPL）。

    The history list maintains conversation context across turns,
    allowing multi-turn conversations with memory.
    history 列表会在多轮对话中维持上下文，从而具备“记忆”效果。
    """
    print(f"Mini Claude Code v1 - {WORKDIR}")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        # Add user message to history
        # 将用户消息加入历史
        history.append({"role": "user", "content": user_input})

        try:
            # Run the agent loop
            # 运行 agent loop
            agent_loop(history)
        except Exception as e:
            print(f"Error: {e}")

        print()
        # Blank line between turns
        # 轮次之间输出空行


if __name__ == "__main__":
    main()
