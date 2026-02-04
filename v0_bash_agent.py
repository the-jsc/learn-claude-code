#!/usr/bin/env python
"""
v0_bash_agent.py - Mini Claude Code: Bash is All You Need (~50 lines core)
v0_bash_agent.py - 迷你 Claude Code：Bash is All You Need（核心约 50 行）

Core Philosophy: "Bash is All You Need"
======================================
核心理念："Bash is All You Need（Bash 就够了）"

This is the ULTIMATE simplification of a coding agent. After building v1-v3,
we ask: what is the ESSENCE of an agent?

The answer: ONE tool (bash) + ONE loop = FULL agent capability.
这是对 coding agent 的“极致简化”。在构建 v1-v3 之后，我们追问：agent 的本质（ESSENCE）到底是什么？

答案是：一个工具（bash）+ 一个循环（loop）= 完整的 agent 能力。

Why Bash is Enough:
------------------
为什么 Bash 就够了：

Unix philosophy says everything is a file, everything can be piped.
Bash is the gateway to this world:
Unix 哲学认为：万物皆文件，万物皆可管道（pipe）。Bash 是进入这个世界的入口：

    | You need      | Bash command                           |
    |---------------|----------------------------------------|
    | Read files    | cat, head, tail, grep                  |
    | Write files   | echo '...' > file, cat << 'EOF' > file |
    | Search        | find, grep, rg, ls                     |
    | Execute       | python, npm, make, any command         |
    | **Subagent**  | python v0_bash_agent.py "task"         |

The last line is the KEY INSIGHT: calling itself via bash implements subagents!
No Task tool, no Agent Registry - just recursion through process spawning.
上面最后一行是关键洞察（KEY INSIGHT）：通过 bash 递归调用自身，就能实现 subagent（子代理）！
不需要 Task 工具、不需要 Agent Registry（代理注册表）——只要通过进程递归（process spawning）即可。

How Subagents Work:
------------------
Subagent 如何工作：

    Main Agent
      |-- bash: python v0_bash_agent.py "analyze architecture"
           |-- Subagent (isolated process, fresh history)
                |-- bash: find . -name "*.py"
                |-- bash: cat src/main.py
                |-- Returns summary via stdout
（上面示意图保持原样；核心点是：子进程有独立上下文，最后只通过 stdout 返回摘要。）

Process isolation = Context isolation:
- Child process has its own history=[]
- Parent captures stdout as tool result
- Recursive calls enable unlimited nesting
进程隔离（process isolation）= 上下文隔离（context isolation）：
- 子进程拥有自己的 history=[]
- 父进程把 stdout 捕获为工具结果（tool result）
- 递归调用让嵌套层级几乎不受限制

Usage:
用法：
    # Interactive mode
    python v0_bash_agent.py

    # Subagent mode (called by parent agent or directly)
    python v0_bash_agent.py "explore src/ and summarize"
"""

from anthropic import Anthropic
from dotenv import load_dotenv
import subprocess
import sys
import os

load_dotenv(override=True)

# Initialize Anthropic client (uses ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL env vars)
# 初始化 Anthropic 客户端（使用 ANTHROPIC_API_KEY 与 ANTHROPIC_BASE_URL 环境变量）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-5-20250929")

# The ONE tool that does everything
# 唯一的工具：用它完成一切
# Notice how the description teaches the model common patterns AND how to spawn subagents
# 注意：description 同时教模型常见用法，并教它如何 spawn subagent（子代理）
TOOL = [{
    "name": "bash",
    "description": """Execute shell command. Common patterns:
- Read: cat/head/tail, grep/find/rg/ls, wc -l
- Write: echo 'content' > file, sed -i 's/old/new/g' file
- Subagent: python v0_bash_agent.py 'task description' (spawns isolated agent, returns summary)""",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]
    }
}]

# System prompt teaches the model HOW to use bash effectively
# system prompt 用来教模型如何高效使用 bash
# Notice the subagent guidance - this is how we get hierarchical task decomposition
# 注意 subagent 指引：这就是实现分层任务拆解（hierarchical decomposition）的方式
SYSTEM = f"""You are a CLI agent at {os.getcwd()}. Solve problems using bash commands.

Rules:
- Prefer tools over prose. Act first, explain briefly after.
- Read files: cat, grep, find, rg, ls, head, tail
- Write files: echo '...' > file, sed -i, or cat << 'EOF' > file
- Subagent: For complex subtasks, spawn a subagent to keep context clean:
  python v0_bash_agent.py "explore src/ and summarize the architecture"

When to use subagent:
- Task requires reading many files (isolate the exploration)
- Task is independent and self-contained
- You want to avoid polluting current conversation with intermediate details

The subagent runs in isolation and returns only its final summary."""


def chat(prompt, history=None):
    """
    The complete agent loop in ONE function.
    用一个函数实现完整的 agent loop。

    This is the core pattern that ALL coding agents share:
        while not done:
            response = model(messages, tools)
            if no tool calls: return
            execute tools, append results
    这是所有 coding agent 共享的核心模式：
        while not done:
            response = model(messages, tools)
            if no tool calls: return
            execute tools, append results

    Args:
        prompt: User's request
        history: Conversation history (mutable, shared across calls in interactive mode)
        参数：
            prompt：用户请求
            history：对话历史（可变；在交互模式下跨调用共享）

    Returns:
        Final text response from the model
        返回：模型最终的文本回复
    """
    if history is None:
        history = []

    history.append({"role": "user", "content": prompt})

    while True:
        # 1. Call the model with tools
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=history,
            tools=TOOL,
            max_tokens=8000
        )

        # 2. Build assistant message content (preserve both text and tool_use blocks)
        content = []
        for block in response.content:
            if hasattr(block, "text"):
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        history.append({"role": "assistant", "content": content})

        # 3. If model didn't call tools, we're done
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        # 4. Execute each tool call and collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                cmd = block.input["command"]
                print(f"\033[33m$ {cmd}\033[0m")  # Yellow color for commands
                # 命令行用黄色显示，方便区分

                try:
                    out = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=os.getcwd()
                    )
                    output = out.stdout + out.stderr
                except subprocess.TimeoutExpired:
                    output = "(timeout after 300s)"

                print(output or "(empty)")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output[:50000]  # Truncate very long outputs
                    # 截断超长输出，避免上下文爆炸
                })

        # 5. Append results and continue the loop
        # 5. 追加结果并继续循环
        history.append({"role": "user", "content": results})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Subagent mode: execute task and print result
        # subagent 模式：执行任务并打印结果
        # This is how parent agents spawn children via bash
        # 父 agent 通过 bash 这样 spawn 子进程
        print(chat(sys.argv[1]))
    else:
        # Interactive REPL mode
        # 交互式 REPL 模式
        history = []
        while True:
            try:
                query = input("\033[36m>> \033[0m")  # Cyan prompt
                # 使用青色提示符
            except (EOFError, KeyboardInterrupt):
                break
            if query in ("q", "exit", ""):
                break
            print(chat(query, history))
