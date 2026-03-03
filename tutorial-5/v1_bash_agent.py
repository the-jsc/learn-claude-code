"""
v1_bash_agent.py - Mini Claude Code: Bash is All You Need (~50 lines core)

This is the ULTIMATE simplification of a coding agent.
After building v1-v3, we ask: what is the ESSENCE of an agent?
The answer: ONE tool (bash) + ONE loop = FULL agent capability.

Why Bash is Enough:
Unix philosophy says everything is a file, everything can be piped.
Bash is the gateway to this world:
Unix 哲学认为：万物皆文件，万物皆可管道（pipe）。
Bash 是进入这个世界的入口：

    | You need      | Bash command                           |
    |---------------|----------------------------------------|
    | Read files    | cat, head, tail, grep
    | Write files   | echo '...' > file, cat << 'EOF' > file |
    | Search        | find, grep, rg, ls                     |
    | Execute       | python, npm, make, any command         |
    | **Subagent**  | python v1_bash_agent.py "task"

The last line is the KEY INSIGHT: calling itself via bash implements subagents!
No Task tool, no Agent Registry - just recursion through process spawning.
上面最后一行是关键：通过 bash 递归调用自身，就能实现 subagent！
不需要 Task 工具、不需要 Agent Registry ——只要通过进程递归即可。

How Subagents Work:
    Main Agent
      |-- bash: python v1_bash_agent.py "analyze architecture"
           |-- Subagent (isolated process, fresh history)
                |-- bash: find . -name "*.py"
                |-- bash: cat src/main.py
                |-- Returns summary via stdout
（子进程有独立上下文，最后通过 stdout 返回摘要。）

Process isolation = Context isolation:
- Child process has its own history=[]
- Parent captures stdout as tool result
- Recursive calls enable unlimited nesting
进程隔离= 上下文隔离：
- 子进程拥有自己的 history=[]
- 父进程把 stdout 捕获为工具结果
- 递归调用让嵌套层级几乎不受限制

Usage:
    # Interactive mode
    python v1_bash_agent.py

    # Subagent mode (called by parent agent or directly)
    python v1_bash_agent.py "explore src/ and summarize"
"""

from anthropic import Anthropic
from dotenv import load_dotenv
import subprocess
import sys
import os

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=os.getenv("ANTHROPIC_BASE_URL"))
model = os.getenv("MODEL")
max_tokens = int(os.getenv("MAX_TOKENS"))

# The ONE tool that does everything
# Notice how the description teaches the model common patterns AND how to spawn subagents
# 注意：description 教模型常见用法，并教它如何生成子代理
tools = [
    {
        "name": "bash",
        "description": """Execute shell command. Common patterns:
- Read: cat/head/tail, grep/find/rg/ls, wc -l
- Write: echo 'content' > file, sed -i 's/old/new/g' file
- Subagent: python v1_bash_agent.py 'task description' (spawns isolated agent, returns summary)""",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    }
]

# System prompt teaches the model HOW to use bash effectively
# Notice the subagent guidance - this is how we get hierarchical task decomposition
# 注意 subagent 指引：这是实现分层任务拆解的方式
system = f"""You are a CLI agent at {os.getcwd()}. Solve problems using bash commands.

Rules:
- Prefer tools over prose. Act first, explain briefly after.
- Read files: cat, grep, find, rg, ls, head, tail
- Write files: echo '...' > file, sed -i, or cat << 'EOF' > file
- Subagent: For complex subtasks, spawn a subagent to keep context clean: python v1_bash_agent.py "explore src/ and summarize the architecture"

When to use subagent:
- Task requires reading many files (isolate the exploration)
- Task is independent and self-contained
- You want to avoid polluting current conversation with intermediate details

The subagent runs in isolation and returns only its final summary.

Communicate with the user in Chinese."""


def chat(prompt, history=None):
    """
    The complete agent loop in ONE function.
    This is the core pattern that ALL coding agents share:
        while not done:
            response = model(messages, tools)
            if no tool calls: return
            execute tools, append results

    Args:
        prompt: User's request
        history: Conversation history (mutable, shared across calls in interactive mode)

    Returns:
        Final text response from the model
    """
    if history is None:
        history = []
    history.append({"role": "user", "content": prompt})

    while True:
        # 1. Call the model with tools
        response = client.messages.create(model=model, system=system, messages=history, tools=tools, max_tokens=max_tokens)

        # 2. Build assistant message content
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        history.append({"role": "assistant", "content": content})

        # 3. If model didn't call tools, we're done
        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        # 4. Execute each tool call and collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                cmd = block.input["command"]
                print(f"\033[33m$ {cmd}\033[0m")  # Yellow color for command

                try:
                    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300, cwd=os.getcwd())
                    output = out.stdout + out.stderr
                except subprocess.TimeoutExpired:
                    output = "(timeout after 300s)"

                print(output or "(empty)")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output[:51200],  # Truncate very long outputs
                    }
                )

        # 5. Append results and continue the loop
        history.append({"role": "user", "content": results})


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Subagent mode: execute task and print result
        # This is how parent agents spawn children via bash
        print(chat(sys.argv[1]))
    else:
        # Interactive REPL mode
        history = []
        while True:
            try:
                user_prompt = input("\033[36m>> \033[0m")  # Cyan color for prompt
            except (EOFError, KeyboardInterrupt):
                break
            if user_prompt in ("quit", "exit"):
                break
            print(chat(user_prompt, history))
