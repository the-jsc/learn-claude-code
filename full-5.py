import os
import sys
import re
import time
import subprocess
from pathlib import Path
from typing import Optional
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
WORK_DIR = Path.cwd()
SKILLS_DIR = WORK_DIR / "skills"
CLIENT = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL")
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))


# =============================================================================
# SkillsLoader
# =============================================================================
class SkillsLoader:
    """
    Load and manage skills from SKILL.md files.

    SKILL.md Format:
    ----------------
        ---
        name: pdf
        description: Process PDF files. Use when reading, creating, or merging PDFs.
        ---
        ...

    The YAML frontmatter provides metadata (name, description).
    The markdown body provides detailed instructions.
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self.load_skills()

    def parse_skill_md(self, path: Path) -> Optional[dict]:
        content = path.read_text()

        # 捕获组1: metadata
        # 捕获组2: body
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if match:
            frontmatter, body = match.groups()
            metadata = {}
            for line in frontmatter.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
            if "name" in metadata and "description" in metadata:
                return {
                    "name": metadata["name"],
                    "description": metadata["description"],
                    "body": body.strip(),
                    "path": path,
                    "dir": path.parent,
                }

    def load_skills(self):
        """
        Scan skills directory and load all valid SKILL.md files.
        Only loads metadata at startup - body is loaded on-demand.
        """
        if self.skills_dir.exists():
            for skill_dir in self.skills_dir.iterdir():
                if skill_dir.is_dir():
                    skill_md = skill_dir / "SKILL.md"
                    if skill_md.exists():
                        skill = self.parse_skill_md(skill_md)
                        if skill:
                            self.skills[skill["name"]] = skill

    def get_skill_info(self) -> str:
        """
        This is Layer 1 - only name and description.
        """
        if not self.skills:
            return "(no skills available)"
        else:
            return "\n".join(f"- {name}: {skill['description']}" for name, skill in self.skills.items())

    def get_skill_content(self, name: str) -> Optional[str]:
        """
        This is Layer 2 - the complete SKILL.md body, plus any available resources (Layer 3 hints).
        """
        if name in self.skills.keys():
            skill = self.skills[name]
            content = f"# Skill: {skill['name']}\n\n{skill['body']}"
            resources = []
            for folder, label in [("scripts", "Scripts"), ("references", "References"), ("assets", "Assets")]:
                folder_path = skill["dir"] / folder
                if folder_path.exists():
                    files = list(folder_path.glob("*"))
                    if files:
                        resources.append(f"{label}: {', '.join(file.name for file in files)}")
            if resources:
                content += f"\n\n**Available resources in {skill['dir']}:**\n"
                content += "\n".join(f"- {resource}" for resource in resources)
            return content

    def list_skills(self) -> str:
        return list(self.skills.keys())


SKILLS = SkillsLoader(SKILLS_DIR)

# =============================================================================
# subAgent Registry
# =============================================================================
SUBAGENT = {
    "explore": {
        "description": "Read-only agent for searching and analyzing, like exploring code、finding files",
        "tools": ["bash", "read_file"],
        "prompt": "You are an exploration agent. Search and analyze, but never modify files. Return a concise summary.",
    },
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["bash", "read_file"],
        "prompt": "You are a planning agent. Analyze and output a numbered implementation plan. Do NOT make changes.",
    },
    "code": {
        "description": "Full-powered agent for implementing features and fixing bugs",
        "tools": "*",
        "prompt": "You are a coding agent. Implement the requested changes efficiently.",
    },
}


def get_subagent_info() -> str:
    return "\n".join(f"- {name}: {detail['description']}" for name, detail in SUBAGENT.items())


# =============================================================================
# System Prompt
# =============================================================================
SYSTEM = f"""You are a CLI agent at {WORK_DIR}.

Loop: plan -> act with tools -> report.

**Skills available**:
{SKILLS.get_skill_info()}

**Subagents available**:
{get_subagent_info()}

**Rules**:
- Communicate in Chinese.
- Use manage_todo tool to track multi-step tasks.
- Use acquire_skill tool when a task matches a skill description.
- Use assign_task tool for subtasks that need focused exploration or implementation.
- Prefer tools over text. Act, don't just explain.
- Never invent file paths. Use bash ls/find first if unsure.
- After finishing, summarize what changed."""


# =============================================================================
# TodoManager
# =============================================================================
class TodoManager:
    def __init__(self):
        self.todos = []
        self.status = ["pending", "in_progress", "completed"]  # 未完成，进行中，已完成

    def update(self, todos: list) -> str:
        """
        大模型每次都会生成完整任务列表，我们对此进行检验并美化输出。
        """
        if len(todos) > 10:
            raise ValueError("Max 10 todos allowed")

        validated = []
        in_progress_count = 0
        for i, todo in enumerate(todos):
            # 任务名称、状态、内容、结果
            name = str(todo.get("name", "")).strip()
            status = str(todo.get("status", "pending")).lower()
            content = str(todo.get("content", "")).strip()
            result = str(todo.get("result", "")).strip()

            if not name:
                raise ValueError(f"Todo {i}: name required")
            if status not in self.status:
                raise ValueError(f"Todo {i}: invalid status '{status}'")
            if not content:
                raise ValueError(f"Todo {i}: content required")
            if not result:
                raise ValueError(f"Todo {i}: result required")

            if status == "in_progress":
                in_progress_count += 1

            validated.append({"name": name, "status": status, "content": content, "result": result})

        if in_progress_count > 1:
            raise ValueError("Only one todo can be in_progress at a time")

        self.todos = validated
        return self.render()

    def render(self) -> str:
        """
        将任务列表渲染为文本输出，并作为工具结果返回。

        Format:
            [x] Completed task: result...
            [>] In progress task: content...
            [ ] Pending task

            (2/3 completed)
        """
        if not self.todos:
            return "No todos."

        lines = []
        for todo in self.todos:
            if todo["status"] == "completed":
                lines.append(f"[x] {todo['name']}: {todo['result']}")
            elif todo["status"] == "in_progress":
                lines.append(f"[>] {todo['name']}: {todo['content']}")
            else:
                lines.append(f"[ ] {todo['name']}")
        completed = sum(1 for todo in self.todos if todo["status"] == "completed")
        lines.append(f"\n({completed}/{len(self.todos)} completed)")
        return "\n".join(lines)


TODO = TodoManager()

# =============================================================================
# Tool Definition
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
        "name": "manage_todo",
        "description": "Use to plan and track progress by update the todo list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Complete list of todos (replaces existing). Max 10 todos allowed.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Todo name"},
                            "status": {
                                "type": "string",
                                "description": "Todo status",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "content": {
                                "type": "string",
                                "description": "Present tense action, e.g. 'Reading files'",
                            },
                            "result": {"type": "string", "description": "Todo result"},
                        },
                        "required": ["name", "status", "content", "result"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
]

SUBAGENT_TOOL = {
    "name": "assign_task",
    "description": f"""Spawn a subagent for a focused subtask.
Subagents run in ISOLATED context - they don't see parent's history.
Use this to keep the main conversation clean.
**Agent types**:
{get_subagent_info()}
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Short task name for progress display"},
            "prompt": {"type": "string", "description": "Detailed instructions for the subagent"},
            "agent_type": {"type": "string", "enum": list(SUBAGENT.keys()), "description": "Type of agent to spawn"},
        },
        "required": ["task", "prompt", "agent_type"],
    },
}

SKILL_TOOL = {
    "name": "acquire_skill",
    "description": f"""Load a skill to gain specialized knowledge for a task.
The skill content will be injected into the conversation, giving you detailed instructions and access to resources.
**Available skills**:
{SKILLS.get_skill_info()}
""",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Name of the skill to load"},
        },
        "required": ["skill"],
    },
}

TOOLS = BASE_TOOLS + [SUBAGENT_TOOL, SKILL_TOOL]


def get_subagent_tools(agent_type: str) -> list:
    """
    禁止subagent使用assign_task，防止无限递归
    """
    allowed_tools = SUBAGENT.get(agent_type, {}).get("tools", "*")
    if allowed_tools == "*":
        return BASE_TOOLS
    else:
        return [tool for tool in TOOLS if tool["name"] != "assign_task" and tool["name"] in allowed_tools]


# =============================================================================
# Tool Implementation
# =============================================================================
def safe_path(p: str) -> Path:
    """
    检查路径是否在项目目录中，防止模型访问项目目录之外的文件。
    """
    path = (WORK_DIR / p).resolve()  # 绝对路径
    if not path.is_relative_to(WORK_DIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    检查命令安全性，拦截危险命令；超时限制60秒；输出截断至50KB、避免上下文溢出
    """
    dangerous = ["rm -rf", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(command, shell=True, cwd=WORK_DIR, capture_output=True, text=True, timeout=60)
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """
    读取文件时可限定行数；输出截断至50KB，避免上下文溢出。
    """
    try:
        texts = safe_path(path).read_text().splitlines()
        if limit and limit < len(texts):
            texts = texts[:limit]
            texts.append(f"... ({len(texts) - limit} more lines)")
        return "\n".join(texts)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    用于创建/覆盖整个文件，自动检测并创建上级目录。
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
    使用精确字符串匹配原文；只替换第一次出现，避免误伤大范围内容。
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        else:
            new_content = content.replace(old_text, new_text, 1)
            fp.write_text(new_content)
            return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_todo(todos: list) -> str:
    try:
        return TODO.update(todos)
    except Exception as e:
        return f"Error: {e}"


def run_skill(skill: str) -> str:
    content = SKILLS.get_skill_content(skill)
    if content is None:
        available = ", ".join(SKILLS.list_skills())
        return f"Error: Unknown skill '{skill}'. Available skills: {available}"
    else:
        return f"""<skill-loaded name={skill}>
{content}
</skill-loaded>"""


def run_task(task: str, prompt: str, agent_type: str) -> str:
    if agent_type not in SUBAGENT:
        return f"Error: Unknown agent type '{agent_type}'"

    subagent = SUBAGENT[agent_type]
    sub_system = f"""You are a {agent_type} subagent at {WORK_DIR}.
Your task is to {subagent["prompt"]}
Complete the task and return a clear, concise summary."""
    sub_tools = get_subagent_tools(agent_type)
    sub_messages = [{"role": "user", "content": prompt}]  # ISOLATED message history

    print(f"🤖 [{agent_type} subagent] -> {task}")
    start = time.time()
    tool_count = 0

    # Run the same agent loop
    while True:
        response = CLIENT.messages.create(model=MODEL, system=sub_system, messages=sub_messages, tools=sub_tools, max_tokens=MAX_TOKENS)
        sub_messages.append({"role": "assistant", "content": response.content})
        tool_calls = [block for block in response.content if block.type == "tool_use"]

        if response.stop_reason != "tool_use":
            break

        results = []
        for tool_call in tool_calls:
            tool_count += 1
            result = execute_tool(tool_call.name, tool_call.input)
            results.append({"type": "tool_result", "tool_use_id": tool_call.id, "content": result})

            end = time.time() - start
            sys.stdout.write(f"\r🤖 [{agent_type} subagent] -> {task} ... {tool_count} tools, {end:.1f}s")
            sys.stdout.flush()
        sub_messages.append({"role": "user", "content": results})

    end = time.time() - start
    sys.stdout.write(f"\r🤖 [{agent_type} subagent] -> {task} - done ({tool_count} tools, {end:.1f}s)\n")

    # Extract and return only the final text
    for block in response.content:
        if block.type == "text":
            return block.text
    return "(subagent returned None)"


def execute_tool(name: str, args: dict) -> str:
    """
    将大模型的工具调用分发到对应的实现函数，每个工具返回字符串结果并回传给模型。
    """
    if name == "bash":
        return run_bash(args["command"])
    elif name == "read_file":
        return run_read(args["path"], args.get("limit"))
    elif name == "write_file":
        return run_write(args["path"], args["content"])
    elif name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    elif name == "manage_todo":
        return run_todo(args["todos"])
    elif name == "assign_task":
        return run_task(args["task"], args["prompt"], args["agent_type"])
    elif name == "acquire_skill":
        return run_skill(args["skill"])
    else:
        return f"Unknown tool: {name}"


# =============================================================================
# The Agent Loop
# =============================================================================
def agent_loop(messages: list) -> list:
    while True:
        response = CLIENT.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=MAX_TOKENS)
        messages.append({"role": "assistant", "content": response.content})

        # 打印输出、收集工具调用
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                print(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        # 不再需要工具调用，说明任务完成
        if response.stop_reason != "tool_use":
            return messages
        # 执行工具，结果添加到上下文
        else:
            results = []
            for tool_call in tool_calls:
                print(f"\n🛠️ {tool_call.name}: {tool_call.input}")
                result = execute_tool(tool_call.name, tool_call.input)
                print(f"\n💡 {result}")
                results.append({"type": "tool_result", "tool_use_id": tool_call.id, "content": result})
            messages.append({"role": "user", "content": results})


# =============================================================================
# Read-Eval-Print Loop (REPL)
# =============================================================================
if __name__ == "__main__":
    print(f"☄️ Mini Claude Code - {WORK_DIR}")
    history = []
    while True:
        try:
            user_prompt = input("\033[36mYou: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_prompt or user_prompt.lower() in ("exit", "quit"):
            break
        else:
            history.append({"role": "user", "content": user_prompt})
            try:
                agent_loop(history)
            except Exception as e:
                print(f"Error: {e}")
    print("Bye")
