"""
v5_skills_agent.py - Mini Claude Code: Skills Mechanism (~550 lines)

v4 gave us subagents for task decomposition. But there's a deeper question:
    How does the model know HOW to handle domain-specific tasks?

- Processing PDFs? It needs to know pdftotext vs PyMuPDF
- Building MCP servers? It needs protocol specs and best practices
- Code review? It needs a systematic checklist

This knowledge isn't a tool - it's EXPERTISE. Skills solve this by letting the model load domain knowledge on-demand.

Traditional AI: Knowledge locked in model parameters
  - To teach new skills: collect data -> train -> deploy
  - Cost: $10K-$1M+, Timeline: Weeks
  - Requires ML expertise, GPU clusters

Skills: Knowledge stored in editable files
  - To teach new skills: write a SKILL.md file
  - Cost: Free, Timeline: Minutes
  - Anyone can do it

It's like attaching a hot-swappable LoRA adapter without any training!
这就像给模型接上一个“可热插拔”的 LoRA 适配器，但不需要任何训练！

Tools vs Skills:
---------------
    | Concept   | What it is              | Example                    |
    |-----------|-------------------------|---------------------------|
    | **Tool**  | What model CAN do       | bash, read_file, write    |
    | **Skill** | How model KNOWS to do   | PDF processing, MCP dev   |
Tools are capabilities. Skills are knowledge.

Progressive Disclosure:
渐进式披露
----------------------
    Layer 1: Metadata (always loaded)      ~100 tokens skill_name + description only
    Layer 2: SKILL.md body (on trigger)    ~2000 tokens Detailed instructions
    Layer 3: Resources (as needed)         Unlimited scripts/, references/, assets/

This keeps context lean while allowing arbitrary depth.
这样既能保持上下文精简，又能在需要时扩展到任意深度。

SKILL.md Standard:
-----------------
    skills/
    |-- pdf/
    |   |-- SKILL.md          # Required: YAML frontmatter + Markdown body
    |-- mcp-builder/
    |   |-- SKILL.md
    |   |-- references/       # Optional: docs, specs
    |-- code-review/
        |-- SKILL.md
                |-- scripts/          # Optional: helper scripts

Cache-Preserving Injection:
--------------------------
Skill content goes into tool_result (user message), NOT system prompt. This preserves prompt cache!
    ❌: Edit system prompt each time (cache invalidated, 20-50x cost)
    ✅: Append skill as tool result (prefix unchanged, cache hit)

This is how production Claude Code works - and why it's cost-efficient.

Usage:
    python v5_skills_agent.py
"""

import os
import re
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
SKILLS_DIR = WORKDIR / "skills"
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), base_url=os.getenv("ANTHROPIC_BASE_URL"))
model = os.getenv("MODEL")
max_tokens = int(os.getenv("MAX_TOKENS"))


# =============================================================================
# SkillLoader - The core addition in v5
# =============================================================================
class SkillLoader:
    """
    Loads and manages skills from SKILL.md files.

    A skill is a FOLDER containing:
    - SKILL.md (required): YAML frontmatter + markdown instructions
    - scripts/ (optional): Helper scripts the model can run
    - references/ (optional): Additional documentation
    - assets/ (optional): Templates, files for output

    SKILL.md Format:
    ----------------
        ---
        name: pdf
        description: Process PDF files. Use when reading, creating, or merging PDFs.
        ---

        # PDF Processing Skill

        ## Reading PDFs

        Use pdftotext for quick extraction:
        ```bash
        pdftotext input.pdf -
        ```
        ...

    The YAML frontmatter provides metadata (name, description).
    The markdown body provides detailed instructions.
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self.load_skills()

    def parse_skill_md(self, path: Path) -> dict:
        """
        Parse a SKILL.md file into metadata and body.

        Returns dict with: name, description, body, path, dir
        Returns None if file doesn't match format.
        """
        content = path.read_text()

        # Match YAML frontmatter between --- markers
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
        if not match:
            return None

        frontmatter, body = match.groups()

        # Parse YAML-like frontmatter (simple key: value)
        metadata = {}
        for line in frontmatter.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip("\"'")

        # Require name and description
        if "name" not in metadata or "description" not in metadata:
            return None

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
        This keeps the initial context lean.
        """
        if not self.skills_dir.exists():
            return

        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            skill = self.parse_skill_md(skill_md)
            if skill:
                self.skills[skill["name"]] = skill

    def get_descriptions(self) -> str:
        """
        Generate skill descriptions for system prompt.

        This is Layer 1 - only name and description, ~100 tokens per skill.
        Full content (Layer 2) is loaded only when Skill tool is called.
        """
        if not self.skills:
            return "(no skills available)"

        return "\n".join(f"- {name}: {skill['description']}" for name, skill in self.skills.items())

    def get_skill_content(self, name: str) -> str:
        """
        Get full skill content for injection.

        This is Layer 2 - the complete SKILL.md body, plus any available resources (Layer 3 hints).

        Returns None if skill not found.
        """
        if name not in self.skills:
            return None

        skill = self.skills[name]
        content = f"# Skill: {skill['name']}\n\n{skill['body']}"

        # List available resources (Layer 3 hints)
        resources = []
        for folder, label in [("scripts", "Scripts"), ("references", "References"), ("assets", "Assets")]:
            folder_path = skill["dir"] / folder
            if folder_path.exists():
                files = list(folder_path.glob("*"))
                if files:
                    resources.append(f"{label}: {', '.join(f.name for f in files)}")

        if resources:
            content += f"\n\n**Available resources in {skill['dir']}:**\n"
            content += "\n".join(f"- {r}" for r in resources)

        return content

    def list_skills(self) -> list:
        """
        Return list of available skill names.
        """
        return list(self.skills.keys())


# Global skill loader instance
SKILLS = SkillLoader(SKILLS_DIR)


# =============================================================================
# Agent Type Registry (from v4)
# =============================================================================
AGENT_TYPES = {
    "explore": {
        "description": "Read-only agent for exploring code, finding files, searching",
        "tools": ["bash", "read_file"],
        "prompt": "You are an exploration agent. Search and analyze, but never modify files. Return a concise summary.",
    },
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["bash", "read_file"],
        "prompt": "You are a planning agent. Analyze the codebase and output a numbered implementation plan. Do NOT make changes.",
    },
    "code": {
        "description": "Full agent for implementing features and fixing bugs",
        "tools": "*",
        "prompt": "You are a coding agent. Implement the requested changes efficiently.",
    },
}


def get_agent_descriptions() -> str:
    return "\n".join(f"- {name}: {cfg['description']}" for name, cfg in AGENT_TYPES.items())


# =============================================================================
# TodoManager (from v2)
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
# System Prompt - Updated for v5
# =============================================================================
system = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

**Skills available** (invoke with Skill tool when task matches):
{SKILLS.get_descriptions()}

**Subagents available** (invoke with Task tool for focused subtasks):
{get_agent_descriptions()}

Rules:
- Use Skill tool IMMEDIATELY when a task matches a skill description
- Use Task tool for subtasks needing focused exploration or implementation
- Use TodoWrite to track multi-step work
- Prefer tools over text. Act, don't just explain.
- After finishing, summarize what changed."""


# =============================================================================
# Tool Definitions
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

# Task tool (from v4)
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

# NEW in v5: Skill tool
SKILL_TOOL = {
    "name": "Skill",
    "description": f"""Load a skill to gain specialized knowledge for a task.

Available skills:
{SKILLS.get_descriptions()}

When to use:
- IMMEDIATELY when user task matches a skill description
- Before attempting domain-specific work (PDF, MCP, etc.)

The skill content will be injected into the conversation, giving you
detailed instructions and access to resources.""",
    "input_schema": {
        "type": "object",
        "properties": {"skill": {"type": "string", "description": "Name of the skill to load"}},
        "required": ["skill"],
    },
}

ALL_TOOLS = BASE_TOOLS + [TASK_TOOL, SKILL_TOOL]


def get_tools_for_agent(agent_type: str) -> list:
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


def run_skill(skill_name: str) -> str:
    """
    Load a skill and inject it into the conversation.

    This is the key mechanism:
    1. Get skill content (SKILL.md body + resource hints)
    2. Return it wrapped in <skill-loaded> tags
    3. Model receives this as tool_result (user message)
    4. Model now "knows" how to do the task

    Why tool_result instead of system prompt?
    - System prompt changes invalidate cache (20-50x cost increase)
    - Tool results append to end (prefix unchanged, cache hit)

    This is how production systems stay cost-efficient.
    """
    content = SKILLS.get_skill_content(skill_name)

    if content is None:
        available = ", ".join(SKILLS.list_skills())
        return f"Error: Unknown skill '{skill_name}'. Available: {available}"

    return f"""<skill-loaded name="{skill_name}">
{content}
</skill-loaded>

Follow the instructions in the skill above to complete the user's task."""


def run_task(description: str, prompt: str, agent_type: str) -> str:
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    config = AGENT_TYPES[agent_type]
    sub_system = f"""You are a {agent_type} subagent at {WORKDIR}.

{config["prompt"]}

Complete the task and return a clear, concise summary."""

    sub_tools = get_tools_for_agent(agent_type)
    sub_messages = [{"role": "user", "content": prompt}]

    print(f"  [{agent_type}] {description}")
    start = time.time()
    tool_count = 0

    while True:
        response = client.messages.create(model=model, system=sub_system, messages=sub_messages, tools=sub_tools, max_tokens=max_tokens)

        if response.stop_reason != "tool_use":
            break

        tool_calls = [block for block in response.content if block.type == "tool_use"]
        results = []

        for tc in tool_calls:
            tool_count += 1
            output = execute_tool(tc.name, tc.input)
            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

            elapsed = time.time() - start
            sys.stdout.write(f"\r  [{agent_type}] {description} ... {tool_count} tools, {elapsed:.1f}s")
            sys.stdout.flush()

        sub_messages.append({"role": "assistant", "content": response.content})
        sub_messages.append({"role": "user", "content": results})

    elapsed = time.time() - start
    sys.stdout.write(f"\r  [{agent_type}] {description} - done ({tool_count} tools, {elapsed:.1f}s)\n")

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
        return run_todo(args["items"])
    if name == "Task":
        return run_task(args["description"], args["prompt"], args["agent_type"])
    if name == "Skill":
        return run_skill(args["skill"])
    return f"Unknown tool: {name}"


# =============================================================================
# Main Agent Loop
# =============================================================================
def agent_loop(messages: list) -> list:
    """
    Main agent loop with skills support.

    Same pattern as v4, but now with Skill tool.
    When model loads a skill, it receives domain knowledge.
    """
    while True:
        response = client.messages.create(model=model, system=system, messages=messages, tools=ALL_TOOLS, max_tokens=max_tokens)

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
            # Special display for different tool types
            if tc.name == "Task":
                print(f"\n> Task: {tc.input.get('description', 'subtask')}")
            elif tc.name == "Skill":
                print(f"\n> Loading skill: {tc.input.get('skill', '?')}")
            else:
                print(f"\n> {tc.name}")

            output = execute_tool(tc.name, tc.input)

            # Skill tool shows summary, not full content
            if tc.name == "Skill":
                print(f"  Skill loaded ({len(output)} chars)")
            elif tc.name != "Task":
                preview = output[:200] + "..." if len(output) > 200 else output
                print(f"  {preview}")

            results.append({"type": "tool_result", "tool_use_id": tc.id, "content": output})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": results})


# =============================================================================
# Main REPL
# =============================================================================
def main():
    print(f"Mini Claude Code v5 (with Skills) - {WORKDIR}")
    print(f"Skills: {', '.join(SKILLS.list_skills())}")
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
