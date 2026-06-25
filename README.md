# Agentic Coding — reference notes

A rough-notes knowledge base for exploring what **Claude Code** (and agentic coding tools in general) can do. The goal is to map out the full landscape: Claude Code's extensibility surface, external tools that interop with it, and the broader field of AI coding agents.

**Status**: living document, rough notes > polish.

## Layout

| Folder | What's inside |
|---|---|
| [`claude-code/`](claude-code/) | Claude Code's extensibility surface — MCP, plugins, skills, subagents, hooks, SDK, settings, IDE integrations |
| [`integrations/`](integrations/) | External tools that talk to Claude Code — OpenClaw, n8n, LangGraph, CrewAI, Letta |
| [`alternatives/`](alternatives/) | Competing agentic coding tools — OpenCode, Cursor, Aider, Cline, Codex, Continue, Windsurf, Zed, Goose, GitHub Copilot |
| [`sources/`](sources/) | Where to find more — awesome-lists, plugin marketplaces, skill repos, MCP registries |
| [`theory/`](theory/) | How LLMs actually work — neural networks, gradient descent, transformers, training (RLHF/DPO), inference. Beginner + advanced sections per file |
| [`experiments/`](experiments/) | Hands-on "I tried X" notes (dated files, added as you go) |

## Start here

- New to Claude Code extension points? → [`claude-code/README.md`](claude-code/README.md)
- Looking for prebuilt skills/plugins? → [`sources/skill-repos.md`](sources/skill-repos.md)
- Comparing tools? → [`alternatives/README.md`](alternatives/README.md)
- Want to understand how LLMs work under the hood? → [`theory/README.md`](theory/README.md)
- Want to read the LLM notebooks on your phone (no install)? → [`theory/code/snapshots/`](theory/code/snapshots/)

## Getting started — running the LLM theory notebooks

`theory/code/` has hands-on notebooks that build the math from scratch (companions to the docs in `theory/`). The other folders are read-only notes — no setup needed.

**Prerequisites**: Python 3.10+. On Windows, install from [python.org](https://www.python.org/downloads/) rather than the Microsoft Store version (which is sandboxed and causes weird issues with venvs and PATH).

**1. Create a virtualenv** (one-time, from this repo's root):

```bash
cd theory/code
python -m venv .venv
```

**2. Activate it** — the command depends on your shell:

| Shell | Activate command |
|---|---|
| Git Bash (Windows) | `source .venv/Scripts/activate` |
| PowerShell (Windows) | `.\.venv\Scripts\Activate.ps1` |
| CMD (Windows) | `.venv\Scripts\activate.bat` |
| bash / zsh (macOS / Linux) | `source .venv/bin/activate` |

Your prompt should now show `(.venv)`. If it doesn't, activation didn't take — `pip install` will go to your system Python and `jupyter lab` won't be found later.

**3. Install dependencies**:

```bash
pip install numpy matplotlib jupyterlab
```

**4. Launch JupyterLab**:

```bash
jupyter lab
```

A browser tab opens at `http://localhost:8888`. Double-click `01-mlp-from-scratch.ipynb`, then **Run → Restart Kernel and Run All Cells…**. Work through the notebooks in order — each builds on the last.

→ Stuck? Full troubleshooting in [`theory/code/README.md`](theory/code/README.md). Never used a notebook before? Start with [`theory/code/NOTEBOOKS.md`](theory/code/NOTEBOOKS.md).

## Conventions

- Rough notes, bullets over prose
- Every external claim links to a source — if no source, mark "unverified"
- Each file: *what it is → where it lives → minimal example → gotchas → docs link*
