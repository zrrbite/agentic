# Aider

## TL;DR

Mature Python-based terminal pair programmer with industry-leading git integration — auto-commits every change with descriptive messages. Batteries-included: repo map, voice-to-code, linting/test loops, 100+ language support.

## Specs

- **License**: Apache-2.0
- **Architecture**: Terminal CLI (Python), no GUI, IDE integration via file-watching only
- **Language**: Python

## Extensibility

- **MCP** — **not supported** per README (notable gap)
- **Multi-LLM** — Claude 3.7 Sonnet, DeepSeek, o1/o3-mini, local via Ollama, and more
- **Config** — `.aider.conf.yml`, CLI flags, convention files
- **Slash commands** — in-chat
- **No formal plugin/hook system**

## Signature feature: git-first

- Every edit is committed automatically with a generated descriptive message
- Repo history becomes a granular timeline of AI-assisted changes
- Easy to revert individual tweaks with standard git tools

## vs Claude Code

- **Git-first philosophy** — automatic commits per edit; CC leaves git to the user
- **No MCP ecosystem** — less extensible for external tools than CC
- **Pioneered repo-map context selection** — still a strong baseline for token-efficient large-repo editing

## Links

- Homepage: <https://aider.chat>
- GitHub: <https://github.com/Aider-AI/aider>
