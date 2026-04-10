# Cline

## TL;DR

Apache-2.0 VS Code extension (formerly "Claude Dev") with human-in-the-loop approvals at every step. Differentiates with built-in browser automation (headless click/screenshot) and workspace checkpoints for rollback. Broad provider support.

## Specs

- **License**: Apache 2.0 (Cline Bot Inc.)
- **Architecture**: VS Code extension
- **Enterprise**: Cline Bot Inc. ships SSO/audit offering

## Extensibility

- **MCP** — yes; agents can even **author and install new MCP servers autonomously**
- **Checkpoints** — workspace snapshots at each step, compare and restore
- **Multi-provider** — OpenRouter, Anthropic, OpenAI, Gemini, Bedrock, Azure, LM Studio, Ollama
- **Plan/Act modes** — separate planning from execution
- **Custom instructions/rules**
- **Context anchors** — `@url`, `@problems`, `@file`, `@folder`

## vs Claude Code

- **Step-by-step approval gates** by default — more cautious than CC's permission modes
- **Native headless-browser tool** — CC has no built-in browser automation
- **Checkpoint/restore snapshots built in** — CC relies on git for rollback

## Links

- Homepage: <https://cline.bot>
- GitHub: <https://github.com/cline/cline>
