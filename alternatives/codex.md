# OpenAI Codex CLI

## TL;DR

OpenAI's official terminal agent, rewritten in Rust for speed and portability. Single binary, ChatGPT-account auth, plus sibling IDE extensions for VS Code/Cursor/Windsurf. Lightweight focus.

## Specs

- **License**: Apache-2.0
- **Architecture**: Terminal CLI (Rust ~95%, with Python/TS components)
- **Distribution**: npm, Homebrew, native binaries
- **Companion products**: VS Code/Cursor/Windsurf extensions, `codex app` desktop/web interface
- **Auth**: ChatGPT account (Plus/Pro/Business/Edu/Enterprise) or API key

## Extensibility

- Sandbox/approval modes and MCP support exist in current releases — *not explicitly surfaced in README, partially unverified*
- Uses OpenAI Codex-series models via OpenAI backend
- Smaller extensibility surface than CC's skills/hooks/subagents ecosystem

## vs Claude Code

- **OpenAI-native** (tied to ChatGPT auth/models) vs CC's Anthropic-native binding
- **Rust single-binary** distribution — lighter install footprint than CC's Node runtime
- **Smaller extensibility surface** today than CC's hooks/skills/subagents

## Links

- Homepage: <https://openai.com/codex> and <https://chatgpt.com/codex>
- GitHub: <https://github.com/openai/codex>
