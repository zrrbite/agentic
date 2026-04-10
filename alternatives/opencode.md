# OpenCode

## TL;DR

Fast-growing 141k-star terminal-first agent from SST. Open-source alternative to Claude Code with client/server architecture, model-agnostic (Claude/OpenAI/Gemini/local), and TUI-native UX built by the Neovim / terminal.shop crew.

## Specs

- **License**: MIT
- **Architecture**: Terminal CLI/TUI primary; desktop app in beta (macOS/Windows/Linux); client/server split enables remote/mobile control; built-in LSP
- **Language**: TypeScript (monorepo via SST)

## Extensibility

- **MCP** — yes, local + remote, with OAuth support (verified from `config.ts`)
- **Plugins** — yes, string or tuple form with options, global/local scope
- **Agents** — `build` and `plan` modes, switchable via `Tab`; subagents via `@general`
- **Slash commands** — yes
- **Permissions** — fine-grained allow/deny over tools
- **Keybindings** — configurable
- **Hooks** — referenced in config schema
- **Config sources** — system-managed → global → well-known remote → project → env vars

## vs Claude Code

- **Fully open source (MIT)** vs Claude Code's proprietary wrapper
- **Model-agnostic out of the box** — any OpenAI-compatible provider works; CC is Anthropic-first
- **Client/server architecture** enables remote/mobile operation — CC is local-only

## Links

- Homepage: <https://opencode.ai>
- GitHub: <https://github.com/sst/opencode>
