# Alternatives — competing agentic coding tools

Rough field guide to the other players. Kept here mainly to understand trade-offs and borrow ideas.

## Tools in this folder

| Tool | Type | License | Headline |
|---|---|---|---|
| [OpenCode](opencode.md) | Terminal/TUI | MIT | Open-source CC alternative, client/server, model-agnostic |
| [Cursor](cursor.md) | Desktop IDE (VS Code fork) | Proprietary | Dominant commercial AI IDE |
| [Aider](aider.md) | Terminal CLI | Apache-2.0 | Git-first pair programmer, auto-commits everything |
| [Cline](cline.md) | VS Code extension | Apache-2.0 | Human-in-the-loop, browser automation, checkpoints |
| [Codex CLI](codex.md) | Terminal CLI | Apache-2.0 | OpenAI's official terminal agent, Rust rewrite |
| [Continue](continue.md) | IDE ext + CLI | Apache-2.0 | BYO-model; CI checks on PRs via `.continue/checks/` |
| [Windsurf](windsurf.md) | Desktop IDE (VS Code fork) | Proprietary | Cascade flows, Supercomplete; owned by Cognition |
| [Zed](zed.md) | Native editor | AGPL/Apache mix | Rust editor; ACP lets you plug CC *into* Zed |
| [Goose](goose.md) | Desktop + CLI | Apache-2.0 | Block's general-purpose agent, 70+ MCP extensions |

## High-level axes

- **Terminal vs IDE vs both** — CC and OpenCode/Aider/Codex are terminal-first; Cursor/Windsurf/Cline/Zed are editor-first; Continue/Goose span both
- **Model-agnostic vs single-vendor** — OpenCode/Continue/Goose/Zed/Cline are multi-provider; CC is Anthropic-first; Codex is OpenAI-first
- **Open source vs proprietary** — Cursor + Windsurf are the only truly closed ones; everything else is open
- **MCP support** — Most now support MCP: CC, OpenCode, Cline, Continue, Windsurf, Zed, Goose. Aider notably doesn't.

## Research gaps

Cursor and Windsurf details are **partially unverified** — their official domains (cursor.com, windsurf.com) blocked WebFetch in research. Treat those entries as drawing on publicly known positioning, not freshly-verified specs.
