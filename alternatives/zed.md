# Zed

## TL;DR

High-performance Rust-native editor from the Atom/Tree-sitter creators. Agent Panel with MCP-compatible context servers and **ACP (Agent Client Protocol)** support; real-time multiplayer editing is its other headline feature. 78k+ stars.

## Specs

- **License**: Multi-licensed — AGPL (editor), Apache 2.0, GPL-3.0 components (Zed Industries, Inc.)
- **Architecture**: Native desktop editor in Rust (97.7%)
- **Platforms**: macOS/Linux/Windows; web version in development
- **Not VS Code-based** — fully custom, GPU-rendered

## Extensibility

- **Extensions system** — first-party
- **MCP context servers** — in Agent Panel
- **ACP (Agent Client Protocol)** — pluggable external agents
- **Convention files** — `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`
- **Tree-sitter parsing** — per-language

## Key insight: ACP changes the game

Zed's ACP actually lets you **plug Claude Code (and other agents) *into* Zed** as the backing agent. So Zed is less a competitor to CC and more a potential **host** for it.

## vs Claude Code

- **Real-time collaborative editing** — unique vs every other tool in this list, including CC
- **Zed's ACP lets you use CC *as* the agent** — complementary more than competing
- **Native-Rust editor performance** with GPU rendering — CC has no editor surface at all

## Links

- Homepage: <https://zed.dev>
- GitHub: <https://github.com/zed-industries/zed>
