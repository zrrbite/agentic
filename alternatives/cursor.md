# Cursor

> ⚠️ **Partially unverified** — cursor.com/docs.cursor.com blocked WebFetch during research. Draws on publicly known positioning as of 2026-04-10.

## TL;DR

Proprietary AI-first VS Code fork from Anysphere — the dominant commercial AI IDE. Known for Tab autocomplete, Composer/Agent mode, and tight inline editing UX.

## Specs

- **License**: Proprietary (commercial); built on open-source VS Code (MIT base)
- **Architecture**: Desktop IDE (VS Code fork) for macOS/Windows/Linux
- **No first-party CLI** — IDE-centric

## Extensibility

- **MCP** — supported in Agent/Composer mode
- **Rules** — `.cursorrules` / custom rules for AI behavior
- **Custom agents** — yes
- **Background agents** — yes
- **Plugins** — inherits VS Code extension marketplace; no formal Cursor-specific plugin system
- **Hooks** — none documented

## vs Claude Code

- **GUI-first** vs CC's terminal-first — better for visual diff review, worse for headless/CI
- **Proprietary, subscription-gated** (~$20/mo) vs CC's usage-based API billing
- **Owns the full editor** — can ship custom UI (Tab, Cmd-K inline) that CC can't match inside another IDE

## Links

- Homepage: <https://cursor.com>
- GitHub: <https://github.com/getcursor/cursor> (issues-only, closed-source)
