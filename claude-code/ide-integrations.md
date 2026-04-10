# IDE Integrations

Claude Code runs in multiple surfaces beyond the raw terminal.

## VS Code extension

- **Install**: `code --install-extension anthropic.claude-code` or from the Extensions view
- **Launch**: Spark icon in editor toolbar, or `Cmd+Esc` / `Ctrl+Esc`
- **Features**: Inline diffs, @-file references, plan review, conversation history
- **@ mentions**: `Option+K` / `Alt+K` to insert a file reference like `@src/auth.ts#1-30`
- Shares config with the CLI (same `~/.claude/settings.json`)

## JetBrains plugin

- **Install**: JetBrains Marketplace → "Claude Code"
- **Supported IDEs**: IntelliJ, PyCharm, WebStorm, GoLand, RubyMine, Rider, CLion, etc.
- **Launch**: `Cmd+Esc`
- **Features**: Diff viewing, selection sharing, quick launch
- **Remote Development**: Plugin must be installed on the remote host, not local client

## Desktop app

- Native Mac/Windows app wrapping the same agent
- Useful when you want CC outside a specific editor

## Web app

- `claude.ai/code` — browser-based, shared session history
- Good for remote work and scheduled agents

## Shared config

All surfaces read from the same settings hierarchy. Define your hooks/plugins/MCP once, they work everywhere.

## Gotchas

- VS Code extension doesn't expose every CLI flag — some flows are CLI-only
- JetBrains keybindings may conflict with IDE defaults
- Remote JetBrains requires plugin on the host side

## Docs

- VS Code: <https://code.claude.com/docs/en/vs-code.md>
- JetBrains: <https://code.claude.com/docs/en/jetbrains.md>
