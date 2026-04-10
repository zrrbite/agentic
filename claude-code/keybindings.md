# Keybindings

## What it is

Keyboard shortcuts for Claude Code's CLI interface.

## Where it lives

`~/.claude/keybindings.json` — **user scope only** (not project-shareable).

## Example

```json
{
  "bindings": {
    "ctrl+s": "submit",
    "ctrl+n": "new-conversation",
    "ctrl+shift+k": "cancel"
  }
}
```

## Configuration via skill

Claude Code ships a `keybindings-help` skill that knows the full action list and chord syntax — invoke it to customize interactively.

## Chord bindings

Multi-key sequences are supported (e.g. `ctrl+k ctrl+s` as a chord). Exact syntax details are best discovered via the `keybindings-help` skill — public docs are sparse here.

## Gotchas

- User scope only — can't commit to a project
- Chord bindings can conflict with terminal emulator shortcuts
- Full action list is **unverified** from public docs

## Docs

- <https://code.claude.com/docs/en/settings.md>
- Use the `keybindings-help` skill
