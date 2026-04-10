# Output Styles

## What it is

Output styles customize Claude's tone, role, and response format by modifying the session system prompt. Built-in styles:

- **Default** — software engineering mode
- **Explanatory** — adds educational insights
- **Learning** — collaborative learn-by-doing

Custom styles are markdown files with frontmatter.

## Where it lives

- `~/.claude/output-styles/<name>.md` — user scope
- `.claude/output-styles/<name>.md` — project scope
- Set via `/config` or the `outputStyle` field in `settings.json`

## Minimal example

`.claude/output-styles/technical-writer.md`:
```markdown
---
name: technical-writer
description: Write clear technical documentation
keep-coding-instructions: true
---
You are a technical writer. Explain concepts clearly with examples.
Use simple language and avoid jargon.
```

## Frontmatter fields

| Field | Meaning |
|---|---|
| `name` | Style identifier |
| `description` | Shown in `/config` picker |
| `keep-coding-instructions` | If `true`, keeps default coding guidance; if `false` (default), strips it |

## Activating

```json
{
  "outputStyle": "technical-writer"
}
```

Or pick interactively via `/config`.

## Gotchas

- Output styles don't change what Claude **knows**, only **how** it responds
- Custom styles exclude default coding instructions unless you set `keep-coding-instructions: true`
- Style changes take effect on **next session** — the system prompt is fixed per session

## Docs

- <https://code.claude.com/docs/en/output-styles.md>
