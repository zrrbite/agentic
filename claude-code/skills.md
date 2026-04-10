# Skills

## What it is

Skills are reusable task-specific instructions that Claude can invoke manually (`/skill-name`) or **automatically** when the description matches the user's intent. Defined as `SKILL.md` files with frontmatter in `.claude/skills/<name>/` directories. Skills supersede the older `.claude/commands/` slash commands.

## Key difference vs slash commands

- Slash commands: **user-invoked only** (`/deploy`)
- Skills: **can auto-trigger** based on the `description` field — Claude picks them up when relevant
- Skills can load supporting files from the skill directory
- Skills can be preloaded into subagents

## Where they live

- `~/.claude/skills/<name>/SKILL.md` — personal
- `.claude/skills/<name>/SKILL.md` — project
- `<plugin>/skills/<name>/SKILL.md` — plugin-bundled

## Minimal example

```markdown
---
name: code-review
description: Review code for bugs and best practices. Use when reviewing PRs or analyzing code quality.
disable-model-invocation: false
---
Review the provided code for:
1. Potential bugs
2. Performance issues
3. Best practices violations
4. Security concerns
```

## Frontmatter fields

| Field | Meaning |
|---|---|
| `name` | Skill identifier, becomes `/name` |
| `description` | Shown to Claude for auto-invocation (max ~250 chars) |
| `disable-model-invocation` | If `true`, skill won't auto-trigger — user must invoke |
| `allowed-tools` | Restrict which tools the skill can use |

## Gotchas

- Descriptions are **truncated at ~250 chars** for context cost — keep them tight
- Skill content is stored once per session; may drop during `/compact` if many are invoked
- Auto-invocation isn't deterministic — don't rely on it for critical flows
- Use `disable-model-invocation: true` for destructive commands (e.g. `/deploy`)

## Notable real-world skill pack

**Superpowers** (`obra/superpowers`) — agentic skills framework with brainstorming, writing-plans, executing-plans, TDD, systematic-debugging, verification-before-completion, and more. Install: `/plugin install superpowers@claude-plugins-official`.

## Docs

- <https://code.claude.com/docs/en/skills.md>
- See also: `../sources/skill-repos.md` for community skill collections
