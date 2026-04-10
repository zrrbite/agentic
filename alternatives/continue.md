# Continue.dev

## TL;DR

Apache-2.0 dual product:
1. **IDE extension** (VS Code + JetBrains) — long-standing, with Hub/Assistants/Rules/MCP
2. **`cn` CLI** — newer; runs AI checks as GitHub status checks from markdown files in `.continue/checks/`

Source-controlled AI review is their current pitch. 32.4k stars.

## Specs

- **License**: Apache 2.0 (Continue Dev, Inc.)
- **Architecture**: IDE extensions (VS Code + JetBrains), CLI `cn` (Node.js 20+)
- **Languages**: TypeScript (84%), JavaScript, Kotlin (JetBrains), Python, Rust
- **Distribution**: bash/PowerShell installers, `npm i -g @continuedev/cli`

## Extensibility

- **MCP** — yes, registry + tool use
- **Hub** — hosted Assistants (shareable agent configs), Rules, Models
- **Custom models/providers** — fully BYO
- **`.continue/checks/`** — markdown check definitions enforced as CI status checks
- **IDE features** — slash commands, context providers, chat, autocomplete

## Unique positioning: AI as CI

Define checks as markdown files committed to your repo. They run on PRs as GitHub status checks — green = pass, red = suggested diffs. Enforceable in deployment pipelines.

Example check idea: "No hardcoded secrets", "API inputs validated", "Error handling consistent".

## vs Claude Code

- **Bring-your-own-model** ethos — CC is Anthropic-first
- **AI as version-controlled CI checks** on PRs — CC has no native GitHub-status-check workflow
- **IDE-embedded chat/autocomplete** is a category CC doesn't enter (terminal-first)

## Links

- Homepage: <https://continue.dev>
- GitHub: <https://github.com/continuedev/continue>
- Docs: <https://docs.continue.dev>
