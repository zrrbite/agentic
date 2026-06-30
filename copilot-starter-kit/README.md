# Copilot starter kit

A drop-in set of GitHub Copilot customizations tuned to **save premium requests**
and standardize behavior across **C, C++, Python, Rust, and TypeScript** projects.
Copy the pieces you want into a repo (or your user profile) and fill in the
bracketed placeholders.

## What's here

| Path | Purpose |
|---|---|
| [`TOKEN-SAVING-PLAYBOOK.md`](TOKEN-SAVING-PLAYBOOK.md) | The strategy: premium-request economics + the levers that cut spend |
| [`AWESOME-COPILOT-PULL-LIST.md`](AWESOME-COPILOT-PULL-LIST.md) | Curated list of community files/plugins to pull from `github/awesome-copilot` |
| `.github/copilot-instructions.md` | Repo-wide context auto-loaded by Copilot Chat (fill in build/test/style) |
| `.github/instructions/*.instructions.md` | Path-scoped coding standards (C/C++, Python, Rust, TypeScript) applied by file glob |
| `.github/prompts/*.prompt.md` | Reusable slash-command tasks (add-tests, refactor, review-diff, explain-code, fix-failing-test, commit-msg, doc-comment, debug) |
| `.github/agents/*.agent.md` | Custom agents (code-reviewer, cpp-modernizer, test-author, security-reviewer, dependency-auditor, migration-helper) |
| `chatmodes/confirm-first.chatmode.md` | Legacy chat-mode persona that confirms before finishing |

## Install

- **Project-level (shared):** copy `.github/` into your repo root. Agents,
  prompts, and instructions are picked up automatically by VS Code / Visual Studio.
- **User-level (personal, all repos):** place agent files in `~/.copilot/agents/`.
- **Chat modes → agents:** the `.chatmode.md` format is being superseded by
  `.agent.md`; rename to migrate. Kept here for older clients.

## First moves to actually save requests

1. Fill in `.github/copilot-instructions.md` (build/test/conventions) — kills
   clarifying round-trips.
2. Turn on **Auto** model selection (0.9× discount) and default to the 0× models.
3. Use the `code-reviewer` / `confirm-first` agents to avoid corrective re-prompts.

See [`TOKEN-SAVING-PLAYBOOK.md`](TOKEN-SAVING-PLAYBOOK.md) for the full rationale.
