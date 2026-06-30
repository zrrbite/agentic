# Copilot starter kit

A drop-in set of GitHub Copilot customizations tuned to **save premium requests**
and standardize behavior across **C, C++, Python, Rust, and TypeScript** projects.
Copy the pieces you want into a repo (or your user profile) and fill in the
bracketed placeholders.

## What's here

| Path | Purpose |
|---|---|
| [`INSTALL.md`](INSTALL.md) | How to install the files, switch on the settings, and use each piece |
| [`WORKFLOW-EXAMPLE.md`](WORKFLOW-EXAMPLE.md) | One task end-to-end — every piece together, with the literal prompts and model tiers |
| [`TOKEN-SAVING-PLAYBOOK.md`](TOKEN-SAVING-PLAYBOOK.md) | The strategy: premium-request economics + the levers that cut spend |
| [`MODEL-ROUTING-CHEAT-CARD.md`](MODEL-ROUTING-CHEAT-CARD.md) | One-screen task-type → model-tier decision table (the biggest spend lever) |
| [`SPEND-AUDIT.md`](SPEND-AUDIT.md) | Weekly 5-minute self-audit: read the usage dashboard, find and fix leaks |
| [`AWESOME-COPILOT-PULL-LIST.md`](AWESOME-COPILOT-PULL-LIST.md) | Curated list of community files/plugins to pull from `github/awesome-copilot` |
| `.github/copilot-instructions.md` | Repo-wide context auto-loaded by Copilot Chat (fill in build/test/style) |
| `.github/instructions/*.instructions.md` | Path-scoped coding standards (C/C++, Python, Rust, TypeScript) applied by file glob |
| `.github/prompts/*.prompt.md` | Reusable slash-command tasks (plan, add-tests, refactor, review-diff, explain-code, fix-failing-test, debug, commit-msg, pr-description, doc-comment) |
| `.github/agents/*.agent.md` | Custom agents (code-reviewer, cpp-modernizer, test-author, security-reviewer, dependency-auditor, migration-helper) |
| `chatmodes/*.chatmode.md` | Chat-mode personas: confirm-first (confirms before finishing), terse-senior, caveman, savage-reviewer |

## Install

Full step-by-step in [`INSTALL.md`](INSTALL.md) (file placement, the VS Code
settings to switch on, how to invoke each piece, and verification). In brief:

- **Project-level (shared):** copy `.github/` into your repo root; move chat modes
  to `.github/chatmodes/`. Picked up by VS Code / Visual Studio.
- **User-level (personal, all repos):** agent files in `~/.copilot/agents/`;
  prompts/instructions/chat modes via VS Code *Configure* → User profile.
- **Switch on:** `useInstructionFiles` + `chat.promptFiles`, and **Auto** model
  selection for the 0.9× discount — details in `INSTALL.md` §2.

## First moves to actually save requests

1. Fill in `.github/copilot-instructions.md` (build/test/conventions) — kills
   clarifying round-trips.
2. Turn on **Auto** model selection (0.9× discount) and default to the 0× models.
3. Use the `code-reviewer` / `confirm-first` agents to avoid corrective re-prompts.

See [`TOKEN-SAVING-PLAYBOOK.md`](TOKEN-SAVING-PLAYBOOK.md) for the full rationale.
