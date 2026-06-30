# Install & usage guide

How to install the starter-kit files, switch the relevant Copilot settings on, and
actually use each piece — with the spend-saving angle called out at every step.

> File formats for Copilot customizations are still evolving (instructions, prompt
> files, chat modes, and custom agents have shifted location/naming across
> releases). Paths below are correct for **current VS Code / Visual Studio**; if a
> file is not picked up, check the official docs linked at the bottom and the
> in-app *Configure* (gear) menu in the Chat view.

---

## 1. What goes where

| Kit file(s) | Copy to (project) | Copy to (personal, all repos) |
|---|---|---|
| `.github/copilot-instructions.md` | repo root `/.github/` | — (project-only) |
| `.github/instructions/*.instructions.md` | repo `/.github/instructions/` | VS Code profile (see §4) |
| `.github/prompts/*.prompt.md` | repo `/.github/prompts/` | VS Code profile (see §4) |
| `.github/agents/*.agent.md` | repo `/.github/agents/` | `~/.copilot/agents/` (Copilot CLI) |
| `chatmodes/*.chatmode.md` | repo `/.github/chatmodes/` | VS Code profile (see §4) |

> Note the move: this kit keeps chat modes under `chatmodes/` for tidiness, but VS
> Code expects them in **`.github/chatmodes/`**. Move the file there on install.

### Fastest project install

From the repo you want to equip, copy the kit's `.github/` directory into the repo
root, then move chat modes into place:

```sh
# from your target repo root; adjust the source path
cp -r /path/to/agentic/copilot-starter-kit/.github .
mkdir -p .github/chatmodes
cp /path/to/agentic/copilot-starter-kit/chatmodes/*.chatmode.md .github/chatmodes/
```

Then **edit the three `Project` lines** in `.github/copilot-instructions.md`
(marked `<!-- ← replace -->`) so they describe the actual repo. Commit the
`.github/` files so your whole team shares them — shared instructions mean fewer
clarifying round-trips for everyone, not just you.

---

## 2. Turn on the settings (VS Code)

Open **Settings (JSON)** — `Ctrl+Shift+P` → *Preferences: Open User Settings
(JSON)* — and add:

```jsonc
{
  // Auto-loads .github/copilot-instructions.md and *.instructions.md
  "github.copilot.chat.codeGeneration.useInstructionFiles": true,

  // Enables /prompt-name reusable prompt files
  "chat.promptFiles": true,

  // OPTIONAL: also load instruction files from extra folders (e.g. user profile)
  "chat.instructionsFilesLocations": { ".github/instructions": true }
}
```

Then, in the **Copilot Chat** view:

- **Enable Auto model selection** (model picker → *Auto*) for the built-in **0.9×
  discount** — see [`TOKEN-SAVING-PLAYBOOK.md`](TOKEN-SAVING-PLAYBOOK.md).
- Leave routine work on the **0× models** (GPT-4o / GPT-4.1); reserve premium
  models for the hard final moves.

### Visual Studio / JetBrains

- **Visual Studio** reads `.github/copilot-instructions.md` and
  `.github/instructions/*.instructions.md` from the solution automatically (recent
  17.x). Prompt files / chat modes support trails VS Code — check the Copilot menu.
- **JetBrains** supports `.github/copilot-instructions.md`; the prompt/agent file
  surface is more limited. Instructions are the highest-leverage piece there.

---

## 3. Using each piece

### Repo instructions (`copilot-instructions.md`)
Always-on, **0 extra requests**. Once installed it silently makes the *first*
answer more correct. Nothing to invoke — just keep it short and factual.

### Path-scoped instructions (`*.instructions.md`)
Auto-applied by the `applyTo:` glob whenever you touch a matching file (e.g. the
Rust file applies to `**/*.rs`). Also free. Nothing to invoke.

### Prompt files (`*.prompt.md`) — your slash commands
In Chat, type `/` then the file name, e.g.:

```
/plan            # plan only, no edits — cheap "think first" step
/review-diff
/add-tests
/fix-failing-test
/explain-code
/debug
/refactor
/commit-msg
/pr-description
/doc-comment
```

Or *Command Palette → Chat: Run Prompt*. Each prompt declares its mode
(`ask` = read-only Q&A, `edit` = scoped edits, `agent` = multi-step). Prefer
`ask`/`edit` for small changes — **agent mode fans out into more premium requests**.

### Custom agents (`*.agent.md`)
Pick them from the agent/mode selector in the Chat view (VS Code), or run them via
the **Copilot CLI** when placed in `~/.copilot/agents/`:

```
code-reviewer       # correctness + risk review of a diff
cpp-modernizer      # behavior-preserving C/C++ modernization
test-author         # focused tests in the repo's framework
security-reviewer   # security-only review
dependency-auditor  # native-tool dependency audit (pip-audit/cargo audit/pnpm audit)
migration-helper    # plan-once, apply-uniformly mechanical migrations
```

### Chat mode (`confirm-first.chatmode.md`)
Select **confirm-first** from the chat mode dropdown for any non-trivial agent task.
It confirms the plan before editing and pauses before declaring done — which is how
you avoid paying for corrective re-prompts.

---

## 4. Personal (all-repos) install in VS Code

To make prompts / instructions / chat modes available in **every** repo without
committing them:

1. Command Palette → *Chat: Configure Instructions* (or *Configure Prompt Files* /
   *Configure Chat Modes*).
2. Choose **User** (profile) as the location and paste the kit file in.

Project files still take precedence and are shared with your team — prefer
project-level for anything stack-specific, user-level for personal conveniences.

---

## 5. MCP (agent mode, optional)

Copilot agent mode can call MCP servers. Add a workspace config at
`.vscode/mcp.json`:

```jsonc
{
  "servers": {
    "example": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."] }
  }
}
```

MCP tool calls run inside agent mode, so the **agent-mode fan-out / premium-request
caution applies** — wire up only servers you will actually use.

---

## 6. Verify it is working

- **Instructions:** start a Chat and ask *"what build/test commands does this repo
  use?"* — it should answer from `copilot-instructions.md` without you pasting
  anything. In recent VS Code, applied instruction files are also listed as
  *References* under the response.
- **Prompts:** typing `/` should list your prompt files by name.
- **Chat modes / agents:** they appear in the mode/agent dropdown of the Chat view.

If something does not appear: confirm the file is in the right folder (§1), the
settings in §2 are on, and reload the window (*Developer: Reload Window*).

---

## 7. Pulling community files

See [`AWESOME-COPILOT-PULL-LIST.md`](AWESOME-COPILOT-PULL-LIST.md) for a curated set
and the `copilot plugin install` syntax. Reconcile any overlaps with the kit's own
files so the model is not reading two competing guides.

---

## Official docs

- Repository custom instructions: <https://docs.github.com/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot>
- Prompt files & chat modes (VS Code): <https://code.visualstudio.com/docs/copilot/copilot-customization>
- Models & pricing (the spend model): <https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing>
- MCP in VS Code: <https://code.visualstudio.com/docs/copilot/chat/mcp-servers>
