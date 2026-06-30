# awesome-copilot — curated pull list

A short, opinionated list of the highest-value items to pull from
[`github/awesome-copilot`](https://github.com/github/awesome-copilot) for a
**C/C++ · Python · Rust · TypeScript** stack — instead of cloning the whole repo.
Pull the few things that match your work; every file you add is context the model
loads, so stay lean.

> Names below were verified against the repo's `instructions/`, `agents/`, and
> `plugins/` folders. Folder contents change — confirm a name on the hub before
> installing. The **website** (linked from the repo README) has search + an
> install button per item.

## How to install

```sh
# one-time: register the marketplace
copilot plugin marketplace add github/awesome-copilot

# install a plugin (bundle of agents + skills)
copilot plugin install <plugin-name>@awesome-copilot
```

- **Plugins** install via the CLI above (agent mode).
- **Instruction / agent / prompt files** are plain Markdown: copy the file from the
  hub into your repo's `.github/instructions/`, `.github/agents/`, or
  `.github/prompts/`. Use the website's per-file install button, or copy the raw
  contents manually.

## The token-smart first move

Install the **meta recommender** and let it suggest only what your repo needs —
this is the cheapest way to avoid over-pulling:

```sh
copilot plugin install awesome-copilot@awesome-copilot
```

It suggests relevant collections/instruction files based on the current repo and
chat context, and avoids duplicating instructions you already have.

## Instruction files (path-scoped standards)

Drop into `.github/instructions/`. These complement — do not duplicate — the ones
already in this starter kit.

| File | Use it for |
|---|---|
| `rust.instructions.md` | General Rust standards (cross-check vs the kit's `rust.instructions.md`; keep one) |
| `security-and-owasp.instructions.md` | OWASP/security guidance applied across files |
| `performance-optimization.instructions.md` | Perf-focused guidance |
| `code-review-generic.instructions.md` | Language-agnostic review checklist |
| `cpp-language-service-tools.instructions.md` | C++ tooling-aware guidance |
| `langchain-python.instructions.md` | If you build LLM apps in Python |
| `playwright-python.instructions.md` / `playwright-typescript.instructions.md` | E2E test authoring |
| `nodejs-javascript-vitest.instructions.md` | Vitest-based JS/TS testing |

> Note: there is no plain `python.instructions.md` on the hub — most Python files
> are framework-specific (dataverse, langchain, playwright). The kit's own
> `python.instructions.md` covers the general case; pull a framework file only if
> you use that framework.

## Agents

Drop into `.github/agents/` (or `~/.copilot/agents/` for all repos).

| File | Use it for |
|---|---|
| `expert-cpp-software-engineer.agent.md` | Deep C++ work |
| `debug.agent.md` | General debugging (compare with the kit's `debug.prompt.md`) |
| `tdd-refactor.agent.md` | Test-driven refactoring |
| `sast-sca-security-analyzer.agent.md` / `se-security-reviewer.agent.md` | Security review (compare with the kit's `security-reviewer.agent.md`) |
| `frontend-performance-investigator.agent.md` | Front-end perf |
| `project-documenter.agent.md` | Generating/maintaining docs |
| `python-mcp-expert.agent.md` / `rust-mcp-expert.agent.md` | Building MCP servers in Python/Rust |

## Plugins (bundles)

Install only the ones matching your work:

| Plugin | Use it for |
|---|---|
| `awesome-copilot` | Meta recommender (install this first — see above) |
| `context-engineering` | Better prompt/context discipline → fewer round-trips |
| `frontend-web-dev` | Front-end TypeScript/React work |
| `database-data-management` | DB-heavy work |
| `devops-oncall` | Ops/incident workflows |
| `go-mcp-development` / `java-development` / `csharp-dotnet-development` | If those stacks appear at work |
| `openapi-to-application-python-fastapi` | Scaffolding FastAPI from OpenAPI |
| `oracle-to-postgres-migration-expert` | DB migration (pairs with the kit's `migration-helper` agent) |

## Also worth knowing

- **`jaktestowac/awesome-copilot-for-testers`** — a separate community repo focused
  on test automation, if QA/testing is a large part of your role.

## Spend discipline when pulling

- Pull the **collection matching your stack**, not the whole repo — unused files
  are dead context.
- Prefer **instruction files** (always-on, 0 extra requests) over agents you have to
  invoke, for things that should just always apply.
- After installing, fold anything redundant into the kit's existing files so the
  model is not reading two overlapping guides.
