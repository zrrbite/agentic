# Workflow example — the kit end-to-end

One realistic task, start to finish, showing the literal text you type, which chat
mode and model tier each step uses, and where the premium-request spend goes (or
does not). It ties together every piece of the kit: instructions, prompts, agents,
personas, the routing card, and the spend audit.

**Task:** add retry-with-backoff to a flaky TypeScript upload client.

> The through-line — consult [`MODEL-ROUTING-CHEAT-CARD.md`](MODEL-ROUTING-CHEAT-CARD.md)
> at every step: **plan and grunt-work on 0×, the body on Auto's 0.9×, premium
> reserved for the one genuinely hard moment** (which usually never arrives).

---

## 0 — Standing setup (once)

- Model picker → **Auto** (built-in 0.9× discount).
- `copilot-instructions.md` + `typescript.instructions.md` already loaded — **0
  requests, always on** — silently enforcing strict types, zod-at-boundary, and
  no swallowed errors.
- Chat mode dropdown → **terse-senior** as your default driver.

See [`INSTALL.md`](INSTALL.md) if any of these are not yet in place.

## 1 — Think first, cheaply

**Mode:** `ask` · **Model:** 0×/Auto

```
/plan add exponential backoff with jitter to UploadClient.upload(); retry only
on 5xx and network errors, max 4 attempts, surface the final error
```

→ Affected files + ordered steps + risks. **No edits, near-free.** This is the step
that stops you paying premium requests to execute a half-formed idea.

## 2 — Lock the plan, then build

**Mode:** confirm-first · **Model:** Auto

```
implement the plan
```

→ It restates the plan; you reply `go`; it edits. The instruction files do the
quality enforcement for free. One pass, no scope creep, no surprise rewrites.

## 3 — Tests

**Mode:** `agent` (or the `test-author` agent) · **Model:** Auto

```
/add-tests
```

→ Detects Vitest, writes happy-path + edge cases (max-attempts exhausted,
non-retryable 4xx, abort mid-retry), shows `pnpm test`.

## 4 — A test fails

**Mode:** `agent` · **Model:** Auto

```
/fix-failing-test
```

→ Reads the assertion, states out loud whether the code or the test is wrong, fixes
the root cause, re-runs the one test.

## 5 — Adversarial review

**Mode:** savage-reviewer · **Model:** Auto (escalate to premium only if a finding
needs deep cross-file reasoning)

```
review the diff
```

→ Blunt, ranked findings as `file:line — wrong — why — fix`.

Specialized cuts, when warranted:
- Security only → invoke the **`security-reviewer`** agent.
- New/changed dependencies → invoke the **`dependency-auditor`** agent (runs
  `pnpm audit`).

## 6 — Caveman for the grunt fixes

**Mode:** caveman · **Model:** 0×

```
apply review fixes
```

→ "Fixed `upload.ts:88` — unhandled rejection. Added `await`. Run `pnpm test`."
Terse, fast to scan, no padding.

## 7 — Ship

**Mode:** `ask` · **Model:** 0×

```
/commit-msg
/pr-description
```

→ Paste-ready commit message + PR body generated from the actual diff — no
fabricated testing claims or invented rationale.

## 8 — Friday

Open [`SPEND-AUDIT.md`](SPEND-AUDIT.md), spend five minutes on the usage dashboard.
The number that matters: **model mix** — was routine work on 0×/Auto, and did any
50× sneak in?

---

## Why this is cheap

| Step | Mode | Model tier | Premium cost |
|---|---|---|---|
| 1 Plan | ask | 0×/Auto | ~0 |
| 2 Implement | confirm-first | Auto | 0.9× |
| 3 Tests | agent | Auto | 0.9× |
| 4 Fix test | agent | Auto | 0.9× |
| 5 Review | savage-reviewer | Auto | 0.9× |
| 6 Apply fixes | caveman | 0× | 0 |
| 7 Commit/PR | ask | 0× | 0 |

The whole feature costs a handful of **discounted** requests instead of a stream of
premium ones — and the `confirm-first` / answer-first personas mean almost no
corrective re-prompts, which is where requests quietly leak.

## Other-language quick mapping

Same arc, different tooling — the instruction files adapt automatically by glob:

- **C/C++** → `test-author`/GoogleTest, `cpp-modernizer`, build via `ctest`. Pair
  reviews with the `security-reviewer` (memory safety).
- **Python** → Vitest→pytest, `pnpm audit`→`pip-audit`, lint via ruff.
- **Rust** → tests via `cargo test`, audit via `cargo audit`, `review the diff`
  leans on clippy-clean expectations.

## See also

- [`README.md`](README.md) — index of every file
- [`INSTALL.md`](INSTALL.md) — install + how to invoke each piece
- [`MODEL-ROUTING-CHEAT-CARD.md`](MODEL-ROUTING-CHEAT-CARD.md) — task → model tier
- [`TOKEN-SAVING-PLAYBOOK.md`](TOKEN-SAVING-PLAYBOOK.md) — the economics & levers
