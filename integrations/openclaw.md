# OpenClaw

## What it is

Open-source personal AI assistant running locally on Mac/Windows/Linux, exposed via chat clients (WhatsApp, Telegram, Discord, Slack). Handles email, calendar, forms, file system, browser, and shell automation with persistent memory and pluggable skills. Model-agnostic — OpenAI, Anthropic, local models.

## Relationship to Claude Code

Does **not** embed Claude Code. Marketing/testimonials reference running "autonomous Claude Code loops" remotely via chat — i.e., it can spawn/manage CC sessions as a tool, not share runtime. Shares LLM backend (Anthropic API) only incidentally.

> ⚠️ **Caveat**: The Claude Code integration angle is based on homepage testimonial copy, not verified from the GitHub README. Treat as marketing until confirmed in docs.

## Install sketch

```bash
npm install -g openclaw@latest
openclaw onboard --install-daemon
# Requires Node 24 (or 22.16+); pnpm for source builds
# Configure model provider (OpenAI/Anthropic/local) via OAuth or API key
```

## Use cases

Concrete examples from testimonials and docs:

- **Inbox triage** — Clear email and unsubscribe from lists, all from a chat message
- **Calendar + commute** — Check the day's schedule and ping you when to leave based on traffic
- **Flight check-ins** — Autonomously handle airline check-ins and boarding passes
- **Wearables / health** — Pull WHOOP biometrics and send daily summaries with recovery tips
- **Autonomous coding loops** — Kick off test/fix loops from your phone; OpenClaw opens PRs in response to webhook-captured errors (this is the "autonomous Claude Code loops" angle)
- **Todoist automation** — Custom skills that manipulate Todoist entirely from chat
- **Smart home / IoT** — Control air purifiers and other devices toward personal optimization goals (e.g. air quality)
- **Web automation** — Fill forms, scrape sites, build custom tools (e.g. multi-provider flight search CLI)
- **Cross-channel notes** — Stitch together conversations across Slack/Discord/Telegram into unified docs
- **Generative side-projects** — Write custom guided meditations with TTS + ambient audio

## Why it's interesting for CC users

The autonomous-coding-loop angle is the relevant one: OpenClaw sits on your laptop, exposes a chat interface via messaging apps, and can invoke CC sessions in response to webhooks or chat commands. So you can "tell Claude Code to fix the failing test" from your phone on the train, and get a PR link back on Telegram when it's done.

## Links

- Homepage: <https://openclaw.ai>
- GitHub: <https://github.com/openclaw/openclaw>
