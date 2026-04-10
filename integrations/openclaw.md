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

- Drive Claude Code (or other agents) from WhatsApp/Telegram while away from the laptop
- Personal task automation (email, calendar, forms)
- Chat-first interface over local file system + browser + shell

## Links

- Homepage: <https://openclaw.ai>
- GitHub: <https://github.com/openclaw/openclaw>
