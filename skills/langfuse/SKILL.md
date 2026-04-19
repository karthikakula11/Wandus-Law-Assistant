---
name: langfuse
description: Interact with Langfuse and access its documentation. Use when needing to (1) query or modify Langfuse data programmatically via the CLI — traces, prompts, datasets, scores, sessions, and any other API resource, (2) look up Langfuse documentation, concepts, integration guides, or SDK usage, or (3) understand how any Langfuse feature works. This skill covers CLI-based API access (via npx) and multiple documentation retrieval methods.
---

# Langfuse

This is a **vendored copy** of the official skill from [github.com/langfuse/skills](https://github.com/langfuse/skills). For updates, re-fetch:

```bash
curl -sL https://raw.githubusercontent.com/langfuse/skills/main/skills/langfuse/SKILL.md -o skills/langfuse/SKILL.md
```

See the upstream repo for full `references/` (instrumentation, CLI, etc.) and installation via Cursor `/add-plugin langfuse` or `npx skills add langfuse/skills --skill "langfuse"`.

## Core principles (summary)

1. **Documentation first** — fetch current Langfuse docs when implementing.
2. **Prefer framework integrations** — OpenAI drop-in, LangChain `CallbackHandler`, etc.
3. **Baseline trace quality** — model name, tokens, descriptive names, nested spans, avoid leaking secrets in trace input.

Full upstream content: [langfuse/skills on GitHub](https://github.com/langfuse/skills).
