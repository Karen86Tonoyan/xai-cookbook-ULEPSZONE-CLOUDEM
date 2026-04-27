# ALFA Context Memory

This example builds a local "context memory" loop for long-running AI work:

1. Read project sources, markdown notes, and zip archives.
2. Skip common secrets and heavy generated folders.
3. Create compact source cards.
4. Save a timestamped snapshot.
5. Render a cumulative context injection for the next session.

Run locally:

```bash
cd examples/alfa_context_memory
python alfa_brain.py snapshot --sources sources.example.json --out memory
python alfa_brain.py inject --memory memory --out memory/context-injection.md
```

Use `memory/context-injection.md` at the start of the next conversation.
