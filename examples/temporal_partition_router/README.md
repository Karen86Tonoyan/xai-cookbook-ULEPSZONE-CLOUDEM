# Temporal Partition Router

This example turns the ALFA Guardian temporal routing idea into a compact cookbook pattern.

Every prompt is labeled before model dispatch:

- `yesterday`: recall, history, lessons, memory
- `today`: active execution, debugging, current state
- `tomorrow`: planning, forecasting, strategy

The router then chooses temperature, token budget, memory window, and system prompt before calling Grok, Ollama, or any OpenAI-compatible model.

Run:

```bash
python temporal_router.py
```
