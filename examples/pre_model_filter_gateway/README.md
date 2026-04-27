# Pre-Model Filter Gateway

This example demonstrates a provider-independent filter gateway that runs before any AI model call.

The important architectural rule:

> The model cannot override, modify, or negotiate pre-model filters because blocked input is never sent to the model.

The gateway includes:

- context gate
- confidence gate
- DLP scanner
- permission gate
- jailbreak / override gate
- input normalization

Run:

```bash
python filter_gateway.py
```
