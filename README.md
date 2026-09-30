# Demo AI

Demo AI is a 100M-parameter decoder-only Transformer with a cloud chat website and a separate developer API.

## Cloud chat

The public website calls `POST /api/chat`. **No `DEMO_AI_KEY` is required** for normal website chat.

## Developer API

The API section generates a cryptographically random developer key. Programmatic inference uses:

```http
POST /api/v1/chat
Authorization: Bearer DEMO_AI_KEY
Content-Type: application/json
```

Keys are generated and validated in deployment memory. They are shown once and are lost when the service restarts/redeploys; this hosted demo does not yet provide durable key storage.

## Hosting

The repository contains both a Vercel configuration and a Render Docker configuration.

- Vercel can host the static site and Python functions, but building a 100M model at deployment time is resource-intensive.
- Render's free web service can run the complete Docker app for hobby/testing use. Free services have 512 MB RAM, 0.1 CPU, and spin down after 15 minutes of inactivity.

## Model

- 100,000,000 parameters
- 1,024-token context
- 17 layers
- 800-dimensional hidden state
- 8 query heads / 2 KV heads (GQA)
- SwiGLU + RMSNorm + RoPE
- tied input/output embeddings
- cloud weight packaging: row-wise INT4

The cloud checkpoint is a development model. Cloud availability does not imply frontier-level or professional training quality.

## Security notes

- Public chat rate limiting uses a process-scoped digest of the proxy client address when available, never the raw address.
- API keys are stored only as SHA-256 digests in deployment memory.
- No secrets or model binaries are committed to Git history by default.
