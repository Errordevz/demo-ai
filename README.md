# Demo AI

Demo AI is a 100M-parameter decoder-only Transformer with a cloud chat website and a separate developer API.

## Cloud chat

The public website calls \`POST /api/chat\`. **No \`DEMO_AI_KEY\` is required** for normal website chat.

## Developer API

The API section generates a cryptographically random developer key. Programmatic inference uses:

\`\`\`http
POST /api/v1/chat
Authorization: Bearer DEMO_AI_KEY
Content-Type: application/json
\`\`\`

The signing secret is deployment-only (\`DEMO_AI_API_SIGNING_SECRET\`) and should never be committed to the repository.

## Cloud deployment

GitHub Actions builds the model artifact from the reproducible training script, publishes \`demo-ai-cloud-runtime.zip\` as the \`cloud-latest\` release asset, and Vercel fetches that asset during deployment. The model itself is **not** committed to Git history.

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

## Local download

The portable development package remains available from the GitHub Releases page. It includes source, training code, tokenizer and model artifacts.

