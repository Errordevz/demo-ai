# Demo AI

Demo AI is a 100,000,000-parameter sparse Mixture-of-Experts language model with a cloud chat website and a separate developer API.

## Cloud chat

The public website calls POST /api/chat. No DEMO_AI_KEY is required for normal website chat.

## Sparse MoE

The exact 100M budget is split across a shared Transformer trunk and four expert FFNs per layer. A learned router selects the top 2 of 4 experts per token, so only half of the expert parameter pool is active for a token.

Architecture:
- 100,000,000 total parameters
- 12 Transformer layers
- 800-dimensional hidden state
- 8 query heads / 2 KV heads
- 4 experts per layer, top-2 routing
- expert FFN width 644
- 1,024-token context
- SwiGLU + RMSNorm + RoPE
- tied input/output embeddings
- row-wise INT4 cloud weights

This is a sparse conditional-compute design: only a subset of expert parameters are selected per token. The router also gets a load-balancing auxiliary objective during training so it does not collapse onto a small number of experts. See the Switch Transformer and load-balancing literature for background.

## Developer API

The API section generates a cryptographically random developer key. Programmatic inference uses:
POST /api/v1/chat
Authorization: Bearer DEMO_AI_KEY
Content-Type: application/json

Keys are generated and validated in deployment memory. They are shown once and are lost when the service restarts/redeploys; this hosted demo does not yet provide durable key storage.

## Hosting

The repository supports Render Python hosting and Docker. The host downloads the published cloud runtime artifact from the public cloud-latest GitHub release instead of retraining the model during every web-service build.

## Model quality

The hosted checkpoint is a development bootstrap, not a professionally pretrained language model. The sparse architecture is implemented and deployable, but strong coding/general/reasoning quality still requires substantially more data and training compute.

## Security notes

- Public chat rate limiting uses a process-scoped digest of the proxy client address when available, never the raw address.
- API keys are stored only as SHA-256 digests in deployment memory.
- The model release is public because this repository is public; no private credentials belong in source control.
