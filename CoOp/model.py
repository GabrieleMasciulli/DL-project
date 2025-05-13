import torch
import torch.nn as nn
import clip


class CoOp(nn.Module):
    """
    CoOp model for zero-shot classification.

    Parameters:
    - clip_model: CLIP model (e.g., ViT-B/32)
    - classnames: List of class names (base classes only)
    - ctx_init: Optional initial context for prompt tokens
    - n_ctx: Number of context tokens
    - ctx_dim: Dimension of context vectors
    - device: Device to run the model on
    """

    def __init__(self, clip_model, classnames, ctx_init=None, n_ctx=16, ctx_dim=512, device="cpu"):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.device = device

        # Initialize context vectors (prompt tokens)
        if ctx_init is not None:
            # Optionally initialize with some text
            ctx_init_tokens = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(
                    ctx_init_tokens).mean(dim=1)
            self.ctx = nn.Parameter(embedding.repeat(n_ctx, 1))
        else:
            # Random initialization
            self.ctx = nn.Parameter(torch.randn(n_ctx, ctx_dim))

        # Tokenize classnames
        self.name_tokens = [clip.tokenize(
            f"a photo of a {name}, a type of flower.").to(device) for name in classnames]

    def forward(self):
        # Build prompts by inserting learnable context into token embeddings
        prompts = []
        for i, tokens in enumerate(self.name_tokens):
            # tokens: [1, token_length]
            tokens = tokens.to(self.device)
            # Get token embeddings: [token_length, ctx_dim]
            embeddings = self.clip_model.token_embedding(tokens).squeeze(0)
            # Find where to insert the context (after the start token)
            prefix = embeddings[:1]
            suffix = embeddings[1 + self.n_ctx:]
            # Concatenate: [start] + [ctx] + [rest]
            prompt = torch.cat([
                prefix,
                self.ctx,
                suffix
            ], dim=0)
            prompts.append(prompt)

        # Stack prompts: [num_classes, token_length, ctx_dim]
        prompts = torch.stack(prompts).to(self.device)

        # Pass through transformer and the rest of the text encoder
        # (following CLIP's encode_text implementation)
        x = prompts
        x = x + self.clip_model.positional_embedding
        x = self.clip_model.transformer(x)
        x = self.clip_model.ln_final(x)
        # Take features at the end-of-text token position (assume it's the same for all)
        eos_token_id = self.clip_model.token_embedding.weight.shape[0] - 1
        # Find the position of the <EOT> token in each prompt
        # (CLIP uses 49407 as <EOT> by default)
        eot_indices = (self.name_tokens[0] == self.clip_model.tokenizer.eot_token).nonzero(
            as_tuple=True)[1].item()
        text_features = x[torch.arange(x.shape[0]), eot_indices]
        # Normalize
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)
        return text_features
