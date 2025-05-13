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

        # Get token embeddings once during initialization
        with torch.no_grad():
            self.token_embeddings = []
            for tokens in self.name_tokens:
                # Get the token embeddings from CLIP
                embeddings = self.clip_model.token_embedding(tokens.squeeze(0))
                self.token_embeddings.append(embeddings)

    def forward(self):
        # Build prompts by concatenating context and class tokens
        prompts = []
        for i, tokens in enumerate(self.name_tokens):
            # Get the original token embeddings
            embeddings = self.token_embeddings[i]

            # Extract the relevant parts
            prefix = embeddings[:1]
            suffix = embeddings[1 + self.n_ctx:]

            # Concatenate with learnable context
            prompt = torch.cat([
                prefix,      # Start token
                self.ctx,    # Context tokens
                suffix       # Rest of the embeddings
            ], dim=0)

            prompts.append(prompt)

        prompts = torch.stack(prompts).to(self.device)
        # Encode prompts using CLIP's text encoder
        text_features = self.clip_model.encode_text(prompts)
        return text_features
