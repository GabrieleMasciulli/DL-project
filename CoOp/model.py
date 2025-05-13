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
        # Build prompts by concatenating context and class tokens
        prompts = []
        for tokens in self.name_tokens:
            # Remove the start token, insert context, then add the rest
            tokens = tokens.squeeze(0)

            # Reshape tokens to match ctx dimensions
            start_token = tokens[0].reshape(1, -1)  # [1, dim]
            # [rest, dim]
            end_tokens = tokens[1 + self.n_ctx:].reshape(-1, self.ctx_dim)

            # Now concatenate along dimension 0
            prompt = torch.cat([
                start_token,  # [1, dim]
                self.ctx,     # [n_ctx, dim]
                end_tokens    # [rest, dim]
            ], dim=0)

            prompts.append(prompt)
        prompts = torch.stack(prompts).to(self.device)
        # Encode prompts using CLIP's text encoder
        text_features = self.clip_model.encode_text(prompts)
        return text_features
