import torch
import torch.nn as nn
import clip


@torch.no_grad()
def get_tokenized_prompts(classnames, tokenizer, device):
    prompts = [
        f"a photo of a {' '.join(['X'] * 16)} {name}, a type of flower." for name in classnames
    ]
    return tokenizer(prompts).to(device)


class CoOp(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=16, ctx_dim=512, device="cpu"):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.device = device

        # Get CLIP's dtype
        clip_dtype = clip_model.token_embedding.weight.dtype

        # Initialize ctx with CLIP's dtype
        self.ctx = nn.Parameter(torch.normal(
            0, 0.02, size=(n_ctx, ctx_dim), dtype=clip_dtype))

        # Tokenize template prompts
        self.tokenized_prompts = get_tokenized_prompts(
            classnames, clip.tokenize, device)

        # CLIP components
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self):
        embeddings = self.token_embedding(
            self.tokenized_prompts.to(self.device))

        # Replace context tokens
        ctx = self.ctx.unsqueeze(0).expand(embeddings.size(0), -1, -1)
        embeddings[:, 1:1+self.n_ctx, :] = ctx

        # Add positional embeddings
        x = embeddings + self.positional_embedding.to(embeddings.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Ensure transformer input dtype
        x = self.transformer(x.to(self.clip_model.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Final processing
        x = self.ln_final(x).type(self.clip_model.dtype)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(
            dim=-1)] @ self.text_projection

        return x
