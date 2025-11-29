import torch
import torch.nn as nn
import clip


@torch.no_grad()
def get_tokenized_classnames(classnames, tokenizer, device):
    """Tokenise only the fixed template "a photo of a {cls}." for each class."""
    prompts = [f"a photo of a {name}." for name in classnames]
    return tokenizer(prompts).to(device)


class CoOp(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=16, ctx_dim=512, device="cpu"):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.device = device

        clip_dtype = clip_model.token_embedding.weight.dtype

        # shared: [n_ctx, ctx_dim]
        ctx_shape = (n_ctx, ctx_dim)
        self.ctx = nn.Parameter(torch.normal(
            0, 0.02, size=ctx_shape, dtype=clip_dtype))

        # 2.  Tokenise only the class-name template ----------------------------
        self.tokenized_prompts = get_tokenized_classnames(
            classnames, clip.tokenize, device)

        # 3.  Pre-compute original EOT positions for each class -----------------
        # In CLIP tokenizer the highest token id is the EOT token, so argmax gives the index.
        self.register_buffer(
            "orig_eot_idx", self.tokenized_prompts.argmax(dim=-1))

        # 4.  CLIP sub-modules --------------------------------------------------
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def build_embeddings(self):
        """Return embeddings with context inserted right after SOS."""
        # Base embeddings of the plain "a {cls}." prompts
        base = self.token_embedding(
            self.tokenized_prompts.to(self.device))  # [n_cls, 77, dim]
        batch, seq_len, dim = base.shape

        # Broadcast shared ctx from [n_ctx, ctx_dim] -> [batch, n_ctx, ctx_dim]
        ctx = self.ctx.unsqueeze(0).expand(batch, -1, -1).to(self.device)

        # Construct new embeddings tensor
        new_embeddings = torch.zeros_like(base)

        for i in range(batch):
            e: Any = base[i]
            sos = e[0:1]  # [1, dim]
            cls_tokens = e[1:self.orig_eot_idx[i]]  # without SOS/EOT
            eot: Any = e[self.orig_eot_idx[i]:self.orig_eot_idx[i]+1]  # [1, dim]

            # Assemble: SOS | ctx | class tokens | EOT
            assembled = torch.cat([sos, ctx[i], cls_tokens, eot], dim=0)
            new_embeddings[i, :assembled.size(0), :] = assembled
        return new_embeddings

    def forward(self):
        x = self.build_embeddings()  # [n_cls, 77, dim]

        # Positional + transformer
        x = x + self.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x.to(self.clip_model.dtype))
        x = x.permute(1, 0, 2)

        # Layer norm + projection
        x = self.ln_final(x).type(self.clip_model.dtype)

        # Updated EOT index = original + n_ctx (shift after inserting ctx)
        eot_idx = self.orig_eot_idx + self.n_ctx
        feats = x[torch.arange(x.size(0)), eot_idx] @ self.text_projection
        return feats
