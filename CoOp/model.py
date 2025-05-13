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

        # Get model's dtype
        dtype = self.clip_model.dtype

        # Initialize context vectors (prompt tokens)
        if ctx_init is not None:
            # Optionally initialize with some text
            ctx_init_tokens = clip.tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(
                    ctx_init_tokens).mean(dim=1)
            # Ensure embedding is on the correct device and dtype before creating Parameter
            # embedding shape is [1, ctx_dim] if ctx_init is a single string
            self.ctx = nn.Parameter(embedding.to(
                device=self.device, dtype=dtype).repeat(n_ctx, 1))
        else:
            # Random initialization
            self.ctx = nn.Parameter(torch.randn(
                n_ctx, ctx_dim, device=self.device, dtype=dtype))

        # Tokenize classnames
        # clip.tokenize returns CPU tensors. Move to specified device.
        self.name_tokens = [clip.tokenize(
            f"a photo of a {name}, a type of flower.").to(self.device) for name in classnames]

        # Stack tokenized classnames for EOT finding later
        self.name_tokens_tensor = torch.cat(self.name_tokens, dim=0)

        # Get token embeddings once during initialization
        with torch.no_grad():
            self.token_embeddings = []
            for tokens_tensor_for_class in self.name_tokens:  # Iterate over list of tensors
                # tokens_tensor_for_class has shape [1, max_seq_len]
                # Squeeze to [max_seq_len] for token_embedding if it expects 1D input for single sequence
                # However, clip_model.token_embedding usually expects [batch, seq_len]
                # If tokens_tensor_for_class is [1, L], then token_embedding(tokens_tensor_for_class) is [1, L, D]
                # Squeezing result to [L, D]
                embeddings = self.clip_model.token_embedding(
                    tokens_tensor_for_class).squeeze(0)
                self.token_embeddings.append(embeddings)

    def forward(self):
        # Build prompts by concatenating context and class tokens
        prompts = []
        for i in range(len(self.classnames)):  # Iterate based on number of classes
            # Get the original token embeddings for the i-th class
            # self.token_embeddings[i] has shape [L_clip, D_embed]
            class_specific_embeddings = self.token_embeddings[i]

            # Extract the relevant parts
            # SOS token embedding: class_specific_embeddings[0]
            # CLS token embeddings (part after context): class_specific_embeddings[1+n_ctx:]
            # This assumes class name tokens start right after the context
            # The original prompt is "a photo of a {CLASS}, a type of flower."
            # Tokenized: [SOS] [a] [photo] [of] [a] [CLASS] [,] [a] [type] [of] [flower] [.] [EOS] [PAD]...
            # If self.ctx replaces e.g. "[a] [photo] [of] [a]" (n_ctx=4),
            # then prefix is [SOS]
            # suffix is [CLASS] [,] ... [EOS]
            # This means the class name itself is part of the suffix.

            # The prompt structure from the paper is typically:
            # [v1]...[vM][CLS1]...[CLSk]
            # Where v are learnable context vectors, and CLS are class name tokens.
            # The current code implements: [SOS] [v1]...[vM] [suffix_from_original_prompt]
            # where suffix_from_original_prompt includes the class name and EOS.

            prefix = class_specific_embeddings[:1]  # SOS token embedding
            # Embeddings after context
            suffix = class_specific_embeddings[1 + self.n_ctx:]

            # Concatenate with learnable context
            # self.ctx has shape [n_ctx, ctx_dim]
            # prefix has shape [1, ctx_dim]
            # suffix has shape [L_clip - 1 - n_ctx, ctx_dim]
            # Total length: 1 + n_ctx + (L_clip - 1 - n_ctx) = L_clip. This is correct.
            prompt = torch.cat([
                prefix,      # Start token embedding
                self.ctx,    # Context token embeddings (learnable)
                # Rest of the embeddings (including class name and EOS)
                suffix
            ], dim=0)

            prompts.append(prompt)

        # Shape: [num_classes, L_clip, D_embed]
        prompts = torch.stack(prompts).to(self.device)

        # `prompts` are now embeddings, not token IDs.
        # Replicate CLIP's text_encoder logic, starting from embeddings.
        x = prompts + \
            self.clip_model.positional_embedding.type(self.clip_model.dtype)
        # NLD -> LND (N=num_classes, L=L_clip, D=D_embed)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # Extract features at EOT token positions
        # self.name_tokens_tensor has shape [num_classes, L_clip] and contains token IDs
        # Finds index of EOT token for each class prompt
        eot_indices = self.name_tokens_tensor.argmax(dim=-1)

        # x has shape [num_classes, L_clip, D_transformer_width]
        # We need features corresponding to EOT token for each item in the batch (num_classes)
        text_features = x[torch.arange(
            x.shape[0], device=self.device), eot_indices] @ self.clip_model.text_projection

        return text_features
