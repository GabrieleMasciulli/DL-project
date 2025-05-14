import torch
import torch.nn as nn
import clip


@torch.no_grad()
def get_tokenized_prompts(classnames, tokenizer, device, n_ctx=16):  # Added n_ctx argument
    prompts = [
        f"a photo of a {' '.join(['X'] * n_ctx)} {name}." for name in classnames]  # Use n_ctx
    return tokenizer(prompts).to(device)


class MetaNet(nn.Module):
    """
    A Meta-Net for generating instance-specific context adjustments.
    Consists of a 2-layer MLP with a bottleneck design.
    """

    def __init__(self, vis_dim, n_ctx, ctx_dim, bottleneck_reduction=16, dtype=torch.float32, drop=0.0):
        super().__init__()
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        hidden_dim = vis_dim // bottleneck_reduction

        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(vis_dim, hidden_dim, dtype=dtype)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_ctx * ctx_dim, dtype=dtype)

    def forward(self, image_features):
        """
        Args:
            image_features (torch.Tensor): Visual features from the image encoder.
                                           Shape: (batch_size_img, vis_dim)
        Returns:
            torch.Tensor: Delta context vectors. Shape: (batch_size_img, n_ctx, ctx_dim)
        """
        x = self.fc1(image_features)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        # Reshape to (batch_size_img, n_ctx, ctx_dim)
        delta_ctx = x.view(-1, self.n_ctx, self.ctx_dim)
        return delta_ctx


class CoCoOp(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=16, ctx_dim=512, vis_dim=512, device="cpu", dropout=0.0):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.vis_dim = vis_dim  # Dimension of visual features
        self.device = device

        # Get CLIP's dtype for parameters
        clip_dtype = clip_model.token_embedding.weight.dtype

        # Initialize shared context (learnable)
        self.ctx = nn.Parameter(torch.normal(
            0, 0.02, size=(n_ctx, ctx_dim), dtype=clip_dtype))

        # Initialize Meta-Net
        # Ensure MetaNet parameters are on the same device and dtype
        self.meta_net = MetaNet(vis_dim, n_ctx, ctx_dim, dropout=dropout,
                                dtype=clip_dtype).to(device)

        # Tokenize template prompts (these are static parts of the prompt)
        self.tokenized_prompts = get_tokenized_prompts(
            classnames, clip.tokenize, device, n_ctx)

        # CLIP components
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, image_features):
        """
        Constructs normalized text features for each image/class combination.
        Args:
            image_features: (B_img, vis_dim)
            tokenized_prompts: (N_cls, L_prompt)
        Returns:
            torch.Tensor: Text features for all classes, for each image.
                          Shape: (B_img, N_cls, D_text_feat)
        """
        # Ensure image_features are on the correct device and dtype for MetaNet
        image_features = image_features.to(
            self.device, dtype=self.meta_net.fc1.weight.dtype)

        # Generate instance-specific context adjustment using Meta-Net
        # Shape: (B_img, n_ctx, ctx_dim)
        delta_ctx = self.meta_net(image_features)

        # Combine shared context with instance-specific adjustment
        # self.ctx shape: (n_ctx, ctx_dim)
        # delta_ctx shape: (B_img, n_ctx, ctx_dim)
        # dynamic_ctx shape: (B_img, n_ctx, ctx_dim)
        dynamic_ctx = self.ctx.unsqueeze(0) + delta_ctx

        B_img = image_features.shape[0]
        N_cls = len(self.classnames)
        L_prompt = self.tokenized_prompts.shape[1]  # Max prompt length

        # Get initial embeddings for the tokenized prompts (templates)
        # Shape: (N_cls, L_prompt, D_embed)
        prompt_embeddings_template = self.token_embedding(
            self.tokenized_prompts.to(self.device))

        # Expand prompt templates for each image in the batch
        # Shape: (B_img, N_cls, L_prompt, D_embed)
        expanded_prompt_embeddings = prompt_embeddings_template.unsqueeze(
            0).expand(B_img, -1, -1, -1)

        # Expand dynamic_ctx for each class
        # Shape: (B_img, N_cls, n_ctx, D_embed)
        expanded_dynamic_ctx = dynamic_ctx.unsqueeze(
            1).expand(-1, N_cls, -1, -1)

        # Create the final embeddings by inserting the dynamic context
        final_embeddings = expanded_prompt_embeddings.clone()
        final_embeddings[:, :, 1:1+self.n_ctx, :] = expanded_dynamic_ctx

        # Reshape for transformer: (B_img * N_cls, L_prompt, D_embed)
        x = final_embeddings.view(B_img * N_cls, L_prompt, self.ctx_dim)

        # Add positional embeddings
        # self.positional_embedding shape: (max_clip_len, D_embed)
        x = x + self.positional_embedding.to(x.dtype)

        # Pass through CLIP's text transformer
        x = x.permute(1, 0, 2)  # NLD -> LND (L_prompt, B_img * N_cls, D_embed)
        # Ensure correct dtype for transformer
        x = self.transformer(x.to(self.clip_model.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD (B_img * N_cls, L_prompt, D_embed)

        # Final layer normalization and projection
        x = self.ln_final(x).type(self.clip_model.dtype)

        # Get features corresponding to the [EOS] token
        # self.tokenized_prompts shape: (N_cls, L_prompt)
        # We need EOS indices for each of the (B_img * N_cls) sequences
        eos_indices = self.tokenized_prompts.argmax(dim=-1)  # Shape: (N_cls)
        eos_indices_expanded = eos_indices.unsqueeze(0).expand(
            B_img, -1).reshape(B_img * N_cls)  # Shape: (B_img * N_cls)

        # Select the features at EOS token positions and project
        # x shape before selection: (B_img * N_cls, L_prompt, D_embed)
        # text_features shape: (B_img * N_cls, D_text_feat)
        text_features = x[torch.arange(
            x.shape[0]), eos_indices_expanded] @ self.text_projection

        # Reshape to (B_img, N_cls, D_text_feat)
        text_features = text_features.view(B_img, N_cls, -1)

        return text_features
