import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import OxfordFlowers102
from tqdm import tqdm
import numpy as np
from model import get_tokenized_prompts  # Import for use in eval_cocoop


def split_data(dataset: Dataset, categories: list[int]) -> tuple[Subset, Subset]:
    idx = [i for i, (_, label) in enumerate(dataset) if label in categories]
    return Subset(dataset, idx), Subset(dataset, [i for i in range(len(dataset)) if i not in idx])


def get_data(transform=None) -> tuple[OxfordFlowers102, OxfordFlowers102, OxfordFlowers102]:
    if transform is None:
        # Default transform if none provided (e.g., for initial inspection)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    train_set = OxfordFlowers102(
        root="./data", split="train", download=True, transform=transform)
    val_set = OxfordFlowers102(
        root="./data", split="val", download=True, transform=transform)
    test_set = OxfordFlowers102(
        root="./data", split="test", download=True, transform=transform)
    return train_set, val_set, test_set


def train_cocoop(
    model,
    clip_model_visual,
    train_loader,
    optimizer,
    criterion,
    epochs,
    device,
    categories,  # Global indices of base classes model is trained on
):
    print(f"Training CoCoOp on {len(categories)} base classes.")
    # Create a mapping from global category index to local index (0 to N_base_classes-1)
    # These are the classes the CoCoOp model's `forward` method will produce logits for.
    global_to_local_label_map = {
        global_idx: local_idx for local_idx, global_idx in enumerate(categories)}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [T]", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Filter out labels not in the current training categories (base classes)
            # and map them to local indices
            valid_indices = [i for i, l_item in enumerate(
                labels.tolist()) if l_item in global_to_local_label_map]
            if not valid_indices:
                continue

            images = images[valid_indices]
            global_labels_for_batch = labels[valid_indices]
            local_labels = torch.tensor(
                [global_to_local_label_map[l.item()] for l in global_labels_for_batch], device=device)

            optimizer.zero_grad()

            with torch.no_grad():
                image_features = clip_model_visual(
                    images.to(model.clip_model.dtype))
                image_features = image_features / \
                    image_features.norm(dim=-1, keepdim=True)

            # Forward pass: model handles all text feature construction and normalization
            logits = model(image_features)

            # Compute logits: (B, N_cls, D) x (B, D) -> (B, N_cls)
            # If model returns text features, compute similarity here
            if logits.dim() == 3:
                # (B, N_cls, D) and (B, D) -> (B, N_cls)
                logits = torch.einsum('bd,bcd->bc', image_features, logits)
                logits *= model.clip_model.logit_scale.exp()

            loss = criterion(logits, local_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {epoch_loss:.4f}")

    return model


def eval(
    cocoop_model,
    clip_model_visual,
    dataset,
    eval_categories,
    all_class_names,
    batch_size,
    device,
    clip_tokenizer,
    label="Evaluation"
):
    print(f"\n🔍 {label} on {len(eval_categories)} categories...")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    cocoop_model.eval()
    correct_predictions = 0
    total_samples = 0

    # Create a mapping from global category index to local index (0 to N_eval_categories-1)
    eval_global_to_local_map = {
        global_idx: local_idx for local_idx, global_idx in enumerate(eval_categories)}
    # Names of the classes being evaluated in this run
    current_eval_classnames = [all_class_names[i] for i in eval_categories]

    # Tokenize prompts for the current evaluation categories
    # These are the text prompts we will generate features for, using the dynamic context
    tokenized_eval_prompts = get_tokenized_prompts(
        current_eval_classnames, clip_tokenizer, device, cocoop_model.n_ctx
    ).to(device)

    # Pre-compute embeddings for the static parts of the evaluation prompts
    # Shape: (N_eval_cls, L_prompt, D_embed)
    prompt_embeddings_template = cocoop_model.token_embedding(
        tokenized_eval_prompts)

    progress_bar = tqdm(data_loader, desc=label, leave=False)
    with torch.no_grad():
        for images, global_labels in progress_bar:
            images, global_labels = images.to(device), global_labels.to(device)

            # Filter out labels not in the current eval_categories and map to local
            valid_indices = [i for i, l_item in enumerate(
                global_labels.tolist()) if l_item in eval_global_to_local_map]
            if not valid_indices:
                continue

            images = images[valid_indices]
            global_labels_for_batch = global_labels[valid_indices]
            # Target labels are local to the `eval_categories`
            target_local_labels = torch.tensor(
                [eval_global_to_local_map[l.item()] for l in global_labels_for_batch], device=device)

            # 1. Get image features
            image_features_full = clip_model_visual(
                images.to(cocoop_model.clip_model.dtype))

            # 2. Generate delta_ctx using MetaNet
            # Ensure image_features_full are on the correct device and dtype for MetaNet
            delta_ctx = cocoop_model.meta_net(
                image_features_full.to(cocoop_model.meta_net.fc1.weight.dtype))

            # 3. Form dynamic_ctx
            # dynamic_ctx shape: (B_img, n_ctx, ctx_dim)
            dynamic_ctx = cocoop_model.ctx.unsqueeze(0) + delta_ctx

            # 4. Construct text features for `current_eval_classnames` using `dynamic_ctx`
            B_img = image_features_full.shape[0]
            N_eval_cls = len(current_eval_classnames)
            L_prompt = tokenized_eval_prompts.shape[1]

            # Expand prompt templates for each image in the batch
            # Shape: (B_img, N_eval_cls, L_prompt, D_embed)
            expanded_prompt_embeddings = prompt_embeddings_template.unsqueeze(
                0).expand(B_img, -1, -1, -1)

            # Expand dynamic_ctx for each class being evaluated
            # Shape: (B_img, N_eval_cls, n_ctx, D_embed)
            expanded_dynamic_ctx = dynamic_ctx.unsqueeze(
                1).expand(-1, N_eval_cls, -1, -1)

            # Create the final embeddings by inserting the dynamic context
            final_embeddings = expanded_prompt_embeddings.clone()
            final_embeddings[:, :, 1:1+cocoop_model.n_ctx,
                             :] = expanded_dynamic_ctx

            # Reshape for transformer: (B_img * N_eval_cls, L_prompt, D_embed)
            x = final_embeddings.view(
                B_img * N_eval_cls, L_prompt, cocoop_model.ctx_dim)

            # Add positional embeddings
            x = x + cocoop_model.positional_embedding.to(x.dtype)

            # Pass through CLIP's text transformer
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = cocoop_model.transformer(x.to(cocoop_model.clip_model.dtype))
            x = x.permute(1, 0, 2)  # LND -> NLD

            # Final layer normalization and projection
            x = cocoop_model.ln_final(x).type(cocoop_model.clip_model.dtype)

            eos_indices = tokenized_eval_prompts.argmax(dim=-1)
            eos_indices_expanded = eos_indices.unsqueeze(
                0).expand(B_img, -1).reshape(B_img * N_eval_cls)

            text_features_eval = x[torch.arange(
                x.shape[0]), eos_indices_expanded] @ cocoop_model.text_projection
            text_features_eval = text_features_eval.view(
                B_img, N_eval_cls, -1)  # (B_img, N_eval_cls, D_text_feat)

            # Normalize features
            image_features_norm = image_features_full / \
                image_features_full.norm(dim=-1, keepdim=True)
            text_features_eval_norm = text_features_eval / \
                text_features_eval.norm(dim=-1, keepdim=True)

            # Compute logits: (B, D) x (B, C, D) -> (B, C)
            logits = torch.einsum(
                'bd,bcd->bc', image_features_norm, text_features_eval_norm)
            # Use logit_scale from the loaded CLIP model
            logits *= cocoop_model.clip_model.logit_scale.exp()

            # Predictions are local to eval_categories
            _, predictions = torch.max(logits, 1)

            correct_predictions += (predictions ==
                                    target_local_labels).sum().item()
            total_samples += target_local_labels.size(0)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(
        f"📊 {label} Accuracy: {accuracy*100:.2f}% ({correct_predictions}/{total_samples})")
    return accuracy


def harmonic_mean(h_new, h_base):
    return 2 * h_new * h_base / (h_new + h_base) if (h_new + h_base) != 0 else 0
