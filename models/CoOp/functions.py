import torch
import torchvision
from torchvision.datasets.flowers102 import Flowers102
from tqdm import tqdm


def get_data(data_dir='./data', transform=None) -> tuple[Flowers102,
                                                         Flowers102,
                                                         Flowers102]:
    """Load Flowers102 train, validation and test sets.
    Args:
        data_dir (str): Directory where the dataset will be stored.
        transform (torch.Compose)
    Returns:
        tuple: A tuple containing the train, validation, and test sets.
    """
    train = torchvision.datasets.Flowers102(
        root=data_dir, split="train", download=True, transform=transform)
    val = torchvision.datasets.Flowers102(
        root=data_dir, split="val", download=True, transform=transform)
    test = torchvision.datasets.Flowers102(
        root=data_dir, split="test", download=True, transform=transform)
    return train, val, test


def split_data(dataset, base_classes):
    # these two lists will store the sample indexes
    base_categories_samples = []
    novel_categories_samples = []

    # we create a set of base classes to compute the test below in O(1)
    # this is optional and can be removed
    base_set = set(base_classes)

    # here we iterate over sample labels and also get the correspondent sample index
    for sample_id, label in enumerate(dataset._labels):
        if label in base_set:
            base_categories_samples.append(sample_id)
        else:
            novel_categories_samples.append(sample_id)

    # here we create the dataset subsets
    # the torch Subset is just a wrapper around the dataset
    # it simply stores the subset indexes and the original dataset (your_subset.dataset)
    # when asking for sample i in the subset, torch will look for its original position in the dataset and retrieve it
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset
    base_dataset = torch.utils.data.Subset(dataset, base_categories_samples)
    novel_dataset = torch.utils.data.Subset(dataset, novel_categories_samples)
    return base_dataset, novel_dataset


def train_one_epoch(model, train_loader, optimizer, criterion, device, categories):
    """Train the CoOp model for one epoch.

    Args:
        model: CoOp model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to run training on
        categories: List of class indices

    Returns:
        tuple: (train_loss, train_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    categories_tensor = torch.tensor(categories, device=device)

    for images, targets in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: get text features from CoOp
        full_text = model()
        text_features = full_text[categories]
        text_features_norm = text_features / \
            text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # Get image features from CLIP
        with torch.no_grad():
            image_features = model.clip_model.encode_image(images)
            image_features_norm = image_features / \
                image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # Compute logits and loss
        logits = 100.0 * image_features_norm @ text_features_norm.T
        loss = criterion(logits, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        pred_idx = logits.argmax(dim=1)
        pred = categories_tensor[pred_idx]
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc


@torch.no_grad()
def eval(model, dataset, categories, batch_size, device, label=""):
    model.eval()

    full_text = model().detach()
    text_features = full_text[categories]
    text_features /= text_features.norm(dim=-1, keepdim=True)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct_predictions = 0

    categories_tensor = torch.tensor(categories, device=device)

    for image, target in tqdm(dataloader, desc=label):
        image = image.to(device)
        target = target.to(device)
        # Use CLIP for image features
        image_features = model.clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        predicted_idx = (image_features @ text_features.T).argmax(dim=-1)
        predicted_class = categories_tensor[predicted_idx]
        correct_predictions += (predicted_class == target).sum().item()
    accuracy = correct_predictions / len(dataset)
    return accuracy


def train_coop(model, train_loader, val_loader, optimizer, criterion, epochs, device, categories, scheduler=None):
    """Train the CoOp model for multiple epochs.
    """
    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            categories=categories
        )

        # Evaluate on validation set
        val_acc = eval(
            model=model,
            dataset=val_loader.dataset,
            categories=categories,
            batch_size=train_loader.batch_size,
            device=device,
            label=f"Validation Epoch {epoch+1}/{epochs}"
        )

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # Early stopping with model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu()
                                for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered. Best validation accuracy: {best_val_acc:.4f}")
                model.load_state_dict(best_model_state)
                break

    return model
