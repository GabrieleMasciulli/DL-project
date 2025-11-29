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


@torch.no_grad()
def eval(model, dataset, categories, batch_size, device, CLASS_NAMES, clip, label=""):
    model.eval()
    contig_cat2idx = {cat: idx for idx, cat in enumerate(categories)}
    text_inputs = clip.tokenize(
        [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in categories]
    ).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct_predictions = 0
    total_processed = 0

    for image, target in tqdm(dataloader, desc=label):
        target = torch.Tensor([contig_cat2idx[t.item()]
                              for t in target]).long()
        image = image.to(device)
        target = target.to(device)

        image_features = model.encode_image(image)  # (B, feats)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T  # (B, B)
        predicted_class = similarity.argmax(dim=-1)  # (B)
        correct_predictions += (predicted_class == target).sum().item()

        total_processed += len(target)

    accuracy = correct_predictions / len(dataset)
    return accuracy
