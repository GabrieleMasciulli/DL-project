from utils import DEVICE, base_novel_categories, harmonic_mean
from functions import split_data, get_data, eval, train_coop
import clip
from model import CoOp
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    # Inspect classes
    _, _, tmp_test = get_data()
    base_classes, novel_classes = base_novel_categories(tmp_test)

    CLASS_NAMES = ["pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
                   "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia?", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"]
    print("Base Class Names:", [(i, CLASS_NAMES[i]) for i in base_classes])
    print("Novel Class Names:", [(i, CLASS_NAMES[i]) for i in novel_classes])

    # available models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    model, preprocess = clip.load("ViT-B/16", device=DEVICE)

    # A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    preprocess

    # get the three datasets
    train_set, val_set, test_set = get_data(transform=preprocess)

    # split classes into base and novel
    base_classes, novel_classes = base_novel_categories(train_set)

    # split the three datasets
    train_base, _ = split_data(train_set, base_classes)
    val_base, _ = split_data(val_set, base_classes)
    test_base, test_novel = split_data(test_set, base_classes)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_base, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_base, batch_size=32, shuffle=False, num_workers=2)

    # ---- CoOp integration ----
    # Only use base classes for prompt learning
    coop = CoOp(
        clip_model=model,
        classnames=[CLASS_NAMES[i] for i in base_classes],
        n_ctx=16,
        ctx_dim=model.ln_final.weight.shape[0] if hasattr(
            model, "ln_final") else 512,
        device=DEVICE
    ).to(DEVICE)

    # ---- Training setup ----
    # Freeze CLIP model parameters
    for param in coop.clip_model.parameters():
        param.requires_grad = False

    # Only optimize the context vectors
    optimizer = optim.Adam(coop.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("\n🚀 Starting CoOp training...")
    coop = train_coop(
        model=coop,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=10,  # You can adjust this
        device=DEVICE,
        categories=base_classes,
        CLASS_NAMES=CLASS_NAMES,
        clip=clip
    )
    print("✅ Training complete!\n")

    # ---- Compute zero-shot accuracy on base and novel classes ----
    base_accuracy = eval(model=coop, dataset=test_base, categories=base_classes,
                         batch_size=128, device=DEVICE, CLASS_NAMES=CLASS_NAMES, clip=clip, label="🧠 CoOp evaluation on Base Classes")
    novel_accuracy = eval(model=coop, dataset=test_novel, categories=novel_classes,
                          batch_size=128, device=DEVICE, CLASS_NAMES=CLASS_NAMES, clip=clip, label="🧠 CoOp evaluation on Novel Classes")

    print()
    print(f"🔍 Base classes accuracy: {base_accuracy*100:.2f}%")
    print(f"🔍 Novel classes accuracy: {novel_accuracy*100:.2f}%")
    print(
        f"🔍 Harmonic Mean: {harmonic_mean(base_accuracy, novel_accuracy)*100:.2f}%")


if __name__ == '__main__':
    main()
