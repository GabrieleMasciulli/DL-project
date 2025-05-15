from utils import DEVICE, base_novel_categories, harmonic_mean, BATCH_SIZE_EVAL, BATCH_SIZE_TRAIN
from functions import split_data, get_data, eval, train_cocoop, clip_contrastive_loss
import clip
from model import CoCoOp
import torch
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
    val_base, val_novel = split_data(
        val_set, base_classes)
    test_base, test_novel = split_data(test_set, base_classes)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_base, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_base, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=2)

    # Create DataLoader for novel validation set
    eval_novel_loader = torch.utils.data.DataLoader(
        val_novel, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=2)

    # Create DataLoaders for test sets
    test_base_loader = torch.utils.data.DataLoader(
        test_base, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=2)
    test_novel_loader = torch.utils.data.DataLoader(
        test_novel, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=2)

    # ---- CoCoOp integration ----
    # Determine vis_dim from the CLIP model
    vis_dim = model.visual.output_dim

    cocoop = CoCoOp(
        clip_model=model,
        classnames=[CLASS_NAMES[i]
                    for i in base_classes],  # Training on base classes
        n_ctx=16,
        ctx_dim=model.ln_final.weight.shape[0],
        vis_dim=vis_dim,
        device=DEVICE
    ).to(DEVICE)

    # ---- Training setup ----
    # Freeze CLIP model's visual encoder and text encoder parameters (except for what CoCoOp manages)
    for name, param in cocoop.clip_model.named_parameters():
        param.requires_grad = False

    # Unfreeze CoCoOp's own parameters (ctx and meta_net)
    cocoop.ctx.requires_grad = True
    for param in cocoop.meta_net.parameters():
        param.requires_grad = True

    # Optimizer should only optimize CoCoOp's parameters (ctx and meta_net)
    optimizer = optim.AdamW(
        list(cocoop.meta_net.parameters()) +
        [cocoop.ctx],
        lr=2e-3,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20)
    criterion = clip_contrastive_loss

    # Train the model
    print("\n🚀 Starting CoCoOp training...")

    cocoop = train_cocoop(
        model=cocoop,
        clip_model_visual=model.visual,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=20,
        device=DEVICE,
        categories=base_classes,
        all_class_names=CLASS_NAMES,
        clip_tokenizer=clip.tokenize,
        scheduler=scheduler,
        patience=5,
        eval_novel_loader=eval_novel_loader,
        novel_categories=novel_classes,
    )
    print("✅ Training complete!\n")

    print("Evaluating with CoCoOp...")

    base_accuracy = eval(
        cocoop_model=cocoop,
        clip_model_visual=model.visual,
        data_loader=test_base_loader,
        eval_categories=base_classes,
        all_class_names=CLASS_NAMES,
        device=DEVICE,
        clip_tokenizer=clip.tokenize,
        label="🧠 CoCoOp evaluation on Base Classes"
    )
    novel_accuracy = eval(
        cocoop_model=cocoop,
        clip_model_visual=model.visual,
        data_loader=test_novel_loader,
        eval_categories=novel_classes,
        all_class_names=CLASS_NAMES,
        device=DEVICE,
        clip_tokenizer=clip.tokenize,
        label="🧠 CoCoOp evaluation on Novel Classes"
    )

    print()
    print(f"🔍 Base classes accuracy (CoCoOp): {base_accuracy*100:.2f}%")
    print(f"🔍 Novel classes accuracy (CoCoOp): {novel_accuracy*100:.2f}%")
    print(
        f"🔍 Harmonic Mean (CoCoOp): {harmonic_mean(base_accuracy, novel_accuracy)*100:.2f}%")


if __name__ == '__main__':
    main()
