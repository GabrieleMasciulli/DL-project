import torch

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def base_novel_categories(dataset) -> tuple[list[int], list[int]]:
    # set returns the unique set of all dataset classes
    all_classes = set(dataset._labels)
    # and let's count them
    num_classes = len(all_classes)

    # here list(range(num_classes)) returns a list from 0 to num_classes - 1
    # then we slice the list in half and generate base and novel category lists
    base_classes = list(range(num_classes))[:num_classes // 2]
    novel_classes = list(range(num_classes))[num_classes // 2:]
    return base_classes, novel_classes


def harmonic_mean(base_accuracy, novel_accuracy):
    if base_accuracy + novel_accuracy == 0:
        return 0.0
    if base_accuracy == 0 or novel_accuracy == 0:
        return 0.0
    return 2 * base_accuracy * novel_accuracy / (base_accuracy + novel_accuracy)
