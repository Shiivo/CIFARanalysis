import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

# All CIFAR-10 data wrangling lives here so notebooks stay tidy.

# Canonical CIFAR-10 class order defined by torchvision.datasets.CIFAR10
CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _to_list(targets) -> List[int]:
    if targets is None:
        return []
    if isinstance(targets, list):
        return targets
    if isinstance(targets, np.ndarray):
        return targets.tolist()
    if torch.is_tensor(targets):
        return targets.detach().cpu().tolist()
    return list(targets)


def load_cifar10_tensors(
    root: str = "./data",
    as_tensors: bool = True,
    download: bool = True,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Download (if needed) and return the CIFAR-10 train/test Dataset objects.

    Args:
        root: Directory where the dataset will be stored/read from.
        as_tensors: If True, applies transforms.ToTensor() to obtain float tensors
            in [0,1]. When False, torchvision will return PIL Images.
        download: Forwarded to torchvision so first run fetches the data.

    Returns:
        (train_dataset, test_dataset)
    """

    transform = transforms.ToTensor() if as_tensors else None
    train = datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    test = datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    return train, test


def split_train_val_test_from_train(
    dataset: Dataset,
    n_train: int,
    n_val: int,
    n_test: int,
    *,
    seed: int = 123,
) -> Tuple[Subset, Subset, Subset]:
    """
    Randomly split the original torchvision CIFAR-10 training dataset into
    train/validation/test subsets with user-provided sizes.

    Any remaining samples (len(dataset) - (n_train + n_val + n_test)) are ignored.
    """

    total_requested = n_train + n_val + n_test
    if total_requested > len(dataset):
        raise ValueError(
            f"Requested {total_requested} samples but dataset only has {len(dataset)}"
        )

    remainder = len(dataset) - total_requested
    lengths = [n_train, n_val, n_test, remainder]
    generator = torch.Generator().manual_seed(seed)
    splits = random_split(dataset, lengths, generator=generator)
    train_subset, val_subset, test_subset = splits[:3]
    return train_subset, val_subset, test_subset


def _tensor_to_image(arr) -> np.ndarray:
    """Convert a tensor or PIL image to an (H, W, C) numpy array for plotting."""

    if torch.is_tensor(arr):
        tensor = arr.detach().cpu()
        if tensor.ndim == 3:
            tensor = tensor.permute(1, 2, 0)
        np_img = tensor.numpy()
    else:
        np_img = np.array(arr)

    # Clip to [0,1] for safe imshow usage irrespective of dtype.
    np_img = np.clip(np_img, 0.0, 1.0)
    return np_img


def one_per_original_class_grid(
    dataset: Dataset,
    *,
    class_names: Sequence[str] = CIFAR10_CLASSES,
    seed: int = 42,
    figsize: Tuple[int, int] = (14, 3),
):
    """
    Display a single-row grid with one example image from each of the original
    CIFAR-10 classes. Useful for the warm-up visualization requirement.
    """

    n_classes = len(class_names)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    picked = {}
    for idx in indices:
        img, label = dataset[idx]
        label_int = int(label) if not torch.is_tensor(label) else int(label.item())
        if label_int not in picked:
            picked[label_int] = img
        if len(picked) == n_classes:
            break

    if len(picked) < n_classes:
        missing = sorted(set(range(n_classes)) - picked.keys())
        raise RuntimeError(f"Could not find examples for class ids: {missing}")

    fig, axes = plt.subplots(1, n_classes, figsize=figsize)
    for class_idx in range(n_classes):
        ax = axes[class_idx]
        ax.imshow(_tensor_to_image(picked[class_idx]))
        ax.set_title(class_names[class_idx])
        ax.axis("off")
    fig.suptitle("One example per CIFAR-10 class", fontsize=14, y=1.05)
    fig.tight_layout()
    return fig


def version_mapping(version: int) -> Dict[int, int]:
    """
    Build a dictionary mapping ORIGINAL CIFAR-10 labels (0-9) to grouped labels,
    based on the specification in the assignment brief.
    """

    name_to_idx = {name: idx for idx, name in enumerate(CIFAR10_CLASSES)}

    # Each branch keeps all 10 original classes assigned exactly once.
    if version == 0:
        grouping = {
            0: ["airplane", "automobile", "ship"],  # ship added so every class is covered
            1: ["truck"],
            2: ["cat", "deer", "dog", "frog", "bird"],
            3: ["horse"],
        }
    elif version == 1:
        grouping = {
            0: ["airplane"],
            1: ["truck", "frog"],
            2: ["deer", "dog", "bird", "cat"],  # cat included to maintain coverage
            3: ["horse"],
            4: ["ship", "automobile"],
        }
    elif version == 2:
        grouping = {
            0: ["automobile", "truck", "airplane"],  # airplane paired with wheeled class
            1: ["frog"],
            2: ["dog"],
            3: ["horse", "bird", "deer"],
            4: ["cat", "ship"],  # ship attached to feline group to keep dataset intact
        }
    else:
        raise ValueError("Version must be 0, 1, or 2.")

    mapping: Dict[int, int] = {}
    for new_label, names in grouping.items():
        for name in names:
            idx = name_to_idx[name]
            mapping[idx] = new_label

    # Safety check: ensure every original class appears exactly once.
    if set(mapping.keys()) != set(range(len(CIFAR10_CLASSES))):
        missing = sorted(set(range(len(CIFAR10_CLASSES))) - mapping.keys())
        raise ValueError(f"Grouping missing original class ids: {missing}")

    return mapping


class RemapTargets(Dataset):
    """
    Torch Dataset wrapper that remaps class targets based on a dictionary.
    """

    def __init__(self, dataset: Dataset, mapping: Mapping[int, int]):
        self.dataset = dataset
        self.mapping = dict(mapping)
        try:
            base_targets = get_targets(dataset)
            self.targets = [self.mapping[int(t)] for t in base_targets]
        except Exception:
            self.targets = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data, target = self.dataset[idx]
        target_int = int(target) if not torch.is_tensor(target) else int(target.item())
        if target_int not in self.mapping:
            raise KeyError(f"Target {target_int} missing from mapping.")
        return data, self.mapping[target_int]


class TransformDataset(Dataset):
    """
    Apply an additional transform to each sample retrieved from an existing Dataset.
    Handy when we want augmentation/normalisation without touching the source dataset.
    """

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        data, target = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, target


def get_targets(dataset: Dataset) -> List[int]:
    """
    Attempt to extract a list of targets from a Dataset (supports Subset/Transform wrappers).
    """

    if isinstance(dataset, RemapTargets) and getattr(dataset, "targets", None) is not None:
        return dataset.targets

    if hasattr(dataset, "targets"):
        return _to_list(dataset.targets)

    if hasattr(dataset, "labels"):
        return _to_list(dataset.labels)

    if isinstance(dataset, TransformDataset):
        return get_targets(dataset.dataset)

    if isinstance(dataset, Subset):
        base_targets = get_targets(dataset.dataset)
        return [base_targets[i] for i in dataset.indices]

    raise AttributeError("Dataset does not expose targets attribute.")


def dataset_class_histogram(dataset: Dataset, num_classes: int) -> np.ndarray:
    """
    Count the number of items per class for a Dataset.
    """

    try:
        targets = get_targets(dataset)
        return np.bincount(np.asarray(targets), minlength=num_classes)
    except Exception:
        counts = np.zeros(num_classes, dtype=np.int64)
        for _, target in dataset:
            counts[int(target)] += 1
        return counts


def loader_class_histogram(loader: DataLoader, num_classes: int) -> np.ndarray:
    """
    Count the number of labels observed while iterating over a DataLoader.
    """

    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, targets in loader:
        counts += torch.bincount(targets.view(-1), minlength=num_classes)
    return counts.cpu().numpy()


def make_weighted_sampler(dataset: Dataset, num_classes: int) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to mitigate class imbalance.
    Weights are inversely proportional to class counts so minority classes show up more often.
    """

    targets = get_targets(dataset)
    counts = np.bincount(np.asarray(targets), minlength=num_classes)
    counts = np.maximum(counts, 1)
    class_weights = 1.0 / counts
    sample_weights = [class_weights[int(t)] for t in targets]
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(sample_weights), replacement=True)


def make_loaders(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    *,
    batch_size: int = 128,
    num_workers: int = 2,
    train_sampler: Optional[torch.utils.data.Sampler] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Helper to produce PyTorch DataLoaders with consistent settings.
    """

    pin_memory = torch.cuda.is_available()
    shuffle_train = train_sampler is None
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
