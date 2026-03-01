import copy
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix


def build_cnn(
    num_classes: int,
    block_channels: Sequence[int],
    *,
    conv_dropout: float = 0.0,
    linear_dropout: float = 0.3,
    pool_type: str = "max",
    use_global_avg_pool: bool = True,
    layers_per_block: int = 1,
) -> nn.Module:
    """
    Generic helper to construct a sequential CNN with configurable blocks.
    Each block: Conv -> BatchNorm -> ReLU -> Pool (max/avg) -> optional Dropout.
    """

    layers: List[nn.Module] = []
    in_channels = 3
    pool_type = pool_type.lower()

    for out_channels in block_channels:
        for layer_idx in range(layers_per_block):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        if pool_type == "avg":
            layers.append(nn.AvgPool2d(2))
        else:
            layers.append(nn.MaxPool2d(2))
        if conv_dropout > 0:
            layers.append(nn.Dropout2d(conv_dropout))
        in_channels = out_channels

    last_channels = block_channels[-1]
    if use_global_avg_pool:
        layers.append(nn.AdaptiveAvgPool2d(1))
        feature_dim = last_channels
    else:
        spatial = 32 // (2 ** len(block_channels))
        feature_dim = last_channels * spatial * spatial

    layers.append(nn.Flatten())
    layers.append(nn.Linear(feature_dim, 256))
    layers.append(nn.ReLU(inplace=True))
    if linear_dropout > 0:
        layers.append(nn.Dropout(linear_dropout))
    layers.append(nn.Linear(256, num_classes))
    return nn.Sequential(*layers)


def build_m2(
    num_classes: int,
    *,
    conv_dropout: float = 0.1,
    linear_dropout: float = 0.45,
    pool_type: str = "max",
    use_global_avg_pool: bool = True,
) -> nn.Module:
    """
    Build the m2 architecture: two convolutional blocks followed by dense layers.
    """

    return build_cnn(
        num_classes,
        block_channels=[64, 128],
        conv_dropout=conv_dropout,
        linear_dropout=linear_dropout,
        pool_type=pool_type,
        use_global_avg_pool=use_global_avg_pool,
        layers_per_block=2,
    )


def build_m3(
    num_classes: int,
    *,
    conv_dropout: float = 0.15,
    linear_dropout: float = 0.5,
    pool_type: str = "max",
    use_global_avg_pool: bool = True,
) -> nn.Module:
    """
    Build the m3 architecture: three convolutional blocks followed by dense layers.
    """

    return build_cnn(
        num_classes,
        block_channels=[64, 128, 256],
        conv_dropout=conv_dropout,
        linear_dropout=linear_dropout,
        pool_type=pool_type,
        use_global_avg_pool=use_global_avg_pool,
        layers_per_block=2,
    )


class ModelBuilder:
    """
    Utility wrapper that handles training loops, metric logging, and plotting.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        self.model = model
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)
        self.device = device
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history: List[Dict[str, float]] = []
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.best_val_acc = 0.0

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        epochs: int = 10,
        log_every: int = 1,
    ) -> List[Dict[str, float]]:
        """
        Train the wrapped model and store per-epoch metrics.
        """

        self.history = []
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._evaluate(val_loader)

            entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            }
            self.history.append(entry)

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_state = copy.deepcopy(self.model.state_dict())

            if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"[Epoch {epoch:02d}/{epochs}] "
                    f"train_loss={entry['train_loss']:.4f} train_acc={entry['train_acc']*100:.2f}% "
                    f"val_loss={entry['val_loss']:.4f} val_acc={entry['val_acc']*100:.2f}%"
                )

        self.model.load_state_dict(self.best_state)
        return self.history

    def _train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            self.optimizer.step()

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_examples += batch_size

        return {
            "loss": total_loss / max(1, total_examples),
            "accuracy": total_correct / max(1, total_examples),
        }

    def _evaluate(self, loader, *, collect_predictions: bool = False) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        preds: List[np.ndarray] = []
        targets: List[np.ndarray] = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                batch_size = xb.size(0)
                total_loss += loss.item() * batch_size
                pred_batch = logits.argmax(dim=1)
                total_correct += (pred_batch == yb).sum().item()
                total_examples += batch_size
                if collect_predictions:
                    preds.append(pred_batch.cpu().numpy())
                    targets.append(yb.cpu().numpy())

        metrics: Dict[str, float] = {
            "loss": total_loss / max(1, total_examples),
            "accuracy": total_correct / max(1, total_examples),
        }
        if collect_predictions:
            metrics["y_pred"] = np.concatenate(preds)
            metrics["y_true"] = np.concatenate(targets)
        return metrics

    def evaluate_loader(self, loader, *, with_predictions: bool = False) -> Dict[str, float]:
        """
        Evaluate the current model on a loader. Optionally return predictions.
        """

        return self._evaluate(loader, collect_predictions=with_predictions)

    def plot_losses(self, ax=None):
        """
        Plot training vs validation loss curves from stored history.
        """

        if not self.history:
            raise RuntimeError("No training history to plot. Run `fit` first.")

        epochs = [entry["epoch"] for entry in self.history]
        train_losses = [entry["train_loss"] for entry in self.history]
        val_losses = [entry["val_loss"] for entry in self.history]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, train_losses, label="train")
        ax.plot(epochs, val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss curves")
        ax.legend()
        return ax

    def plot_accuracies(self, ax=None):
        """
        Plot training vs validation accuracy curves from stored history.
        """

        if not self.history:
            raise RuntimeError("No training history to plot. Run `fit` first.")

        epochs = [entry["epoch"] for entry in self.history]
        train_acc = [entry["train_acc"] for entry in self.history]
        val_acc = [entry["val_acc"] for entry in self.history]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, train_acc, label="train")
        ax.plot(epochs, val_acc, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy curves")
        ax.legend()
        return ax


def _collect_predictions(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred_batch = logits.argmax(dim=1).cpu().numpy()
            preds.append(pred_batch)
            targets.append(yb.cpu().numpy())

    return np.concatenate(targets), np.concatenate(preds)


def _plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], normalize: bool, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title(title)

    text_kwargs = {"ha": "center", "va": "center"}
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if normalize:
                text = f"{value:.2f}"
            else:
                text = str(int(value))
            color = "white" if value > (cm.max() / 2) else "black"
            ax.text(j, i, text, color=color, **text_kwargs)

    fig.tight_layout()
    return fig


def evaluate_full(
    model: nn.Module,
    data_loader,
    *,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, object]:
    """
    Run inference on a loader and produce raw + normalized confusion matrices.
    """

    y_true, y_pred = _collect_predictions(model, data_loader, device)
    accuracy = float((y_pred == y_true).mean())
    labels = list(range(len(class_names)))
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_raw.astype(float) / row_sums

    fig_raw = _plot_confusion_matrix(cm_raw, class_names, normalize=False, title="Confusion Matrix (raw)")
    fig_norm = _plot_confusion_matrix(cm_norm, class_names, normalize=True, title="Confusion Matrix (normalized)")

    return {
        "accuracy": accuracy,
        "raw_cm": cm_raw,
        "norm_cm": cm_norm,
        "fig_raw": fig_raw,
        "fig_norm": fig_norm,
        "y_true": y_true,
        "y_pred": y_pred,
    }
