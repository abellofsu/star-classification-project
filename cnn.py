"""
CNN on ASAS-SN light curves: does a model trained on one band (V or g) still work on the other?

Compares three models on the SAME inputs, splits, and metrics:
  MLP       feed-forward baseline (128 -> 64 -> 32 hidden units)
  CNN       Aiden's design: conv -> max-pool -> conv -> global average pool + dense -> softmax
  CNN+FGSM  the same CNN trained on clean + FGSM adversarial curves (Goodfellow et al., 2015):
            x_adv = x + eps * sign(grad_x L(theta, x, y))

Input per star and band: the .dat light curve phase-folded on its catalog period and
median-binned into 128 phase bins (delta magnitude), plus log10(period) and a has-period flag.
Folding follows recent 1D-CNN work on ASAS-SN g-band curves (arXiv:2508.11964); the period is
added because some classes look alike once folded (arXiv:2512.13395).

Run:   python cnn.py
Time:  about 5 minutes on one GPU, about 20 minutes on a laptop CPU (progress and ETA are printed).
Needs: torch, numpy, pandas, scikit-learn, matplotlib, plus data/ from sample_asassn.py and
       data/{train,val,test}.csv from data-split.py.
Writes results/*.csv, pictures/cnn_*.png, and models/cnn_*.pt.
"""
import copy
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without a display (works on the GPU cluster)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA, RESULTS, PICTURES, MODELS = ROOT / "data", ROOT / "results", ROOT / "pictures", ROOT / "models"
BANDS = ("v", "g")
MODEL_NAMES = ("MLP", "CNN", "CNN+FGSM")
BINS = 128                 # phase bins per folded light curve (same length as mlp.py)
MIN_POINTS = 20            # fewer good points than this = no usable light curve
SEEDS = (0, 1, 2, 3, 4)    # every result is a mean and spread over 5 seeds
EPS_TRAIN = 0.05           # FGSM step used in adversarial training, in magnitudes
NOISE_LEVELS = (0, 0.01, 0.03, 0.10, 0.20)   # Gaussian noise std (mag), levels from our proposal
ATTACK_LEVELS = (0, 0.01, 0.03, 0.05, 0.10)  # FGSM attack size at test time (mag)
BATCH, MAX_EPOCHS, PATIENCE = 128, 500, 40  # the CNN was still improving at epoch 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
START = time.time()


def log(message):
    print(f"[{time.time() - START:5.0f}s] {message}", flush=True)


# ---------------------------------------------------------------- light curves

def read_curve(path, band):
    """Return (time, mag) of the good points in one .dat file.

    Drops what mlp.py kept or could not read: rows with no magnitude, upper limits such
    as '>17.4' (become NaN), and points whose error is 99.99 (bad measurements).
    Some V-band files are only a header (no observations); they come back empty.
    """
    try:
        df = pd.read_csv(path, sep=r"\s+", skiprows=1 if band == "v" else 0)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return np.array([]), np.array([])
    mag_col, err_col = ("MAG", "MAG_ERR") if band == "v" else ("mag", "mag_err")
    if df.empty or mag_col not in df:
        return np.array([]), np.array([])
    t = pd.to_numeric(df["HJD"], errors="coerce").to_numpy()
    m = pd.to_numeric(df[mag_col], errors="coerce").to_numpy()
    e = pd.to_numeric(df[err_col], errors="coerce").to_numpy()
    good = np.isfinite(t) & np.isfinite(m) & (m > 0) & (m < 30) & np.isfinite(e) & (e < 1)
    return t[good], m[good]


def fold(t, m, period):
    """Phase-fold on the period and median-bin into BINS bins of delta magnitude.

    Stars without a period (irregular 'L' variables) are binned over their whole time span
    instead. Empty bins are filled by circular interpolation, periodic curves are rolled so
    the brightest bin comes first, and the star's median magnitude is subtracted.
    """
    has_period = bool(np.isfinite(period) and period > 0)
    p = period if has_period else (t.max() - t.min()) * 1.000001 + 1e-6
    phase = ((t - t.min()) / p) % 1.0
    idx = np.minimum((phase * BINS).astype(int), BINS - 1)
    order = np.argsort(idx, kind="stable")
    cuts = np.flatnonzero(np.diff(idx[order])) + 1
    curve = np.full(BINS, np.nan)
    curve[idx[order][np.r_[0, cuts]]] = [np.median(g) for g in np.split(m[order], cuts)]
    filled = np.isfinite(curve)
    x = np.arange(BINS)
    curve = np.interp(x, x[filled], curve[filled], period=BINS)
    if has_period:
        curve = np.roll(curve, -int(np.argmin(curve)))  # brightest = smallest magnitude
    return (curve - np.median(m)).astype("float32"), np.log10(p), float(has_period)


def catalog_periods(rows, band):
    """Catalog period of each star in one band (same name mapping as mlp.py)."""
    names = rows[f"{band}band_files"].str.rsplit("/", n=1).str[-1].str.removesuffix(".dat")
    if band == "v":
        cat = pd.read_csv(DATA / "vband_catalog.csv", usecols=["asassn_name", "period"]).set_index("asassn_name")
        return cat.loc[names.str.replace("ASASSN-VJ", "ASASSN-V J", regex=False), "period"].to_numpy(float)
    cat = pd.read_csv(DATA / "gband_catalog.csv", usecols=["ID", "Period"]).set_index("ID")
    return cat.loc[names.str.replace("ASASSN-V_", "ASASSN-V ", regex=False), "Period"].to_numpy(float)


def load_split(name, classes):
    """Fold every star of one split in both bands.

    A star is kept only when BOTH bands have a usable light curve, so every model and every
    train/test band pair sees the same stars. Dropped stars are returned with the reason.
    """
    rows = pd.read_csv(DATA / f"{name}.csv")
    periods = {band: catalog_periods(rows, band) for band in BANDS}
    curves = {band: [] for band in BANDS}
    extras = {band: [] for band in BANDS}
    keep, dropped = [], []
    for i, star in enumerate(rows.itertuples()):
        if star.v_class not in classes:
            dropped.append((name, star.star, star.v_class, "class too rare to test (<7 stars)"))
            continue
        folded = {}
        for band in BANDS:
            t, m = read_curve(DATA / getattr(star, f"{band}band_files"), band)
            if len(t) < MIN_POINTS:
                break
            folded[band] = fold(t, m, periods[band][i])
        if len(folded) < 2:
            dropped.append((name, star.star, star.v_class, f"{band}-band light curve has <{MIN_POINTS} good points"))
            continue
        for band in BANDS:
            curves[band].append(folded[band][0])
            extras[band].append(folded[band][1:])
        keep.append(i)
    split = {"rows": rows.iloc[keep].reset_index(drop=True),
             "y": np.array([classes.index(c) for c in rows.v_class.iloc[keep]])}
    for band in BANDS:
        split[band] = (np.stack(curves[band]), np.array(extras[band], dtype="float32"))
    return split, dropped


# ---------------------------------------------------------------- models

class MLP(nn.Module):
    """Feed-forward baseline: the 128 curve bins and the 2 period inputs go in side by side."""

    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BINS + 2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes))

    def forward(self, curve, extra):
        return self.net(torch.cat([curve, extra], dim=1))


class CNN(nn.Module):
    """Aiden's architecture. A 1D convolution slides filters across the curve to find local
    patterns (dips, rises, flat parts); max pooling halves the length; a second convolution
    finds patterns of patterns; global average pooling summarizes them; a dense layer adds
    the period and combines everything; the output layer gives one score per class (softmax
    turns the scores into probabilities). Circular padding because a folded curve wraps around.
    """

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4, padding_mode="circular"), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4, padding_mode="circular"), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(64 + 2, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes))

    def forward(self, curve, extra):
        return self.head(torch.cat([self.features(curve.unsqueeze(1)), extra], dim=1))


def fgsm(model, curve, extra, y, eps, loss_fn):
    """Fast gradient sign method: push every bin eps magnitudes in the direction that raises the loss."""
    curve = curve.clone().requires_grad_(True)
    grad, = torch.autograd.grad(loss_fn(model(curve, extra), y), curve)
    return (curve + eps * grad.sign()).detach()


def to_tensors(split, band, extra_mean, extra_std):
    curve, extra = split[band]
    extra = (extra - extra_mean) / extra_std  # standardize with training statistics only
    return (torch.tensor(curve, device=DEVICE), torch.tensor(extra, dtype=torch.float32, device=DEVICE),
            torch.tensor(split["y"], device=DEVICE))


def macro_f1(y_true, y_pred):
    """F1 averaged equally over the classes present, so rare types count as much as common ones."""
    return f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)


def predict(model, curve, extra):
    model.eval()
    with torch.no_grad():
        return model(curve, extra).argmax(dim=1).cpu().numpy()


def train(name, band, seed, train_split, val_split, n_classes):
    """Train on one band; keep the epoch with the best validation macro-F1 in that SAME band
    (the other band is never looked at during training or model selection)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    extra_mean = train_split[band][1].mean(axis=0)
    extra_std = train_split[band][1].std(axis=0) + 1e-6
    x, z, y = to_tensors(train_split, band, extra_mean, extra_std)
    xv, zv, yv = to_tensors(val_split, band, extra_mean, extra_std)
    model = (MLP if name == "MLP" else CNN)(n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Class-balanced loss: rare types weigh more, so EW and SR do not dominate training.
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=train_split["y"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    best_f1, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(y), device=DEVICE)
        for start in range(0, len(y), BATCH):
            b = order[start:start + BATCH]
            loss = loss_fn(model(x[b], z[b]), y[b])
            if name == "CNN+FGSM":  # half clean loss, half loss on adversarial versions of the batch
                x_adv = fgsm(model, x[b], z[b], y[b], EPS_TRAIN, loss_fn)
                loss = 0.5 * loss + 0.5 * loss_fn(model(x_adv, z[b]), y[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f1 = macro_f1(yv.cpu().numpy(), predict(model, xv, zv))
        if f1 > best_f1:
            best_f1, best_state, best_epoch = f1, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, (extra_mean, extra_std), best_f1, best_epoch


def evaluate(model, stats, test_split, seed):
    """Score on both bands: clean, with Gaussian noise added to the curve, and under FGSM attack."""
    rows, predictions = [], {}
    plain_loss = nn.CrossEntropyLoss()
    generator = torch.Generator(device=DEVICE).manual_seed(1000 + seed)
    for band in BANDS:
        x, z, y = to_tensors(test_split, band, *stats)
        y_np = y.cpu().numpy()
        for sigma in NOISE_LEVELS:
            noisy = x + sigma * torch.randn(x.shape, generator=generator, device=DEVICE)
            p = predict(model, noisy, z)
            if sigma == 0:
                predictions[band] = p
            rows.append(("noise", sigma, band, (p == y_np).mean(), macro_f1(y_np, p)))
        model.eval()
        for eps in ATTACK_LEVELS[1:]:
            p = predict(model, fgsm(model, x, z, y, eps, plain_loss), z)
            rows.append(("fgsm", eps, band, (p == y_np).mean(), macro_f1(y_np, p)))
    return rows, predictions


# ---------------------------------------------------------------- figures

def plot_examples(split, classes):
    """One star per class: raw light curve over time (left) and the folded curve the models see (right)."""
    fig, axes = plt.subplots(len(classes), 2, figsize=(12, 2.0 * len(classes)))
    colors = {"v": "tab:green", "g": "tab:blue"}
    for row, label in enumerate(classes):
        i = int(np.flatnonzero(split["y"] == row)[0])
        star = split["rows"].iloc[i]
        for band in BANDS:
            t, m = read_curve(DATA / star[f"{band}band_files"], band)
            axes[row, 0].scatter(t - 2450000, m, s=2, color=colors[band], label=f"{band.upper()} band")
            axes[row, 1].plot(np.arange(BINS) / BINS, split[band][0][i], color=colors[band], label=f"{band.upper()} band")
        period = 10 ** split["v"][1][i][0] if split["v"][1][i][1] else None
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")
        axes[row, 0].invert_yaxis()
        axes[row, 1].invert_yaxis()
        axes[row, 1].text(0.99, 0.04, f"folded, P = {period:.3g} d" if period else "no period: binned over time",
                          transform=axes[row, 1].transAxes, ha="right", va="bottom", fontsize=8,
                          bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    axes[0, 0].set_title("Raw light curve (magnitude vs HJD - 2450000)")
    axes[0, 1].set_title("Model input: folded, binned, median-subtracted")
    axes[0, 0].legend(markerscale=4, fontsize=8)
    axes[-1, 0].set_xlabel("HJD - 2450000 (days)")
    axes[-1, 1].set_xlabel("phase")
    fig.tight_layout()
    fig.savefig(PICTURES / "cnn_lightcurve_examples.png", dpi=150)
    plt.close(fig)


def plot_class_means(split, classes):
    """Average folded curve per class in each band: what separates the types, and how V and g differ."""
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(classes) / cols)), cols, figsize=(14, 2.6 * np.ceil(len(classes) / cols)))
    for k, (ax, label) in enumerate(zip(axes.flat, classes)):
        for band, color in (("v", "tab:green"), ("g", "tab:blue")):
            curves = split[band][0][split["y"] == k]
            ax.plot(np.arange(BINS) / BINS, np.median(curves, axis=0), color=color, label=f"{band.upper()} band")
        ax.set_title(f"{label} (n={len(curves)})")
        ax.invert_yaxis()
    for ax in axes.flat[len(classes):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Median folded light curve per class (training stars)")
    fig.tight_layout()
    fig.savefig(PICTURES / "cnn_class_median_curves.png", dpi=150)
    plt.close(fig)


def plot_results(runs, confusions, classes):
    clean = runs[(runs.kind == "noise") & (runs.level == 0)]
    stats = clean.groupby(["train", "test", "model"]).f1.agg(["mean", "std"]).reset_index()
    pairs = [("v", "v"), ("v", "g"), ("g", "g"), ("g", "v")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.27
    for k, name in enumerate(MODEL_NAMES):
        s = stats[stats.model == name].set_index(["train", "test"]).loc[pairs]
        ax.bar(np.arange(4) + (k - 1) * width, s["mean"], width, yerr=s["std"], capsize=3, label=name)
    ax.set_xticks(np.arange(4), [f"train {a.upper()} -> test {b.upper()}" for a, b in pairs])
    ax.set_ylabel("test macro-F1 (mean ± std, 5 seeds)")
    ax.set_title("Same band vs cross band")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PICTURES / "cnn_cross_band_f1.png", dpi=150)
    plt.close(fig)

    for kind, levels, xlabel in (("noise", NOISE_LEVELS, "Gaussian noise std added to each bin (mag)"),
                                 ("fgsm", ATTACK_LEVELS, "FGSM attack size eps (mag)")):
        part = runs[runs.kind == kind]
        if kind == "fgsm":  # eps = 0 is the clean score
            part = pd.concat([part, clean.assign(kind="fgsm")])
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
        for ax, (a, b) in zip(axes, pairs):
            for name in MODEL_NAMES:
                s = part[(part.model == name) & (part.train == a) & (part.test == b)].groupby("level").f1.agg(["mean", "std"])
                ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=name)
            ax.set_title(f"train {a.upper()} -> test {b.upper()}")
            ax.set_xlabel(xlabel, fontsize=8)
        axes[0].set_ylabel("test macro-F1")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PICTURES / f"cnn_{kind}_robustness.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for row, (a, b) in enumerate((("v", "g"), ("g", "v"))):
        for col, name in enumerate(MODEL_NAMES):
            cm = confusions[(name, a, b)].astype(float)
            cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            ax = axes[row, col]
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
            ax.set_yticks(range(len(classes)), classes, fontsize=8)
            ax.set_title(f"{name}: train {a.upper()} -> test {b.upper()}")
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
    fig.suptitle("Cross-band confusion matrices (rows sum to 1, all 5 seeds pooled)")
    fig.tight_layout()
    fig.savefig(PICTURES / "cnn_confusion_cross_band.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    for folder in (RESULTS, PICTURES, MODELS):
        folder.mkdir(exist_ok=True)
    log(f"Device: {DEVICE}. Estimated time: ~5 min on a GPU, ~20 min on a CPU.")

    # Classes that have stars in validation and test (data-split.py puts types with <7 stars in train only).
    classes = sorted(set(pd.read_csv(DATA / "val.csv").v_class) & set(pd.read_csv(DATA / "test.csv").v_class))
    log(f"Reading and folding light curves for {len(classes)} classes: {', '.join(classes)}")
    splits, dropped = {}, []
    for name in ("train", "val", "test"):
        splits[name], lost = load_split(name, classes)
        dropped += lost
        log(f"  {name}: {len(splits[name]['y'])} stars kept, {len(lost)} dropped")
    pd.DataFrame(dropped, columns=["split", "star", "v_class", "reason"]).to_csv(RESULTS / "cnn_dropped_stars.csv", index=False)
    log("Light curves ready (dropped stars and reasons: results/cnn_dropped_stars.csv)")

    plot_examples(splits["train"], classes)
    plot_class_means(splits["train"], classes)
    log("Light-curve figures saved to pictures/")

    runs, epochs, confusions = [], [], {}
    total = len(MODEL_NAMES) * len(BANDS) * len(SEEDS)
    done, train_start = 0, time.time()
    for name in MODEL_NAMES:
        for band in BANDS:
            for seed in SEEDS:
                model, stats, val_f1, epoch = train(name, band, seed, splits["train"], splits["val"], len(classes))
                rows, predictions = evaluate(model, stats, splits["test"], seed)
                runs += [(name, band, test_band, seed, kind, level, acc, f1)
                         for kind, level, test_band, acc, f1 in rows]
                epochs.append((name, band, seed, epoch, val_f1))
                for test_band in BANDS:
                    cm = confusion_matrix(splits["test"]["y"], predictions[test_band], labels=range(len(classes)))
                    confusions[(name, band, test_band)] = confusions.get((name, band, test_band), 0) + cm
                if seed == 0:
                    torch.save({"weights": model.state_dict(), "classes": classes, "extra_mean": stats[0],
                                "extra_std": stats[1], "bins": BINS},
                               MODELS / f"cnn_{name.lower().replace('+', '_')}_{band}.pt")
                done += 1
                eta = (time.time() - train_start) / done * (total - done)
                clean = {r[2]: r[4] for r in rows if r[0] == "noise" and r[1] == 0}
                log(f"Training completed {done}/{total}: {name} on {band.upper()}, seed {seed}, best epoch {epoch} | "
                    f"test F1 {band.upper()}->V {clean['v']:.3f}, {band.upper()}->G {clean['g']:.3f} | ETA {eta / 60:.1f} min")

    runs = pd.DataFrame(runs, columns=["model", "train", "test", "seed", "kind", "level", "accuracy", "f1"])
    runs.to_csv(RESULTS / "cnn_all_runs.csv", index=False)
    pd.DataFrame(epochs, columns=["model", "train", "seed", "best_epoch", "val_f1"]).to_csv(RESULTS / "cnn_training.csv", index=False)

    clean = runs[(runs.kind == "noise") & (runs.level == 0)]
    summary = clean.groupby(["model", "train", "test"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("f1", "mean"), macro_f1_std=("f1", "std")).round(3).reset_index()
    summary.to_csv(RESULTS / "cnn_summary.csv", index=False)

    per_class = []
    for (name, a, b), cm in confusions.items():
        precision = np.diag(cm) / cm.sum(axis=0).clip(min=1)
        recall = np.diag(cm) / cm.sum(axis=1).clip(min=1)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
        per_class += [(name, a, b, c, int(cm[k].sum() / len(SEEDS)), round(recall[k], 3), round(f1[k], 3))
                      for k, c in enumerate(classes)]
    pd.DataFrame(per_class, columns=["model", "train", "test", "class", "test_stars", "recall", "f1"]).to_csv(
        RESULTS / "cnn_per_class.csv", index=False)

    plot_results(runs, confusions, classes)
    log("Result figures saved to pictures/cnn_*.png")
    print("\nClean test results (mean over 5 seeds):")
    print(summary.to_string(index=False))
    log("Done")


if __name__ == "__main__":
    main()
