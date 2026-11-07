# Auto-exported from notebooks/QWafer_Mamba_WM811K_v4.ipynb (cells in order). Shell lines start with "# !".

# # QWafer-Mamba v4 — revision experiments for IJIES (paper 20266781)
# 
# v4 reruns the complete v3 pipeline (same data, lot-aware split, seeds and hyperparameters) and adds every experiment requested by the two reviewers. Nothing from v3 is needed: all checkpoints are recreated.
# 
# | Reviewer comment | What v4 adds | Output table |
# |---|---|---|
# | R1-9, R2-3 | Capacity-matched classical heads on the same backbone and schedule: MLP head, bounded tanh head, trigonometric head | `table_stage2_heads_ablation_mean_std.csv` |
# | R1-10 | Quantum-layer ablations: no entanglement, frozen θ, reps = 0, reps = 1 | same table |
# | R1-2, R2-5 | Finite-shot evaluation (64 to 8192 shots), IBM fake-backend noise models (gate + readout noise), transpiled gate counts, depth, duration and a latency breakdown | `table_shots_*.csv`, `table_noise_*.csv`, `table_transpile_*.csv`, `table_latency_breakdown.csv` |
# | R1-3, R2-2 | Three additional independent lot-disjoint partitions (QWafer-Mamba, classical control, WaferMamba, ResNet-18) with minority-class recall | `table_partitions_*.csv` |
# | R1-11 | Random vs lot-aware split on four matched partitions with identical subset sizes | `table_split_protocol_partitions.csv` |
# | R1-12, R2-6 | Modern lightweight baselines on the same split: EfficientNet-B0 and MobileViT-XXS (5 seeds) | main table |
# | R1-7, R2-4 | Primary hypothesis, Holm correction, TOST equivalence test, hierarchical (seed × wafer) bootstrap | `table_stats_*.csv` |
# | R1-4 | Stored vs active parameters for every model | `table_param_accounting.csv` |
# | R1-5 | Per-seed confusion matrices and per-class metrics for every model | `table_confusion_per_seed.csv`, `table_per_class_per_seed.csv` |
# | R1-8 | Machine-readable artefacts: wafer-level split assignments with lot names, per-wafer test predictions, training histories, configuration, package versions, all checkpoints | `artifact/` folder + checkpoint ZIP |
# 
# **Storage.** Google Drive is now mandatory (`REQUIRE_DRIVE = True`). v3 silently fell back to local storage when the mount failed, and the checkpoints were lost. v4 stops instead. Every finished run is written to `MyDrive/QWafer_Mamba_v2/runs_v4` immediately, so after a disconnect simply re-run all cells: finished runs are loaded, not retrained.
# 
# **Runtime (T4 GPU).** Roughly 10 to 12 h in total, so expect two or three Colab sessions; the notebook resumes where it stopped. Approximate shares: stage 1 with five backbones × five seeds about 4 h, stage 2 with nine heads × five seeds about 3.5 h, extra partitions about 2 h, noise study and statistics about 1 h. Switches in Cell 2 turn each block off if time is short.
# 
# **Runtime → Change runtime type → T4 GPU** before starting.
# 
# **If the Drive mount fails** ("credential propagation was unsuccessful"): run the cell again and, in the pop-up, sign in with the same Google account that owns this Colab session and tick every permission box. If the pop-up is blocked, allow pop-ups and third-party cookies for colab.research.google.com. Alternatively, mount Drive from the folder icon in the left sidebar (Files → Mount Drive) and re-run the cell.

# CELL 1 — INSTALL
# !pip -q install kaggle "scikit-learn>=1.4" scipy qiskit qiskit-aer qiskit-ibm-runtime timm pylatexenc

# CELL 2 — CONFIGURATION
import os, json, math, time, random, zipfile, shutil, gc, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from scipy.stats import binomtest, ttest_1samp
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             matthews_corrcoef, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc)
warnings.filterwarnings("ignore")

# ---------------- user switches ----------------
USE_DRIVE = True          # persist results/checkpoints in Google Drive
REQUIRE_DRIVE = True      # stop if Drive cannot be mounted (prevents losing checkpoints again)
QUICK     = False         # True = 1 seed, 25 epochs (pipeline check only)

SEEDS   = [42] if QUICK else [42, 7, 123, 2024, 31]
EPOCHS  = 25 if QUICK else 60
PATIENCE = 12
WARMUP  = 3
BATCH   = 128
IMG_SIZE = 64
LATENT  = 64
NCLS    = 8
LR_MAIN = {"CNN": 2e-3, "ResNet18": 1e-3, "WaferMamba": 2e-3, "EfficientNetB0": 1e-3, "MobileViTXXS": 1e-3}
WD      = 0.05
LABEL_SMOOTH = 0.05
IMBALANCE = "loss"        # "loss" = class-weighted CE + natural sampling ; "sampler" = weighted sampling + plain CE
MODELS  = ["CNN", "ResNet18", "WaferMamba", "EfficientNetB0", "MobileViTXXS"]   # modern lightweight baselines added (R1-12)
RUN_RANDOM_SPLIT = True
RANDOM_SPLIT_MODELS = ["WaferMamba", "ResNet18"]

QUBITS  = 8
QDEPTH  = 3               # re-uploading layers -> QUBITS*QDEPTH PCA inputs
RES_SEED = 0              # fixed reservoir (part of the method definition)
N_BOOT  = 1000
SPLIT_SEED = 42           # identical lot-aware split to v1

# quantum circuits (Qiskit RealAmplitudes)
HQ_QUBITS, HQ_REPS = 8, 3                 # trainable quantum layer inside QWafer-Mamba (8 qubits = 8 classes)
FT_EPOCHS   = 8 if QUICK else 20          # end-to-end hybrid training epochs (starting from the trained backbone)
FT_PATIENCE = 6
FT_HEAD_WARMUP = 2 if QUICK else 3        # epochs with frozen backbone (new head learns first)
FT_CW_POWER = 0.5                         # class weight = (inverse frequency)^power in stage 2, both arms
LR_BACKBONE, LR_QUANTUM = 2e-4, 1e-2      # learning rates: backbone vs quantum layer + bridge
VQC_QUBITS, VQC_REPS = 8, 2               # COBYLA-trained VQC baseline
VQC_PER_CLASS = 300                       # class-balanced COBYLA training subset
VQC_MAXITER = 150 if QUICK else 400

# ---------------- v4 revision experiments ----------------
RUN_HEAD_CONTROLS     = True    # R1-9 / R2-3: MLP, bounded tanh and trigonometric classical heads
RUN_CIRCUIT_ABLATIONS = True    # R1-10: no entanglement, frozen theta, reps 0, reps 1
HEAD_KINDS = ["MLP_FT", "Tanh_FT", "Trig_FT"]
ABL_KINDS  = ["Q_NoEnt", "Q_Frozen", "Q_Reps0", "Q_Reps1"]

RUN_PARTITIONS  = True          # R1-3 / R2-2 / R1-11: additional independent lot-disjoint partitions
PARTITION_SEEDS = [101] if QUICK else [101, 202, 303]
PARTITION_MODELS = ["WaferMamba", "ResNet18"]      # stage 1 models trained on every extra partition (seed 42)
PARTITION_STAGE2 = ["QWaferMamba_E2E", "WaferMamba_FT"]

RUN_NOISE    = True             # R1-2 / R2-5: finite shots + IBM fake-backend noise models
SHOTS_LIST   = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
SHOT_REPEATS = 5
NOISE_SHOTS  = 1024
NOISE_N      = 300 if QUICK else None   # None = full test set
FAKE_BACKENDS = ["FakeTorino", "FakeSherbrooke"]  # IBM Heron (CZ) and Eagle (ECR) calibration snapshots
REP_DELAY_US = 250.0            # default repetition delay of IBM devices, used for the QPU time estimate

EQUIV_MARGIN = 0.01             # R1-7: equivalence margin = 1 Macro-F1 point
N_HBOOT      = 2000             # hierarchical (seed x wafer) bootstrap replicates
# ------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = Path("/content/QWafer_Mamba_v2")            # local fallback
if USE_DRIVE:
    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        ROOT = Path("/content/drive/MyDrive/QWafer_Mamba_v2")
        print("Google Drive mounted: results will survive a disconnect.")
    except Exception as e:
        if REQUIRE_DRIVE:
            raise RuntimeError("Google Drive could not be mounted (" + str(e) + "). Fix the mount (see the note "
                               "above) and re-run this cell. Set REQUIRE_DRIVE = False only if you accept losing "
                               "checkpoints when the runtime disconnects.")
        print("WARNING: Drive mount failed ->", e, "| continuing with local storage")
RAW   = Path("/content/wm811k_raw")          # 2 GB pickle stays on local disk
CACHE = ROOT/"cache"; OUT = ROOT/"results_v4"; RUNS = ROOT/"runs_v4"
for p in [RAW, CACHE, OUT, OUT/"figures", OUT/"tables", OUT/"artifact", RUNS]:
    p.mkdir(parents=True, exist_ok=True)

DEFECTS = ["Center","Donut","Edge-Loc","Edge-Ring","Loc","Random","Scratch","Near-full"]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

print("DEVICE:", DEVICE, "| AMP:", AMP, "| seeds:", SEEDS, "| epochs:", EPOCHS)
if DEVICE.type == "cuda": print(torch.cuda.get_device_name(0))
(RUNS/"_write_test.txt").write_text("ok")                 # fails loudly if the run folder is not writable
print("Checkpoints and per-run predictions are saved in:", RUNS)

# ## Data
# If the preprocessed cache (`cache/wm811k_8class_64.npz`) already exists in Drive, the Kaggle download and the 2 GB pickle are skipped entirely. Otherwise upload `kaggle.json` when asked.

# CELL 3 — DOWNLOAD (skipped when the cache exists)
CACHE_FILE = CACHE/"wm811k_8class_64.npz"
pkl = RAW/"LSWMD.pkl"

if not CACHE_FILE.exists() and not pkl.exists():
    from google.colab import files
    kj = Path("/root/.kaggle/kaggle.json"); kj.parent.mkdir(parents=True, exist_ok=True)
    if not kj.exists():
        print("Upload kaggle.json")
        up = files.upload()
        if "kaggle.json" not in up: raise FileNotFoundError("kaggle.json required")
        kj.write_bytes(up["kaggle.json"]); os.chmod(kj, 0o600)
    os.system(f'kaggle datasets download -d qingyi/wm811k-wafer-map -p "{RAW}" --force')
    z = list(RAW.glob("*.zip"))
    if not z: raise FileNotFoundError("Kaggle download failed")
    with zipfile.ZipFile(z[0]) as f: f.extractall(RAW)
    z[0].unlink()
    if not pkl.exists():
        found = list(RAW.rglob("LSWMD.pkl"))
        if not found: raise FileNotFoundError("LSWMD.pkl not found")
        pkl = found[0]
print("Cache present:" if CACHE_FILE.exists() else "Raw pickle:", CACHE_FILE if CACHE_FILE.exists() else pkl)

# CELL 4 — LOAD, CLEAN LABELS, PREPROCESS TO 3x64x64 (cached)
def unwrap(x):
    while isinstance(x, (list, tuple, np.ndarray)) and np.asarray(x, dtype=object).size == 1:
        x = np.asarray(x, dtype=object).reshape(-1)[0]
    if isinstance(x, (list, tuple, np.ndarray)) or pd.isna(x): return ""
    return str(x).strip()

def wafer_to_channels(arr):
    t = torch.tensor(np.asarray(arr, dtype=np.float32))[None, None]
    r = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="nearest")[0, 0].numpy()
    # channels: wafer support, passing dies, failing dies
    return np.stack([r > 0, r == 1, r == 2]).astype(np.uint8)

if CACHE_FILE.exists():
    d = np.load(CACHE_FILE, allow_pickle=False)
    X, Y, LOTS = d["X"], d["Y"], d["LOTS"]
else:
    try: df = pd.read_pickle(pkl)
    except Exception:
        import pickle
        with open(pkl, "rb") as f: df = pickle.load(f, encoding="latin1")
    print("Raw shape:", df.shape)
    canon = {"center":"Center","donut":"Donut","edge-loc":"Edge-Loc","edge-ring":"Edge-Ring",
             "loc":"Loc","random":"Random","scratch":"Scratch","near-full":"Near-full","near full":"Near-full"}
    df["label"] = df["failureType"].apply(unwrap).str.lower().map(canon)
    work = df[df["label"].isin(DEFECTS)].reset_index(drop=True)
    del df; gc.collect()
    Y = work["label"].map({c: i for i, c in enumerate(DEFECTS)}).values.astype(np.int64)
    LOTS = work["lotName"].apply(unwrap).values.astype(str)
    X = np.stack([wafer_to_channels(a) for a in work["waferMap"].values])
    del work; gc.collect()
    np.savez_compressed(CACHE_FILE, X=X, Y=Y, LOTS=LOTS)

print("X:", X.shape, X.dtype, "| N =", len(Y))
display(pd.Series(np.bincount(Y, minlength=NCLS), index=DEFECTS, name="count").to_frame())

fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for k, ax in enumerate(axs.flat):
    i = np.where(Y == k)[0][0]
    ax.imshow(X[i, 1] + 2 * X[i, 2], cmap="viridis"); ax.set_title(DEFECTS[k]); ax.axis("off")
plt.tight_layout(); fig.savefig(OUT/"figures/fig_samples.png", dpi=300, bbox_inches="tight"); plt.show()

# CELL 5 — SPLITS: LOT-AWARE (primary + extra partitions) and SIZE-MATCHED STRATIFIED RANDOM (leakage reference)
idx_all = np.arange(len(Y))

def lot_split(p):
    gss = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=p)
    trv, te = next(gss.split(idx_all, Y, groups=LOTS))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=p + 1)
    a, b = next(gss2.split(trv, Y[trv], groups=LOTS[trv]))
    tr, va = trv[a], trv[b]
    assert not set(LOTS[tr]) & set(LOTS[te]) and not set(LOTS[va]) & set(LOTS[te]) and not set(LOTS[tr]) & set(LOTS[va])
    return tr, va, te

def random_split_matched(p, sizes):
    # stratified random split with exactly the same subset sizes as the lot-aware partition p (R1-11)
    n_tr, n_va, n_te = sizes
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_te, random_state=p)
    trv, te = next(sss.split(idx_all, Y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=n_va, random_state=p + 1)
    a, b = next(sss2.split(trv, Y[trv]))
    return trv[a], trv[b], te

SPLITS = {"lot": lot_split(SPLIT_SEED)}                        # primary partition, identical to v1-v3
SPLITS["random"] = random_split_matched(SPLIT_SEED, [len(s) for s in SPLITS["lot"]])
PARTITIONS = [("lot", "random", SPLIT_SEED)]
if RUN_PARTITIONS:
    for p in PARTITION_SEEDS:
        SPLITS[f"lotP{p}"] = lot_split(p)
        SPLITS[f"randP{p}"] = random_split_matched(p, [len(s) for s in SPLITS[f"lotP{p}"]])
        PARTITIONS.append((f"lotP{p}", f"randP{p}", p))
print("Lot leakage audit: PASSED for", [k for k in SPLITS if k.startswith("lot")])
for a, b, p in PARTITIONS[1:]:
    ov = len(set(LOTS[SPLITS[a][2]]) & set(LOTS[SPLITS["lot"][2]])) / len(set(LOTS[SPLITS[a][2]]))
    print(f"partition {p}: share of its test lots that are also primary test lots = {ov:.2%}")

rows = []
for sp, (tr_, va_, te_) in SPLITS.items():
    for nm, ii in [("train", tr_), ("validation", va_), ("test", te_)]:
        rows.append({"protocol": sp, "split": nm, "n": len(ii), "lots": len(set(LOTS[ii])),
                     **dict(zip(DEFECTS, np.bincount(Y[ii], minlength=NCLS)))})
split_table = pd.DataFrame(rows)
display(split_table)
split_table.to_csv(OUT/"tables/table_split_distribution.csv", index=False)

# wafer-level split assignment with lot identifiers (R1-8)
asg = []
for sp, (tr_, va_, te_) in SPLITS.items():
    lab = np.full(len(Y), "", dtype=object)
    lab[tr_], lab[va_], lab[te_] = "train", "validation", "test"
    asg.append(pd.DataFrame({"protocol": sp, "wafer_index": idx_all, "lot": LOTS, "label": np.array(DEFECTS)[Y], "subset": lab}))
pd.concat(asg).to_csv(OUT/"artifact/split_assignments.csv.gz", index=False)

XG = torch.from_numpy(X).to(DEVICE)          # uint8 on GPU (~0.3 GB)
YG = torch.from_numpy(Y).to(DEVICE)

# ## Models
# * **WaferMamba (SS2D):** convolutional stem → 8×8 token grid → 2 VSS blocks. Each block runs a selective state-space scan (S6: input-dependent Δ, B, C; diagonal A; skip D) over the token grid in four orders (row-major, column-major and both reversed), then merges them. MLP sub-layer and residuals as in VMamba.
# * **CNN baseline:** same stem, the two VSS blocks replaced by two residual 3×3 convolution blocks (isolates the contribution of state-space mixing).
# * **ResNet-18:** standard reference backbone, 3×3 first convolution for 64×64 inputs.
# 
# * **EfficientNet-B0** (torchvision) and **MobileViT-XXS** (timm): modern lightweight convolutional and attention baselines (R1-12), trained from scratch with a stride-1 first convolution for 64×64 inputs.
# 
# All backbones share the same 64-dim latent projection and linear classifier.

# CELL 6 — MODEL DEFINITIONS
class ConvStem(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        def cbg(i, o, s): return [nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.GELU()]
        self.net = nn.Sequential(*cbg(3, 32, 1), *cbg(32, 32, 2), *cbg(32, 64, 2), *cbg(64, d, 2))  # 64 -> 8
    def forward(self, x): return self.net(x)

class LatentHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, LATENT), nn.GELU(), nn.Dropout(0.1))
        self.cls = nn.Linear(LATENT, NCLS)
    def forward(self, f, return_latent=False):
        h = self.proj(f); z = self.cls(h)
        return (z, h) if return_latent else z

class SS2D(nn.Module):
    # Selective state-space (S6) scan over a 2-D token grid in four directions.
    def __init__(self, d_model, d_state=8, expand=2, dt_min=1e-3, dt_max=0.1):
        super().__init__()
        d_in = expand * d_model
        self.d_in, self.N, self.R = d_in, d_state, math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_in, bias=False)
        self.dwconv = nn.Conv2d(d_in, d_in, 3, padding=1, groups=d_in)
        self.x_proj = nn.Linear(d_in, self.R + 2 * d_state, bias=False)   # -> dt, B, C (input dependent)
        self.dt_proj = nn.Linear(self.R, d_in)
        dt = torch.exp(torch.rand(d_in) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad(): self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_in, 1)))
        self.D = nn.Parameter(torch.ones(d_in))
        self.out_norm = nn.LayerNorm(d_in)
        self.out_proj = nn.Linear(d_in, d_model, bias=False)

    def scan(self, u):                                   # u: (B, L, d_in)
        with torch.autocast(device_type="cuda", enabled=False):
            u = u.float()
            dt, Bm, Cm = torch.split(self.x_proj(u), [self.R, self.N, self.N], -1)
            dt = F.softplus(self.dt_proj(dt))                       # (B,L,d)
            A = -torch.exp(self.A_log.float())                      # (d,N)
            dA = torch.exp(dt.unsqueeze(-1) * A)                    # (B,L,d,N)
            dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)
            h = u.new_zeros(u.size(0), self.d_in, self.N); ys = []
            dA_t, dBu_t, C_t = dA.unbind(1), dBu.unbind(1), Cm.unbind(1)   # unbind: fast backward pass
            for t in range(u.size(1)):                              # recurrent selective scan
                h = dA_t[t] * h + dBu_t[t]
                ys.append((h * C_t[t].unsqueeze(1)).sum(-1))
            return torch.stack(ys, 1) + u * self.D

    def forward(self, x):                                # x: (B, H, W, C)
        B, H, W, _ = x.shape; L = H * W
        xi, z = self.in_proj(x).chunk(2, -1)
        xi = F.silu(self.dwconv(xi.permute(0, 3, 1, 2)))            # (B,d,H,W)
        row = xi.flatten(2).transpose(1, 2)                         # row-major order
        col = xi.transpose(2, 3).flatten(2).transpose(1, 2)         # column-major order
        y = self.scan(torch.cat([row, col, row.flip(1), col.flip(1)], 0))
        y_r, y_c, y_rr, y_cr = y.chunk(4, 0)
        col2row = lambda t: t.reshape(B, W, H, -1).transpose(1, 2).reshape(B, L, -1)
        y = y_r + y_rr.flip(1) + col2row(y_c) + col2row(y_cr.flip(1))
        y = self.out_norm(y) * F.silu(z.reshape(B, L, -1))
        return self.out_proj(y).reshape(B, H, W, -1)

class VSSBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n1, self.ssm = nn.LayerNorm(d), SS2D(d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
    def forward(self, x):
        x = x + self.ssm(self.n1(x))
        return x + self.mlp(self.n2(x))

class WaferMamba(nn.Module):
    def __init__(self, d=64, depth=2):
        super().__init__()
        self.stem = ConvStem(d)
        self.pos = nn.Parameter(torch.zeros(1, 8, 8, d)); nn.init.trunc_normal_(self.pos, std=.02)
        self.blocks = nn.Sequential(*[VSSBlock(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d); self.head = LatentHead(d)
    def forward(self, x, return_latent=False):
        t = self.stem(x).permute(0, 2, 3, 1) + self.pos
        t = self.norm(self.blocks(t)).mean((1, 2))
        return self.head(t, return_latent)

class ResConv(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(d, d, 3, 1, 1, bias=False), nn.BatchNorm2d(d), nn.GELU(),
                               nn.Conv2d(d, d, 3, 1, 1, bias=False), nn.BatchNorm2d(d))
    def forward(self, x): return F.gelu(x + self.f(x))

class CNNNet(nn.Module):
    def __init__(self, d=64, depth=2):
        super().__init__()
        self.stem = ConvStem(d); self.blocks = nn.Sequential(*[ResConv(d) for _ in range(depth)])
        self.head = LatentHead(d)
    def forward(self, x, return_latent=False):
        return self.head(self.blocks(self.stem(x)).mean((2, 3)), return_latent)

class ResNetNet(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); m.fc = nn.Identity()
        self.body = m; self.head = LatentHead(512)
    def forward(self, x, return_latent=False):
        return self.head(self.body(x), return_latent)

class EffNetB0Net(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.efficientnet_b0(weights=None)
        m.features[0][0] = nn.Conv2d(3, 32, 3, 1, 1, bias=False)      # stride 1 for 64x64 wafer maps
        m.classifier = nn.Identity()
        self.body = m; self.head = LatentHead(1280)
    def forward(self, x, return_latent=False):
        return self.head(self.body(x), return_latent)

try:
    import timm
    class MobileViTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = timm.create_model("mobilevit_xxs", pretrained=False, num_classes=0)
            self.body.stem.conv.stride = (1, 1)                         # stride 1 for 64x64 wafer maps
            self.head = LatentHead(self.body.num_features)
        def forward(self, x, return_latent=False):
            return self.head(self.body(x), return_latent)
except Exception as e:
    MobileViTNet = None; print("timm unavailable -> MobileViT-XXS skipped:", e)

BUILD = {"CNN": CNNNet, "ResNet18": ResNetNet, "WaferMamba": WaferMamba, "EfficientNetB0": EffNetB0Net}
if MobileViTNet is not None: BUILD["MobileViTXXS"] = MobileViTNet
MODELS = [m for m in MODELS if m in BUILD]
nparams = lambda m: sum(p.numel() for p in m.parameters())

# sanity check: direction merge in SS2D must be an exact inverse permutation
_s = SS2D(16); _x = torch.randn(2, 8, 8, 16)
_s.scan = lambda u: u                                    # identity scan
xi = F.silu(_s.dwconv(_s.in_proj(_x).chunk(2, -1)[0].permute(0, 3, 1, 2))).flatten(2).transpose(1, 2)
z = _s.in_proj(_x).chunk(2, -1)[1].reshape(2, 64, -1)
ref = _s.out_proj(_s.out_norm(4 * xi) * F.silu(z)).reshape(2, 8, 8, -1)
assert torch.allclose(_s(_x), ref, atol=1e-5); del _s
print("SS2D cross-scan merge check: PASSED")

for n, cls in BUILD.items():
    m = cls(); out = m(torch.zeros(2, 3, IMG_SIZE, IMG_SIZE), return_latent=True)
    print(f"{n:11s} params={nparams(m):>10,d}  logits={tuple(out[0].shape)}  latent={tuple(out[1].shape)}")

# CELL 7 — TRAINING / INFERENCE UTILITIES
def gpu_augment(x):                                   # label-preserving: flips + 90° rotations
    B = x.size(0)
    m = torch.rand(B, device=x.device) < .5; x[m] = x[m].flip(-1)
    m = torch.rand(B, device=x.device) < .5; x[m] = x[m].flip(-2)
    k = torch.randint(0, 4, (B,), device=x.device)
    for r in (1, 2, 3):
        m = k == r
        if m.any(): x[m] = torch.rot90(x[m], r, (-2, -1))
    return x

@torch.no_grad()
def predict(model, idx, bs=512):
    model.eval(); P, H = [], []
    for s in range(0, len(idx), bs):
        bi = torch.as_tensor(idx[s:s + bs], device=DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            z, h = model(XG[bi].float(), return_latent=True)
        P.append(torch.softmax(z.float(), 1).cpu().numpy()); H.append(h.float().cpu().numpy())
    return np.vstack(P), np.vstack(H)

def train_model(name, tr, va, seed, verbose=5):
    set_seed(seed)
    model = BUILD[name]().to(DEVICE)
    trG = torch.as_tensor(tr, device=DEVICE)
    counts = torch.bincount(YG[trG], minlength=NCLS).float()
    cw = counts.sum() / (NCLS * counts)
    if IMBALANCE == "loss":
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTH)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH); sw = cw[YG[trG]]
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAIN[name], weight_decay=WD)
    steps = math.ceil(len(tr) / BATCH); total = EPOCHS * steps; warm = WARMUP * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    best, best_state, best_ep, bad, hist = -1, None, 0, 0, []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); losses = []
        order = trG[torch.randperm(len(trG), device=DEVICE)] if IMBALANCE == "loss" \
                else trG[torch.multinomial(sw, len(trG), replacement=True)]
        for s in range(steps):
            bi = order[s * BATCH:(s + 1) * BATCH]
            xb = gpu_augment(XG[bi].float()); yb = YG[bi]
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        pv, _ = predict(model, va)
        f1v = f1_score(Y[va], pv.argmax(1), average="macro")
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_macro_f1": float(f1v)})
        if verbose and (ep % verbose == 0 or ep == 1):
            print(f"   ep {ep:02d} loss={np.mean(losses):.4f} valF1={f1v:.4f} ({time.time()-t0:.0f}s)")
        if f1v > best + 1e-4:
            best, best_ep, bad = f1v, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    return model, hist, best_ep, time.time() - t0

def run_one(split, name, seed):
    tag = f"{split}_{name}_s{seed}_e{EPOCHS}"; f = RUNS/f"{tag}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        out = {k: d[k] for k in d.files}; out["hist"] = json.loads(str(out["hist"]))
        print(f"[loaded] {tag}"); return out
    tr, va, te = SPLITS[split]
    print(f"[train ] {tag}")
    model, hist, best_ep, secs = train_model(name, tr, va, seed)
    prob_va, lat_va = predict(model, va); prob_te, lat_te = predict(model, te); _, lat_tr = predict(model, tr)
    torch.save(model.state_dict(), RUNS/f"{tag}.pt")
    out = dict(prob_va=prob_va, prob_te=prob_te, lat_tr=lat_tr, lat_va=lat_va, lat_te=lat_te,
               hist=json.dumps(hist), best_epoch=best_ep, train_sec=secs, params=nparams(model))
    np.savez(f, **out); out["hist"] = hist
    del model; gc.collect(); torch.cuda.empty_cache()
    return out

def metrics(y, prob):
    pred = prob.argmax(1)
    p, r, _, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_precision": p, "macro_recall": r,
            "macro_f1": f1_score(y, pred, average="macro"), "weighted_f1": f1_score(y, pred, average="weighted"),
            "mcc": matthews_corrcoef(y, pred),
            "macro_auc": roc_auc_score(label_binarize(y, classes=range(NCLS)), prob, average="macro")}

# CELL 8 — STAGE 1: CLASSICAL BACKBONES x SEEDS, LOT-AWARE PROTOCOL
RES = {}
for name in MODELS:
    for seed in SEEDS:
        RES[("lot", name, seed)] = run_one("lot", name, seed)

Yte = Y[SPLITS["lot"][2]]; Yva = Y[SPLITS["lot"][1]]; Ytr = Y[SPLITS["lot"][0]]
rows = []
for (sp, name, seed), r in RES.items():
    rows.append({"model": name, "seed": seed, "best_epoch": int(r["best_epoch"]),
                 "params": int(r["params"]), "train_min": float(r["train_sec"]) / 60,
                 "val_macro_f1": f1_score(Yva, r["prob_va"].argmax(1), average="macro"),
                 **metrics(Yte, r["prob_te"])})
net_runs = pd.DataFrame(rows)
display(net_runs.round(4))
net_runs.to_csv(OUT/"tables/table_network_runs_per_seed.csv", index=False)

# ## Stage 3a — quantum reservoir (fixed circuit) and classical control
# * WaferMamba latent (64) → standardise → PCA with whitening (24 components, fitted on training data only) → angles π·tanh(x/2).
# * 8-qubit circuit, 3 re-uploading layers. Layer *l* encodes components 8*l … 8*l+7 with RY(θ), RZ(θ/2), followed by fixed random RY, RZ and a CNOT ring.
# * Read-out: ⟨Z_i⟩ (8) + ⟨Z_i Z_j⟩ for all pairs (28) = **36 features**, computed from the joint probabilities.
# * **Control:** 36 random Fourier features cos(Wx + b) of the same 24 inputs. If the quantum features help but the random features help equally, the gain is not quantum-specific.

# CELL 9 — VECTORISED 8-QUBIT RESERVOIR + BRUTE-FORCE VERIFICATION
_rng = np.random.default_rng(RES_SEED)
RAND_Y = _rng.uniform(-np.pi, np.pi, (QDEPTH, QUBITS))
RAND_Z = _rng.uniform(-np.pi, np.pi, (QDEPTH, QUBITS))
PAIRS = [(i, j) for i in range(QUBITS) for j in range(i + 1, QUBITS)]
_idx = np.arange(2 ** QUBITS)
SIGN = np.stack([1 - 2 * ((_idx >> (QUBITS - 1 - q)) & 1) for q in range(QUBITS)], 1).astype(np.float64)  # (256,8)
OBS = np.concatenate([SIGN, np.stack([SIGN[:, i] * SIGN[:, j] for i, j in PAIRS], 1)], 1)                # (256,36)

def ry_b(t):
    c, s = np.cos(t / 2), np.sin(t / 2); G = np.zeros((len(t), 2, 2), complex)
    G[:, 0, 0], G[:, 0, 1], G[:, 1, 0], G[:, 1, 1] = c, -s, s, c; return G
def rz_b(t):
    G = np.zeros((len(t), 2, 2), complex); G[:, 0, 0], G[:, 1, 1] = np.exp(-.5j * t), np.exp(.5j * t); return G
def apply1(psi, G, q, n=QUBITS):
    B = psi.shape[0]
    a = np.moveaxis(psi.reshape((B,) + (2,) * n), q + 1, 1).reshape(B, 2, -1)
    a = np.einsum("bij,bjk->bik", G, a) if G.ndim == 3 else np.einsum("ij,bjk->bik", G, a)
    return np.moveaxis(a.reshape((B, 2) + (2,) * (n - 1)), 1, q + 1).reshape(B, -1)
def cnot(psi, c, t, n=QUBITS):
    bit = (_idx >> (n - 1 - c)) & 1
    return psi[:, _idx ^ (bit << (n - 1 - t))]

def circuit_state(theta):                     # theta: (B, QUBITS*QDEPTH)
    B = theta.shape[0]; psi = np.zeros((B, 2 ** QUBITS), complex); psi[:, 0] = 1
    for l in range(QDEPTH):
        for q in range(QUBITS):
            a = theta[:, l * QUBITS + q]
            psi = apply1(psi, ry_b(a), q); psi = apply1(psi, rz_b(.5 * a), q)
            psi = apply1(psi, ry_b(np.array([RAND_Y[l, q]]))[0], q)
            psi = apply1(psi, rz_b(np.array([RAND_Z[l, q]]))[0], q)
        for q in range(QUBITS):               # CNOT ring
            psi = cnot(psi, q, (q + 1) % QUBITS)
    return psi

def reservoir(theta, chunk=2048):
    return np.vstack([(np.abs(circuit_state(theta[s:s + chunk])) ** 2) @ OBS
                      for s in range(0, len(theta), chunk)]).astype(np.float32)

# brute-force check with explicit Kronecker-product operators
def _kron(ops):
    M = np.array([[1.]]);
    for o in ops: M = np.kron(M, o)
    return M
def _brute(th):
    I = np.eye(2); Z = np.diag([1., -1.]); Xp = np.array([[0, 1], [1, 0]])
    P0, P1 = np.diag([1., 0.]), np.diag([0., 1.])
    def one(G, q): ops = [I] * QUBITS; ops[q] = G; return _kron(ops)
    def cx(c, t):
        a = [I] * QUBITS; a[c] = P0; b = [I] * QUBITS; b[c] = P1; b[t] = Xp; return _kron(a) + _kron(b)
    psi = np.zeros(2 ** QUBITS, complex); psi[0] = 1
    for l in range(QDEPTH):
        for q in range(QUBITS):
            a = th[l * QUBITS + q]
            for G in [ry_b(np.array([a]))[0], rz_b(np.array([.5 * a]))[0],
                      ry_b(np.array([RAND_Y[l, q]]))[0], rz_b(np.array([RAND_Z[l, q]]))[0]]:
                psi = one(G, q) @ psi
        for q in range(QUBITS): psi = cx(q, (q + 1) % QUBITS) @ psi
    ev = lambda O: np.real(np.conj(psi) @ O @ psi)
    zs = [ev(one(Z, q)) for q in range(QUBITS)]
    zz = [ev(_kron([Z if k in (i, j) else I for k in range(QUBITS)])) for i, j in PAIRS]
    return np.array(zs + zz)

_th = np.random.default_rng(1).uniform(-np.pi, np.pi, (2, QUBITS * QDEPTH))
_fast = reservoir(_th)
for k in range(2): assert np.allclose(_fast[k], _brute(_th[k]), atol=1e-5)
_prod = _fast[:, [i for i, _ in PAIRS]] * _fast[:, [j for _, j in PAIRS]]
print("Reservoir vs brute-force simulation: PASSED")
print("Max |<ZiZj> - <Zi><Zj>| on test states:", float(np.abs(_fast[:, QUBITS:] - _prod).max()),
      "(non-zero => correlators carry information the v1 product did not)")

def make_features(Htr, Hva, Hte):
    ss = StandardScaler().fit(Htr)
    pca = PCA(n_components=QUBITS * QDEPTH, whiten=True, random_state=RES_SEED).fit(ss.transform(Htr))
    ang = lambda H: np.pi * np.tanh(pca.transform(ss.transform(H)) / 2)
    Atr, Ava, Ate = ang(Htr), ang(Hva), ang(Hte)
    Q = [reservoir(A) for A in (Atr, Ava, Ate)]
    r = np.random.default_rng(RES_SEED + 1)
    W = r.normal(0, 1, (QUBITS * QDEPTH, OBS.shape[1])); b = r.uniform(0, 2 * np.pi, OBS.shape[1])
    R = [np.cos(A @ W + b).astype(np.float32) for A in (Atr, Ava, Ate)]
    return Q, R, float(pca.explained_variance_ratio_.sum())

# ## Quantum circuits in Qiskit
# Cell 10 builds the two RealAmplitudes circuits and the reservoir circuit in Qiskit, saves the circuit diagrams, and checks that the fast simulators used for training reproduce `qiskit.quantum_info.Statevector` exactly.

# CELL 10 — QISKIT CIRCUITS (RESERVOIR + COBYLA VQC), DRAWINGS, SIMULATOR VERIFICATION
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
try:
    from qiskit.circuit.library import real_amplitudes
    make_ansatz = lambda n, reps: real_amplitudes(n, entanglement="reverse_linear", reps=reps)
except ImportError:
    from qiskit.circuit.library import RealAmplitudes
    make_ansatz = lambda n, reps: RealAmplitudes(n, entanglement="reverse_linear", reps=reps)

def save_circuit(qc, path, fold=40):
    try:
        fig = qc.draw(output="mpl", fold=fold, style="iqp")
        fig.savefig(path, dpi=300, bbox_inches="tight"); plt.show()
    except Exception as e:                # matplotlib drawer needs pylatexenc
        print("mpl drawing failed (", e, ") -> saving text drawing")
        Path(str(path).replace(".png", ".txt")).write_text(str(qc.draw(output="text", fold=120)))
        print(qc.draw(output="text", fold=120))

def flatten_ry_cx(qc):                    # unpack wrapper gates only, keep RY and CX
    for _ in range(4):
        other = {i.operation.name for i in qc.data} - {"ry", "cx", "barrier"}
        if not other: break
        qc = qc.decompose(gates_to_decompose=list(other))
    return qc

def compile_program(flat, xvec, thetas):  # Qiskit circuit -> gate list for the batched simulators
    ref_of = {p: ("x", i) for i, p in enumerate(xvec)}; ref_of.update({p: ("t", j) for j, p in enumerate(thetas)})
    prog = []
    for ins in flat.data:
        op, qs = ins.operation, [flat.find_bit(b).index for b in ins.qubits]
        if op.name == "barrier": continue
        if op.name == "ry":
            p = op.params[0]; prog.append(("ry", qs, ref_of[p] if isinstance(p, Parameter) else ("c", float(p))))
        elif op.name == "cx": prog.append(("cx", qs, None))
        else: raise NotImplementedError(op.name)
    return prog

# ---------- (a) quantum reservoir as a Qiskit circuit ----------
xr = ParameterVector("x", QUBITS * QDEPTH)
res_qc = QuantumCircuit(QUBITS, name="QReservoir")
for l in range(QDEPTH):
    for q in range(QUBITS):
        res_qc.ry(xr[l * QUBITS + q], q); res_qc.rz(0.5 * xr[l * QUBITS + q], q)
        res_qc.ry(float(RAND_Y[l, q]), q); res_qc.rz(float(RAND_Z[l, q]), q)
    for q in range(QUBITS): res_qc.cx(q, (q + 1) % QUBITS)
    if l < QDEPTH - 1: res_qc.barrier()
print("Reservoir circuit: depth", res_qc.depth(), "| ops", dict(res_qc.count_ops()))
save_circuit(res_qc, OUT/"figures/fig_circuit_reservoir.png")

obs = [SparsePauliOp.from_sparse_list([("Z", [q], 1)], QUBITS) for q in range(QUBITS)] + \
      [SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1)], QUBITS) for i, j in PAIRS]
_th = np.random.default_rng(3).uniform(-np.pi, np.pi, (2, QUBITS * QDEPTH))
for k in range(2):
    sv = Statevector(res_qc.assign_parameters(dict(zip(xr, _th[k]))))
    ref = np.array([np.real(sv.expectation_value(o)) for o in obs])
    assert np.allclose(reservoir(_th[k:k + 1])[0], ref, atol=1e-5)
print("Reservoir simulator vs Qiskit Statevector: PASSED")

# ---------- (b) COBYLA VQC: angle encoding + RealAmplitudes, readout on qubits 0,1,2 ----------
xv = ParameterVector("x", VQC_QUBITS)
fmap = QuantumCircuit(VQC_QUBITS, name="AngleEncoding")
for q in range(VQC_QUBITS): fmap.ry(xv[q], q)
ansatz = make_ansatz(VQC_QUBITS, VQC_REPS)
THETA = list(ansatz.parameters)
vqc_qc = fmap.compose(ansatz)
vqc_flat = flatten_ry_cx(vqc_qc)
PROGRAM = compile_program(vqc_flat, xv, THETA)
print(f"VQC: {VQC_QUBITS} qubits, RealAmplitudes reps={VQC_REPS}, trainable θ = {len(THETA)}, "
      f"depth = {vqc_flat.depth()}, ops = {dict(vqc_flat.count_ops())}")
vqc_draw = vqc_flat.copy(); vqc_draw.add_register(ClassicalRegister(3, "c"))
vqc_draw.barrier(); vqc_draw.measure([0, 1, 2], [0, 1, 2])
save_circuit(vqc_draw, OUT/"figures/fig_circuit_vqc_realamplitudes.png")

CLASS_MASK = np.stack([(np.arange(2 ** VQC_QUBITS) & 7) == k for k in range(NCLS)], 1).astype(float)

def vqc_probs(Xa, theta):
    # Xa: (B, 8) encoding angles; returns (B, 8) class probabilities = P(qubits 0,1,2)
    B, n = len(Xa), VQC_QUBITS
    psi = np.zeros((B,) + (2,) * n); psi[(slice(None),) + (0,) * n] = 1.0     # RY/CX keep amplitudes real
    for name, qs, ref in PROGRAM:
        if name == "ry":
            a = Xa[:, ref[1]] if ref[0] == "x" else np.full(B, theta[ref[1]] if ref[0] == "t" else ref[1])
            ax = n - qs[0]                                   # Qiskit little-endian qubit -> array axis
            c = np.cos(a / 2).reshape([B] + [1] * (n - 1)); s = np.sin(a / 2).reshape([B] + [1] * (n - 1))
            p0, p1 = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
            psi = np.stack([c * p0 - s * p1, s * p0 + c * p1], axis=ax)
        else:
            ca, ta = n - qs[0], n - qs[1]
            sl = [slice(None)] * (n + 1); sl[ca] = 1; sl = tuple(sl)
            psi = psi.copy(); psi[sl] = np.flip(psi[sl], axis=ta - 1 if ta > ca else ta)
    return (psi.reshape(B, -1) ** 2) @ CLASS_MASK

_x = np.random.default_rng(4).uniform(-np.pi, np.pi, (2, VQC_QUBITS))
_t = np.random.default_rng(5).uniform(-np.pi, np.pi, len(THETA))
for k in range(2):
    sv = Statevector(vqc_qc.assign_parameters({**dict(zip(xv, _x[k])), **dict(zip(THETA, _t))}))
    assert np.allclose(vqc_probs(_x[k:k + 1], _t)[0], sv.probabilities([0, 1, 2]), atol=1e-6)
print("VQC simulator vs Qiskit Statevector: PASSED")

# ## Stage 2 — QWafer-Mamba, classical head controls and quantum-layer ablations
# Every stage-2 model starts from the same stage-1 WaferMamba checkpoint of its seed and uses the identical schedule: `FT_HEAD_WARMUP` epochs with the backbone frozen, then joint training; softened class weights; AdamW with the same two learning rates; the same early-stopping rule on validation Macro-F1. Only the decision layer differs.
# 
# | Kind | Decision layer on the 64-d latent *h* | Purpose |
# |---|---|---|
# | `QWaferMamba_E2E` | Linear(64→8)+LN → α = π·tanh → RY(α) + RealAmplitudes reps 3 (32 θ, 21 CNOT) → ⟨Z_q⟩ → s·⟨Z_q⟩ + b | proposed |
# | `WaferMamba_FT` | the backbone's own Linear(64→8) | same-schedule linear control |
# | `MLP_FT` | Linear(64→8)+LN → GELU → Linear(8→8) | nonlinear control of matched size (R1-9) |
# | `Tanh_FT` | Linear(64→8)+LN → α = π·tanh → tanh(α) → Linear(8→8) | bounded-activation control (R1-9) |
# | `Trig_FT` | Linear(64→8)+LN → α = π·tanh → [cos α, sin α] → Linear(16→8) | classical trigonometric control (R1-10) |
# | `Q_NoEnt` | as proposed, all CNOT gates removed (product state) | is entanglement needed? |
# | `Q_Frozen` | as proposed, θ fixed at its random initial value | are trained rotations needed? |
# | `Q_Reps0`, `Q_Reps1` | as proposed with RealAmplitudes reps 0 (8 θ) and reps 1 (16 θ, 7 CNOT) | depth ablation |

# CELL 11 — TRAINABLE QUANTUM LAYER (RealAmplitudes) + VERIFICATION + HYBRID MODEL
xh = ParameterVector("x", HQ_QUBITS)
hq_qc = QuantumCircuit(HQ_QUBITS, name="QLayer")
for q in range(HQ_QUBITS): hq_qc.ry(xh[q], q)
h_ans = make_ansatz(HQ_QUBITS, HQ_REPS); HTHETA = list(h_ans.parameters)
hq_qc = hq_qc.compose(h_ans)
hq_flat = flatten_ry_cx(hq_qc)
HPROG = compile_program(hq_flat, xh, HTHETA)
print(f"Quantum layer: {HQ_QUBITS} qubits, RealAmplitudes reps={HQ_REPS}, trainable θ = {len(HTHETA)}, "
      f"depth = {hq_flat.depth()}, ops = {dict(hq_flat.count_ops())}")
hq_draw = hq_flat.copy(); hq_draw.add_register(ClassicalRegister(HQ_QUBITS, "c"))
hq_draw.barrier(); hq_draw.measure(range(HQ_QUBITS), range(HQ_QUBITS))
save_circuit(hq_draw, OUT/"figures/fig_circuit_qwafer_quantum_layer.png")

_bits = np.arange(2 ** HQ_QUBITS)
ZSIGN = torch.tensor(np.stack([1 - 2 * ((_bits >> q) & 1) for q in range(HQ_QUBITS)], 1), dtype=torch.float32)

class TorchQLayer(nn.Module):
    # Differentiable statevector simulation of a compiled Qiskit circuit; returns <Z_q> for every qubit.
    def __init__(self, prog=None, ntheta=None, trainable=True):
        super().__init__()
        self.prog = HPROG if prog is None else prog
        self.theta = nn.Parameter(0.1 * torch.randn(len(HTHETA) if ntheta is None else ntheta), requires_grad=trainable)
        self.register_buffer("zsign", ZSIGN.clone())
    def forward(self, ang):
        return self.probs(ang) @ self.zsign.to(ang.dtype)    # (B, 8) expectation values
    def probs(self, ang):                                    # ang: (B, 8) rotation angles -> (B, 256) probabilities
        B, n = ang.size(0), HQ_QUBITS
        psi = torch.zeros((B,) + (2,) * n, device=ang.device, dtype=ang.dtype)
        psi[(slice(None),) + (0,) * n] = 1.0
        for name, qs, ref in self.prog:
            if name == "ry":
                if ref[0] == "x": a = ang[:, ref[1]]
                elif ref[0] == "t": a = self.theta[ref[1]].expand(B)
                else: a = torch.full((B,), ref[1], device=ang.device, dtype=ang.dtype)
                ax = n - qs[0]; sh = (B,) + (1,) * (n - 1)
                c, s = torch.cos(a / 2).view(sh), torch.sin(a / 2).view(sh)
                p0, p1 = psi.select(ax, 0), psi.select(ax, 1)
                psi = torch.stack([c * p0 - s * p1, s * p0 + c * p1], dim=ax)
            else:
                ca, ta = n - qs[0], n - qs[1]
                p0, p1 = psi.select(ca, 0), psi.select(ca, 1).flip(ta - 1 if ta > ca else ta)
                psi = torch.stack([p0, p1], dim=ca)
        return psi.reshape(B, -1) ** 2

# check 1: torch quantum layer == Qiskit Statevector
_ql = TorchQLayer().double()
_a = torch.tensor(np.random.default_rng(6).uniform(-np.pi, np.pi, (2, HQ_QUBITS)))
with torch.no_grad(): _ez = _ql(_a).numpy()
for k in range(2):
    bind = {**dict(zip(xh, _a[k].numpy())), **dict(zip(HTHETA, _ql.theta.detach().numpy()))}
    sv = Statevector(hq_qc.assign_parameters(bind))
    ref = [np.real(sv.expectation_value(SparsePauliOp.from_sparse_list([("Z", [q], 1)], HQ_QUBITS))) for q in range(HQ_QUBITS)]
    assert np.allclose(_ez[k], ref, atol=1e-6)
print("Quantum layer vs Qiskit Statevector: PASSED")

# check 2: training gradient == parameter-shift rule gradient
_ql.zero_grad(); _ql(_a)[:, 0].sum().backward(); g_auto = _ql.theta.grad.clone()
g_ps = torch.zeros_like(g_auto)
with torch.no_grad():
    for j in range(len(HTHETA)):
        e = torch.zeros_like(g_auto); e[j] = np.pi / 2
        _ql.theta += e;  fp = _ql(_a)[:, 0].sum()
        _ql.theta -= 2 * e; fm = _ql(_a)[:, 0].sum()
        _ql.theta += e;  g_ps[j] = (fp - fm) / 2
assert torch.allclose(g_auto, g_ps, atol=1e-8)
print("Training gradient vs parameter-shift gradient: PASSED (max diff %.1e)" % (g_auto - g_ps).abs().max())
del _ql

# ---------- circuit variants for the ablation study (R1-10) ----------
VARIANTS = {"full": (hq_qc, HPROG, len(HTHETA), HTHETA)}
VARIANTS["noent"] = (None, [op for op in HPROG if op[0] != "cx"], len(HTHETA), HTHETA)    # all CNOTs removed
for r_ in (0, 1):
    qc_ = QuantumCircuit(HQ_QUBITS)
    for q in range(HQ_QUBITS): qc_.ry(xh[q], q)
    an_ = make_ansatz(HQ_QUBITS, r_); th_ = list(an_.parameters); qc_ = qc_.compose(an_)
    VARIANTS[f"reps{r_}"] = (qc_, compile_program(flatten_ry_cx(qc_), xh, th_), len(th_), th_)
for k, (qc_, pg_, nt_, _) in VARIANTS.items():
    print(f"variant {k:6s}: trainable θ = {nt_:2d}, CNOT = {sum(op[0] == 'cx' for op in pg_):2d}, RY = {sum(op[0] == 'ry' for op in pg_)}")
for k in ("reps0", "reps1"):                          # verify the compiled variants against Qiskit as well
    qc_, pg_, nt_, th_ = VARIANTS[k]; _ql = TorchQLayer(pg_, nt_).double()
    with torch.no_grad(): _ez = _ql(_a).numpy()
    for j in range(2):
        sv = Statevector(qc_.assign_parameters({**dict(zip(xh, _a[j].numpy())), **dict(zip(th_, _ql.theta.detach().numpy()))}))
        ref = [np.real(sv.expectation_value(SparsePauliOp.from_sparse_list([("Z", [q], 1)], HQ_QUBITS))) for q in range(HQ_QUBITS)]
        assert np.allclose(_ez[j], ref, atol=1e-6)
print("Ablation circuits vs Qiskit Statevector: PASSED")

def encoder_latent(b, x):                              # 64-d latent h of a WaferMamba backbone
    t = b.norm(b.blocks(b.stem(x).permute(0, 2, 3, 1) + b.pos)).mean((1, 2))
    return b.head.proj(t)

class QWaferMamba(nn.Module):
    def __init__(self, variant="full", trainable_theta=True):
        super().__init__()
        _, prog, nt, _ = VARIANTS[variant]
        self.backbone = WaferMamba()                          # Stage 1 encoder (SS2D)
        self.bridge = nn.Sequential(nn.Linear(LATENT, HQ_QUBITS), nn.LayerNorm(HQ_QUBITS))
        self.qlayer = TorchQLayer(prog, nt, trainable_theta)  # quantum circuit (trainable unless frozen)
        self.scale = nn.Parameter(torch.tensor(3.0)); self.bias = nn.Parameter(torch.zeros(NCLS))
    def angles(self, x):                                      # rotation angles fed to the circuit
        h = encoder_latent(self.backbone, x)
        with torch.autocast(device_type="cuda", enabled=False):
            return np.pi * torch.tanh(self.bridge(h.float()))
    def forward(self, x, return_latent=False):
        h = encoder_latent(self.backbone, x)
        with torch.autocast(device_type="cuda", enabled=False):
            ez = self.qlayer(np.pi * torch.tanh(self.bridge(h.float())))   # latent -> angles -> circuit -> <Z_q>
            z = self.scale * ez + self.bias                                 # one readout qubit per class
        return (z, ez) if return_latent else z

class ClassicalHeadNet(nn.Module):
    # WaferMamba backbone with a capacity-matched classical decision layer (R1-9, R1-10)
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.backbone = WaferMamba()
        self.bridge = nn.Sequential(nn.Linear(LATENT, HQ_QUBITS), nn.LayerNorm(HQ_QUBITS))
        self.out = nn.Linear(2 * HQ_QUBITS if kind == "Trig_FT" else HQ_QUBITS, NCLS)
    def forward(self, x, return_latent=False):
        h = encoder_latent(self.backbone, x)
        with torch.autocast(device_type="cuda", enabled=False):
            u = self.bridge(h.float())
            if self.kind == "MLP_FT":    f = F.gelu(u)
            elif self.kind == "Tanh_FT": f = torch.tanh(np.pi * torch.tanh(u))
            else:                        a = np.pi * torch.tanh(u); f = torch.cat([torch.cos(a), torch.sin(a)], 1)
            z = self.out(f)
        return (z, f) if return_latent else z

BUILD["QWaferMamba"] = QWaferMamba
_m = QWaferMamba(); print("QWafer-Mamba parameters:", f"{nparams(_m):,}", "| quantum θ:", len(HTHETA)); del _m

# CELL 12 — STAGE 2: END-TO-END TRAINING OF QWAFER-MAMBA + SAME-SCHEDULE CLASSICAL CONTROL
QKINDS = {"QWaferMamba_E2E": ("full", True), "Q_NoEnt": ("noent", True), "Q_Frozen": ("full", False),
          "Q_Reps0": ("reps0", True), "Q_Reps1": ("reps1", True)}

def build_stage2(kind):
    if kind in QKINDS: return QWaferMamba(*QKINDS[kind])
    if kind in ("MLP_FT", "Tanh_FT", "Trig_FT"): return ClassicalHeadNet(kind)
    return WaferMamba()

def finetune(kind, seed, split="lot", verbose=1):
    # every kind starts from the same stage-1 checkpoint and uses the identical stage-2 schedule
    tag = f"{split}_{kind}_s{seed}_e{EPOCHS}_ft{FT_EPOCHS}_w{FT_HEAD_WARMUP}_cw{FT_CW_POWER}"; f = RUNS/f"{tag}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False); out = {k: d[k] for k in d.files}
        out["hist"] = json.loads(str(out["hist"])); print(f"[loaded] {tag}"); return out
    print(f"[train ] {tag}")
    tr, va, te = SPLITS[split]; set_seed(seed)
    base = torch.load(RUNS/f"{split}_WaferMamba_s{seed}_e{EPOCHS}.pt", map_location="cpu")
    if kind != "WaferMamba_FT":
        m = build_stage2(kind); m.backbone.load_state_dict(base)
        head_p = [p for n_, p in m.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]
        body_p = [p for n_, p in m.backbone.named_parameters() if not n_.startswith("head.cls")]
        for p in m.backbone.head.cls.parameters(): p.requires_grad_(False)   # stored but unused (R1-4)
    else:
        m = WaferMamba(); m.load_state_dict(base)
        head_p = list(m.head.cls.parameters())
        body_p = [p for n_, p in m.named_parameters() if not n_.startswith("head.cls")]
    m = m.to(DEVICE)
    trG = torch.as_tensor(tr, device=DEVICE)
    counts = torch.bincount(YG[trG], minlength=NCLS).float()
    cw = (counts.sum() / (NCLS * counts)) ** FT_CW_POWER             # softened class weights (both arms)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTH)
    opt = torch.optim.AdamW([{"params": body_p, "lr": LR_BACKBONE},
                             {"params": head_p, "lr": LR_QUANTUM, "weight_decay": 0.0}], weight_decay=WD)
    steps = math.ceil(len(tr) / BATCH); total = FT_EPOCHS * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    best, best_state, best_ep, bad, hist, t0 = -1, None, 0, 0, [], time.time()
    for ep in range(1, FT_EPOCHS + 1):
        m.train(); losses = []
        for p in body_p: p.requires_grad_(ep > FT_HEAD_WARMUP)          # head warm-up: backbone frozen first
        order = trG[torch.randperm(len(trG), device=DEVICE)]
        for s in range(steps):
            bi = order[s * BATCH:(s + 1) * BATCH]
            xb = gpu_augment(XG[bi].float()); yb = YG[bi]
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss = crit(m(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 2.0)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(loss.item())
        pv, _ = predict(m, va); f1v = f1_score(Y[va], pv.argmax(1), average="macro")
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_macro_f1": float(f1v)})
        if verbose: print(f"   ep {ep:02d} loss={np.mean(losses):.4f} valF1={f1v:.4f} ({time.time()-t0:.0f}s)")
        if f1v > best + 1e-4:
            best, best_ep, bad = f1v, ep, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= FT_PATIENCE: break
    m.load_state_dict(best_state)
    prob_va, _ = predict(m, va); prob_te, q_te = predict(m, te)
    if not np.isfinite(prob_te).all(): raise FloatingPointError(f"non-finite predictions in {tag}")
    torch.save(m.state_dict(), RUNS/f"{tag}.pt")
    out = dict(prob_va=prob_va, prob_te=prob_te, qout_te=q_te, hist=json.dumps(hist), best_epoch=best_ep,
               train_sec=time.time() - t0, params=nparams(m),
               theta=m.qlayer.theta.detach().cpu().numpy() if kind in QKINDS else np.zeros(1))
    np.savez(f, **out); out["hist"] = hist
    del m; gc.collect(); torch.cuda.empty_cache()
    return out

STAGE2_KINDS = ["QWaferMamba_E2E", "WaferMamba_FT"] + (HEAD_KINDS if RUN_HEAD_CONTROLS else []) \
               + (ABL_KINDS if RUN_CIRCUIT_ABLATIONS else [])
for seed in SEEDS:
    for kind in STAGE2_KINDS:
        RES[("lot", kind, seed)] = finetune(kind, seed)

rows = []
for kind in ["WaferMamba"] + STAGE2_KINDS:
    for seed in SEEDS:
        r = RES[("lot", kind, seed)]
        rows.append({"model": kind, "seed": seed, "best_epoch": int(r["best_epoch"]), "params": int(r["params"]),
                     "val_macro_f1": f1_score(Yva, r["prob_va"].argmax(1), average="macro"), **metrics(Yte, r["prob_te"])})
e2e_runs = pd.DataFrame(rows)
e2e_runs.to_csv(OUT/"tables/table_stage2_hybrid_per_seed.csv", index=False)
abl = e2e_runs.groupby("model", sort=False)[["accuracy", "macro_f1", "mcc", "macro_auc"]].agg(["mean", "std"])
abl.to_csv(OUT/"tables/table_stage2_heads_ablation_mean_std.csv"); display((100 * abl).round(2))
pd.DataFrame({f"seed {s}": RES[("lot", "QWaferMamba_E2E", s)]["theta"] for s in SEEDS},
             index=[str(p) for p in HTHETA]).to_csv(OUT/"tables/table_quantum_layer_trained_theta.csv")

# CELL 13 — STAGE 3a: QUANTUM RESERVOIR AND CLASSICAL CONTROL ON THE FROZEN LATENT (identical logistic regression)
def lr_head(A, C):
    ss = StandardScaler().fit(A)
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=0)
    clf.fit(ss.transform(A), Ytr)
    return clf.predict_proba(ss.transform(C))

HEADS, pca_var = {}, {}
for seed in SEEDS:
    t0 = time.time(); r = RES[("lot", "WaferMamba", seed)]
    Htr, Hva, Hte = r["lat_tr"], r["lat_va"], r["lat_te"]
    (Qtr, Qva, Qte), (Rtr, Rva, Rte), pca_var[seed] = make_features(Htr, Hva, Hte)
    sets = {"Latent": (Htr, Hte), "Quantum reservoir only": (Qtr, Qte), "RFF only": (Rtr, Rte),
            "Latent + Quantum reservoir": (np.c_[Htr, Qtr], np.c_[Hte, Qte]),
            "Latent + RFF (control)": (np.c_[Htr, Rtr], np.c_[Hte, Rte])}
    for k, (A, C) in sets.items(): HEADS[(seed, k)] = lr_head(A, C)
    print(f"seed {seed}: heads done in {time.time()-t0:.0f}s | PCA variance kept = {pca_var[seed]:.3f}")

# ## Stage 3b — VQC trained with COBYLA (gradient-free quantum baseline)
# Loop: circuit output → cross-entropy loss → COBYLA updates the 24 θ → circuit runs again (up to `VQC_MAXITER` evaluations). The trained circuit's 8 class probabilities are used directly as a prediction ("VQC only") and are also joined to the 64-d latent and passed to the same logistic regression ("Latent + VQC").

# CELL 14 — STAGE 3b: VQC (RealAmplitudes) TRAINED WITH COBYLA + FUSION WITH THE LATENT
from scipy.optimize import minimize

def vqc_angles(Htr, *others):
    ss = StandardScaler().fit(Htr)
    pca = PCA(n_components=VQC_QUBITS, whiten=True, random_state=RES_SEED).fit(ss.transform(Htr))
    f = lambda H: np.pi * np.tanh(pca.transform(ss.transform(H)) / 2)
    return [f(Htr)] + [f(H) for H in others]

def train_vqc(Xa, y, seed):
    rng = np.random.default_rng(seed)
    sub = np.concatenate([rng.choice(np.where(y == k)[0], min(VQC_PER_CLASS, (y == k).sum()), replace=False)
                          for k in range(NCLS)])
    Xs, ys = Xa[sub], y[sub]; log = []
    def loss(th):
        p = vqc_probs(Xs, th)[np.arange(len(ys)), ys]
        l = float(-np.mean(np.log(np.clip(p, 1e-10, 1)))); log.append(l); return l
    th0 = rng.uniform(-np.pi, np.pi, len(THETA))
    res = minimize(loss, th0, method="COBYLA", options={"maxiter": VQC_MAXITER, "rhobeg": 0.5})
    return res.x, log, len(sub)

VQC_LOG, vqc_rows = {}, []
for seed in SEEDS:
    t0 = time.time(); r = RES[("lot", "WaferMamba", seed)]
    Atr, Ava, Ate = vqc_angles(r["lat_tr"], r["lat_va"], r["lat_te"])
    th, VQC_LOG[seed], nsub = train_vqc(Atr, Ytr, seed)
    Ptr, Pte = vqc_probs(Atr, th), vqc_probs(Ate, th)
    HEADS[(seed, "VQC only (RealAmplitudes + COBYLA)")] = Pte
    HEADS[(seed, "Latent + VQC")] = lr_head(np.c_[r["lat_tr"], Ptr], np.c_[r["lat_te"], Pte])
    vqc_rows.append({"seed": seed, "train_subset": nsub, "cobyla_evals": len(VQC_LOG[seed]),
                     "initial_loss": VQC_LOG[seed][0], "final_loss": min(VQC_LOG[seed]),
                     "theta": json.dumps(np.round(th, 6).tolist()), "minutes": (time.time() - t0) / 60})
    print(f"seed {seed}: COBYLA {len(VQC_LOG[seed])} evals, loss {VQC_LOG[seed][0]:.4f} -> {min(VQC_LOG[seed]):.4f}, "
          f"VQC test Macro-F1 = {f1_score(Yte, Pte.argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)")
pd.DataFrame(vqc_rows).to_csv(OUT/"tables/table_vqc_cobyla_training.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
for s, l in VQC_LOG.items(): ax.plot(np.minimum.accumulate(l), label=f"seed {s}")
ax.axhline(np.log(NCLS), ls="--", c="k", lw=.8, label="random guess (ln 8)")
ax.set_xlabel("COBYLA function evaluation"); ax.set_ylabel("Cross-entropy loss (best so far)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/"figures/fig_vqc_cobyla_convergence.png", dpi=300, bbox_inches="tight"); plt.show()

head_runs = pd.DataFrame([{"head": k, "seed": s, **metrics(Yte, p)} for (s, k), p in HEADS.items()])
head_runs.to_csv(OUT/"tables/table_stage3_heads_per_seed.csv", index=False)
display(head_runs.groupby("head")[["accuracy", "macro_f1", "mcc"]].mean().round(4))

# CELL 15 — SIGNIFICANCE: EXACT McNEMAR + BOOTSTRAP 95% CI OF ΔMacro-F1, AND QUANTUM VERDICTS
def mcnemar(y, pa, pb):
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return b, c, (binomtest(b, b + c, .5).pvalue if b + c else 1.0)

def boot_delta(y, pa, pb, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed); d = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(y), len(y))
        d[i] = f1_score(y[j], pb[j], average="macro") - f1_score(y[j], pa[j], average="macro")
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)

def available(seed, k):
    return (("lot", k[:-5], seed) in RES) if k.endswith(" head") else ((seed, k) in HEADS)
def preds(seed, k):
    return RES[("lot", k[:-5], seed)]["prob_te"].argmax(1) if k.endswith(" head") else HEADS[(seed, k)].argmax(1)

COMPARE = [("WaferMamba_FT head", "QWaferMamba_E2E head"),      # proposed vs exact classical control
           ("WaferMamba head", "QWaferMamba_E2E head"),
           ("ResNet18 head", "QWaferMamba_E2E head"),
           ("CNN head", "QWaferMamba_E2E head"),
           ("Latent", "Latent + Quantum reservoir"),
           ("Latent + RFF (control)", "Latent + Quantum reservoir"),
           ("Latent", "Latent + VQC"),
           ("Latent + RFF (control)", "Latent + VQC"),
           ("Latent", "Latent + RFF (control)"),
           ("CNN head", "WaferMamba head"), ("ResNet18 head", "WaferMamba head"),
           ("EfficientNetB0 head", "QWaferMamba_E2E head"), ("MobileViTXXS head", "QWaferMamba_E2E head")]
COMPARE += [(f"{k} head", "QWaferMamba_E2E head") for k in HEAD_KINDS + ABL_KINDS]
COMPARE += [("WaferMamba_FT head", f"{k} head") for k in HEAD_KINDS]
rows = []
for seed in SEEDS:
    for a, b in COMPARE:
        if not (available(seed, a) and available(seed, b)): continue
        pa, pb = preds(seed, a), preds(seed, b)
        bb, cc, p = mcnemar(Yte, pa, pb); m, lo, hi = boot_delta(Yte, pa, pb, seed=seed)
        rows.append({"seed": seed, "A": a, "B": b, "dMacroF1(B-A)": m, "CI_low": lo, "CI_high": hi,
                     "A_only_correct": bb, "B_only_correct": cc, "McNemar_p": p})
sig = pd.DataFrame(rows)
sig.to_csv(OUT/"tables/table_significance_per_seed.csv", index=False)
display(sig.round(4))

def verdict(title, a, b, control=None):
    q = sig[(sig.A == a) & (sig.B == b)]; d = q["dMacroF1(B-A)"].values
    p = ttest_1samp(d, 0).pvalue if len(d) > 1 else np.nan
    ci = int((q.CI_low > 0).sum())
    beats = True if control is None else bool(sig[(sig.A == control) & (sig.B == b)]["dMacroF1(B-A)"].mean() > 0)
    ok = bool(d.mean() > 0 and ci >= math.ceil(len(SEEDS) / 2) and beats and (len(SEEDS) == 1 or p < .05))
    print(f"{title}: mean ΔMacro-F1 = {d.mean():+.4f} | seeds with CI>0: {ci}/{len(SEEDS)} | p = {p:.4f}"
          + ("" if control is None else f" | beats classical control: {beats}") + f"  ->  {'SUPPORTED' if ok else 'NOT supported'}")
    return {"mean_delta_macro_f1": float(d.mean()), "seeds_ci_positive": ci,
            "ttest_p_across_seeds": None if np.isnan(p) else float(p),
            "beats_classical_control": beats, "claim_supported": ok}

VERDICTS = {
    "QWaferMamba_E2E_vs_same_schedule_classical": verdict("Stage 2  QWafer-Mamba (trainable quantum layer) vs classical head",
                                                          "WaferMamba_FT head", "QWaferMamba_E2E head"),
    "quantum_reservoir_fusion": verdict("Stage 3a Quantum reservoir fusion vs latent", "Latent",
                                        "Latent + Quantum reservoir", control="Latent + RFF (control)"),
    "vqc_cobyla_fusion": verdict("Stage 3b COBYLA VQC fusion vs latent", "Latent", "Latent + VQC",
                                 control="Latent + RFF (control)")}

# CELL 16 — MAIN RESULTS TABLE (mean ± std over seeds) + COMPARISON CHART
cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "mcc", "macro_auc"]
LABEL = {"CNN": "CNN (network head)", "ResNet18": "ResNet-18 (network head)", "WaferMamba": "WaferMamba (network head)",
         "WaferMamba_FT": "WaferMamba fine-tuned, classical head (control)",
         "QWaferMamba_E2E": "QWafer-Mamba, trainable quantum layer (proposed)",
         "EfficientNetB0": "EfficientNet-B0 (network head)", "MobileViTXXS": "MobileViT-XXS (network head)",
         "MLP_FT": "WaferMamba fine-tuned, MLP head (control)", "Tanh_FT": "WaferMamba fine-tuned, bounded tanh head (control)",
         "Trig_FT": "WaferMamba fine-tuned, trigonometric head (control)",
         "Q_NoEnt": "QWafer-Mamba ablation: no entanglement", "Q_Frozen": "QWafer-Mamba ablation: frozen θ",
         "Q_Reps0": "QWafer-Mamba ablation: reps = 0", "Q_Reps1": "QWafer-Mamba ablation: reps = 1"}
allrows = [{"method": LABEL[k[1]], "params": int(r["params"]), **metrics(Yte, r["prob_te"])}
           for k, r in RES.items() if k[0] == "lot" and k[1] in LABEL]
allrows += [{"method": (r.head if r.head.startswith("VQC only") else f"WaferMamba latent → LR: {r.head}"),
             **{c: getattr(r, c) for c in cols}} for r in head_runs.itertuples()]
long = pd.DataFrame(allrows)
mean, std = long.groupby("method")[cols].mean(), long.groupby("method")[cols].std().fillna(0)
main = (mean.map(lambda v: f"{100*v:.2f}") + " ± " + std.map(lambda v: f"{100*v:.2f}"))
main["params"] = long.groupby("method")["params"].first().map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
main = main.loc[mean.sort_values("macro_f1", ascending=False).index]
display(main)
main.to_csv(OUT/"tables/table_main_results_mean_std.csv")
mean.to_csv(OUT/"tables/table_main_results_mean.csv"); std.to_csv(OUT/"tables/table_main_results_std.csv")

fig, ax = plt.subplots(figsize=(9, 6))
order = mean.sort_values("macro_f1").index
colors = ["tab:purple" if ("quantum" in o.lower() or "vqc" in o.lower()) else "tab:blue" for o in order]
ax.barh(range(len(order)), 100 * mean.loc[order, "macro_f1"], xerr=100 * std.loc[order, "macro_f1"], capsize=3, color=colors)
ax.set_yticks(range(len(order)), order, fontsize=8); ax.set_xlabel("Macro-F1 (%)  (purple = quantum)"); ax.set_xlim(0, 100)
fig.tight_layout(); fig.savefig(OUT/"figures/fig_method_comparison.png", dpi=300, bbox_inches="tight"); plt.show()

# CELL 17 — PER-CLASS RESULTS, CONFUSION MATRICES, ROC (proposed hybrid and classical WaferMamba)
def per_class(kind):
    rep = []
    for seed in SEEDS:
        d = classification_report(Yte, RES[("lot", kind, seed)]["prob_te"].argmax(1), target_names=DEFECTS,
                                  output_dict=True, zero_division=0)
        rep += [{"class": c, "seed": seed, "precision": d[c]["precision"], "recall": d[c]["recall"],
                 "f1": d[c]["f1-score"], "support": d[c]["support"]} for c in DEFECTS]
    rep = pd.DataFrame(rep)
    pc = rep.groupby("class", sort=False)[["precision", "recall", "f1"]].agg(["mean", "std"]).round(4)
    pc["support"] = rep.groupby("class", sort=False)["support"].first().astype(int)
    return pc

def figures_for(kind, suffix):
    bs = max(SEEDS, key=lambda s: f1_score(Yva, RES[("lot", kind, s)]["prob_va"].argmax(1), average="macro"))
    prob = RES[("lot", kind, bs)]["prob_te"]
    cm = confusion_matrix(Yte, prob.argmax(1), labels=range(NCLS)); cmn = cm / cm.sum(1, keepdims=True)
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
    for ax, M, ttl, fmt in [(axs[0], cm, f"Counts, seed {bs}", "d"), (axs[1], cmn, f"Row-normalised, seed {bs}", ".2f")]:
        im = ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(NCLS), DEFECTS, rotation=45, ha="right"); ax.set_yticks(range(NCLS), DEFECTS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(ttl)
        for i in range(NCLS):
            for j in range(NCLS):
                ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > M.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(OUT/f"figures/fig_confusion_matrix_{suffix}.png", dpi=300, bbox_inches="tight"); plt.show()
    Yb = label_binarize(Yte, classes=range(NCLS))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, c in enumerate(DEFECTS):
        f_, t_, _ = roc_curve(Yb[:, i], prob[:, i]); ax.plot(f_, t_, label=f"{c} (AUC = {auc(f_, t_):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT/f"figures/fig_roc_{suffix}.png", dpi=300, bbox_inches="tight"); plt.show()
    return int(bs)

FIG_SEEDS = {}
for kind, suffix in [("QWaferMamba_E2E", "qwafer_mamba"), ("WaferMamba", "wafermamba")]:
    pc = per_class(kind); print(LABEL[kind]); display(pc)
    pc.to_csv(OUT/f"tables/table_per_class_{suffix}.csv")
    FIG_SEEDS[kind] = figures_for(kind, suffix)
print("Figures use the seed with the highest validation Macro-F1:", FIG_SEEDS)

# R1-5: per-seed confusion matrices and per-class metrics for every lot-aware network model
cm_rows, pc_rows = [], []
for (sp, kind, seed), r in RES.items():
    if sp != "lot": continue
    pred = r["prob_te"].argmax(1); cm = confusion_matrix(Yte, pred, labels=range(NCLS))
    for i in range(NCLS):
        cm_rows.append({"model": kind, "seed": seed, "actual": DEFECTS[i], **{DEFECTS[j]: int(cm[i, j]) for j in range(NCLS)}})
    d = classification_report(Yte, pred, target_names=DEFECTS, output_dict=True, zero_division=0)
    pc_rows += [{"model": kind, "seed": seed, "class": c, "precision": d[c]["precision"], "recall": d[c]["recall"],
                 "f1": d[c]["f1-score"], "support": int(d[c]["support"])} for c in DEFECTS]
pd.DataFrame(cm_rows).to_csv(OUT/"tables/table_confusion_per_seed.csv", index=False)
pcs = pd.DataFrame(pc_rows); pcs.to_csv(OUT/"tables/table_per_class_per_seed.csv", index=False)
display(pcs.pivot_table(index="class", columns="model", values="f1", aggfunc="mean", sort=False).round(3))

# CELL 18 — TRAINING CURVES (stage 1 and stage 2) + EFFICIENCY
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for name in MODELS:
    h = pd.DataFrame(RES[("lot", name, SEEDS[0])]["hist"])
    axs[0].plot(h.epoch, h.loss, label=name); axs[1].plot(h.epoch, h.val_macro_f1, label=name)
axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Training loss"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Validation Macro-F1")
for a in axs: a.legend()
fig.tight_layout(); fig.savefig(OUT/"figures/fig_training_curves_stage1.png", dpi=300, bbox_inches="tight"); plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for k in ["QWaferMamba_E2E", "WaferMamba_FT"]:
    h = pd.DataFrame(RES[("lot", k, SEEDS[0])]["hist"])
    axs[0].plot(h.epoch, h.loss, marker="o", label=LABEL[k]); axs[1].plot(h.epoch, h.val_macro_f1, marker="o", label=LABEL[k])
axs[0].set_xlabel("Stage-2 epoch"); axs[0].set_ylabel("Training loss")
axs[1].set_xlabel("Stage-2 epoch"); axs[1].set_ylabel("Validation Macro-F1")
for a in axs: a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT/"figures/fig_training_curves_stage2_quantum.png", dpi=300, bbox_inches="tight"); plt.show()

@torch.no_grad()
def bench(name, reps=100):
    m = BUILD[name]().to(DEVICE).eval()
    x1 = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE); xb = torch.zeros(256, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (lambda: None)
    for _ in range(10): m(x1)
    sync(); t = time.time()
    for _ in range(reps): m(x1)
    sync(); lat = (time.time() - t) / reps * 1e3
    for _ in range(3): m(xb)
    sync(); t = time.time()
    for _ in range(10): m(xb)
    sync(); thr = 2560 / (time.time() - t)
    return {"model": name, "params": nparams(m), "latency_ms_batch1": lat, "throughput_wafers_per_s": thr}
effic = pd.DataFrame([bench(n) for n in MODELS + ["QWaferMamba"]])
tm = pd.concat([pd.DataFrame([{"model": k[1], "train_min": float(r["train_sec"]) / 60, "best_epoch": int(r["best_epoch"])}
                              for k, r in RES.items() if k[0] == "lot"])]).groupby("model").mean().reset_index()
tm["model"] = tm["model"].replace({"QWaferMamba_E2E": "QWaferMamba"})
effic = effic.merge(tm, on="model", how="left")
display(effic.round(3)); effic.to_csv(OUT/"tables/table_efficiency.csv", index=False)
print("QWaferMamba train_min is stage 2 only (add the WaferMamba stage-1 time for the total).")

# CELL 19 — REPEATED LOT-DISJOINT PARTITIONS (R1-3, R2-2) AND MATCHED RANDOM vs LOT-AWARE CONTRAST (R1-11)
# Each extra partition trains WaferMamba and ResNet-18 (seed 42), then QWafer-Mamba and its same-schedule control.
MINORITY = ["Donut", "Near-full", "Scratch", "Random", "Loc"]
part_rows, proto_rows = [], []
s0 = SEEDS[0]
for lot_sp, rnd_sp, p in PARTITIONS:
    yte_p = Y[SPLITS[lot_sp][2]]
    runs_p = {}
    for name in PARTITION_MODELS:
        runs_p[name] = RES[("lot", name, s0)] if lot_sp == "lot" else run_one(lot_sp, name, s0)
    for kind in PARTITION_STAGE2:
        runs_p[kind] = RES[("lot", kind, s0)] if lot_sp == "lot" else finetune(kind, s0, split=lot_sp)
    for k, r in runs_p.items():
        if lot_sp != "lot": RES[(lot_sp, k, s0)] = r
        rep = classification_report(yte_p, r["prob_te"].argmax(1), target_names=DEFECTS, output_dict=True, zero_division=0)
        part_rows.append({"partition": p, "model": k, "test_lots": len(set(LOTS[SPLITS[lot_sp][2]])), **metrics(yte_p, r["prob_te"]),
                          **{f"recall_{c}": rep[c]["recall"] for c in MINORITY}})
    # matched random split for the leakage contrast (WaferMamba, same subset sizes)
    rr = run_one(rnd_sp, "WaferMamba", s0)
    m_l, m_r = metrics(yte_p, runs_p["WaferMamba"]["prob_te"]), metrics(Y[SPLITS[rnd_sp][2]], rr["prob_te"])
    proto_rows.append({"partition": p, "n_test": len(yte_p), "lot_macro_f1": m_l["macro_f1"], "random_macro_f1": m_r["macro_f1"],
                       "lot_accuracy": m_l["accuracy"], "random_accuracy": m_r["accuracy"],
                       "delta_macro_f1_random_minus_lot": m_r["macro_f1"] - m_l["macro_f1"]})
parts = pd.DataFrame(part_rows); parts.to_csv(OUT/"tables/table_partitions_per_run.csv", index=False)
cols_p = ["accuracy", "macro_f1", "mcc"] + [f"recall_{c}" for c in MINORITY]
psum = parts.groupby("model")[cols_p].agg(["mean", "std", "min", "max"])
psum.to_csv(OUT/"tables/table_partitions_mean_std.csv"); display((100 * psum).round(2))
wide = parts.pivot(index="partition", columns="model", values="macro_f1")
pdelta = pd.DataFrame({"QWafer_minus_FT": wide["QWaferMamba_E2E"] - wide["WaferMamba_FT"],
                       "QWafer_minus_ResNet18": wide["QWaferMamba_E2E"] - wide["ResNet18"],
                       "QWafer_minus_WaferMamba": wide["QWaferMamba_E2E"] - wide["WaferMamba"]})
pdelta.loc["mean"] = pdelta.mean(); pdelta.loc["std"] = pdelta.iloc[:-1].std()
pdelta.to_csv(OUT/"tables/table_partitions_deltas.csv"); display((100 * pdelta).round(2))
proto = pd.DataFrame(proto_rows); proto.loc["mean"] = proto.mean(numeric_only=True); proto.loc["std"] = proto.iloc[:-1].std(numeric_only=True)
proto.to_csv(OUT/"tables/table_split_protocol_partitions.csv"); display(proto.round(4))

# v3-compatible single-partition table (primary partition, WaferMamba and ResNet-18)
if RUN_RANDOM_SPLIT:
    Yte_r = Y[SPLITS["random"][2]]; rows = []
    for name in RANDOM_SPLIT_MODELS:
        r = run_one("random", name, SEEDS[0])
        rows.append({"model": name, "protocol": "random (stratified, size-matched)", **metrics(Yte_r, r["prob_te"])})
        rows.append({"model": name, "protocol": "lot-aware", **metrics(Yte, RES[("lot", name, SEEDS[0])]["prob_te"])})
    leak = pd.DataFrame(rows)
    display(leak.round(4)); leak.to_csv(OUT/"tables/table_split_protocol_comparison.csv", index=False)

# ## Finite shots and hardware noise (R1-2, R2-5)
# The trained quantum layer of each seed is evaluated in three ways on the test wafers: (i) exact expectation values (the reported results); (ii) finite-shot estimates obtained by sampling bitstrings from the exact output distribution, for 64 to 8192 shots and five repetitions; (iii) the transpiled circuit executed by Qiskit Aer with the gate, thermal-relaxation and readout noise models of two IBM fake backends (calibration snapshots of a Heron and an Eagle processor), 1024 shots, no error mitigation. The cell also reports transpiled depth, two-qubit gate count, estimated circuit duration, calibration errors of the physical qubits used, and a per-wafer latency breakdown. Communication and queue time of a cloud quantum service are not measured.

# CELL 20 — FINITE-SHOT AND NOISY-HARDWARE EVALUATION OF THE TRAINED QUANTUM LAYER
@torch.no_grad()
def qlayer_inputs(m, idx, bs=512):
    m.eval(); A = []
    for s in range(0, len(idx), bs):
        bi = torch.as_tensor(idx[s:s + bs], device=DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            A.append(m.angles(XG[bi].float()).float())
    return torch.cat(A)

def load_stage2(kind, seed, split="lot"):
    m = build_stage2(kind)
    m.load_state_dict(torch.load(RUNS/f"{split}_{kind}_s{seed}_e{EPOCHS}_ft{FT_EPOCHS}_w{FT_HEAD_WARMUP}_cw{FT_CW_POWER}.pt",
                                 map_location="cpu"))
    return m.to(DEVICE).eval()

def f1acc(y, pred): return f1_score(y, pred, average="macro"), accuracy_score(y, pred)

if RUN_NOISE:
    te = SPLITS["lot"][2]
    if NOISE_N is not None:
        sub = StratifiedShuffleSplit(n_splits=1, test_size=NOISE_N, random_state=0)
        sel = next(sub.split(te, Y[te]))[1]; te_n = te[sel]
    else:
        te_n = te
    y_n = Y[te_n]
    shot_rows, exact = [], {}
    for seed in SEEDS:
        m = load_stage2("QWaferMamba_E2E", seed)
        A = qlayer_inputs(m, te_n)
        with torch.no_grad():
            P = m.qlayer.probs(A.to(DEVICE)).float()                     # (N, 256) exact output distribution
            zs = m.qlayer.zsign.float(); s_, b_ = m.scale.detach().float(), m.bias.detach().float()
            pred_exact = (s_ * (P @ zs) + b_).argmax(1).cpu().numpy()
        exact[seed] = (A.cpu().numpy(), m.qlayer.theta.detach().cpu().numpy().astype(float),
                       float(s_), b_.cpu().numpy(), pred_exact)
        f1e, acce = f1acc(y_n, pred_exact)
        shot_rows.append({"seed": seed, "shots": "exact", "repeat": 0, "macro_f1": f1e, "accuracy": acce, "agreement_with_exact": 1.0})
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        for S_ in SHOTS_LIST:
            for rep in range(SHOT_REPEATS):
                preds = []
                for c0 in range(0, len(P), 512):
                    k = torch.multinomial(P[c0:c0 + 512], S_, replacement=True, generator=g)   # sampled bitstrings
                    ez = zs[k].mean(1)                                                         # estimated <Z_q>
                    preds.append((s_ * ez + b_).argmax(1))
                pr = torch.cat(preds).cpu().numpy(); f1s, accs = f1acc(y_n, pr)
                shot_rows.append({"seed": seed, "shots": S_, "repeat": rep, "macro_f1": f1s, "accuracy": accs,
                                  "agreement_with_exact": float((pr == pred_exact).mean())})
        del m; torch.cuda.empty_cache()
    shots = pd.DataFrame(shot_rows); shots.to_csv(OUT/"tables/table_shots_per_run.csv", index=False)
    shot_sum = shots.groupby("shots", sort=False)[["macro_f1", "accuracy", "agreement_with_exact"]].agg(["mean", "std"])
    shot_sum.to_csv(OUT/"tables/table_shots_mean_std.csv"); display((100 * shot_sum).round(2))

    # ---------- noisy hardware simulation with IBM fake backends ----------
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    import qiskit_ibm_runtime.fake_provider as FP
    qc_m = hq_qc.copy(); qc_m.measure_all()
    tr_rows, noise_rows, calib_rows = [], [], []
    for bname in FAKE_BACKENDS:
        try:
            backend = getattr(FP, bname)()
        except Exception as e:
            print("skip", bname, e); continue
        tqc = transpile(qc_m, backend=backend, optimization_level=3, seed_transpiler=0)
        ops = dict(tqc.count_ops()); twoq = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx", "rzz"))
        try:
            dur = float(tqc.estimate_duration(backend.target, unit="s"))
        except Exception:
            dur = float("nan")
        phys = tqc.layout.final_index_layout()[:HQ_QUBITS] if tqc.layout is not None else list(range(HQ_QUBITS))
        try:
            ro = [backend.target["measure"][(q,)].error for q in phys]
        except Exception:
            ro = [np.nan]
        tw = []
        for ins in tqc.data:
            if ins.operation.num_qubits == 2 and ins.operation.name != "barrier":
                qq = tuple(tqc.find_bit(b).index for b in ins.qubits)
                try: tw.append(backend.target[ins.operation.name][qq].error)
                except Exception: pass
        tr_rows.append({"backend": bname, "processor_qubits": backend.num_qubits, "depth": tqc.depth(), "two_qubit_gates": twoq,
                        "total_ops": int(sum(v for k, v in ops.items() if k not in ("barrier", "measure"))), "ops": json.dumps(ops),
                        "circuit_duration_us": dur * 1e6, "mean_readout_error": float(np.nanmean(ro)),
                        "mean_two_qubit_error": float(np.nanmean(tw)) if tw else np.nan,
                        "qpu_time_per_wafer_ms_1024_shots": NOISE_SHOTS * (dur * 1e6 + REP_DELAY_US) / 1e3 if np.isfinite(dur) else np.nan})
        sim = AerSimulator.from_backend(backend)
        for seed in SEEDS:
            A, th, s_, b_, pred_exact = exact[seed]
            bind_t = dict(zip(HTHETA, th))
            circs = [tqc.assign_parameters({**dict(zip(xh, a)), **bind_t}) for a in A]
            t0 = time.time(); EZ = np.zeros((len(circs), HQ_QUBITS))
            for c0 in range(0, len(circs), 500):
                res = sim.run(circs[c0:c0 + 500], shots=NOISE_SHOTS, seed_simulator=seed).result()
                for i in range(len(circs[c0:c0 + 500])):
                    for key, cnt in res.get_counts(i).items():
                        bits = key.replace(" ", "")[::-1]                   # bits[q] = outcome of logical qubit q
                        EZ[c0 + i] += cnt * (1 - 2 * np.array([int(bits[q]) for q in range(HQ_QUBITS)]))
            EZ /= NOISE_SHOTS
            pr = (s_ * EZ + b_).argmax(1); f1n, accn = f1acc(y_n, pr)
            noise_rows.append({"backend": bname, "seed": seed, "shots": NOISE_SHOTS, "macro_f1": f1n, "accuracy": accn,
                               "agreement_with_exact": float((pr == pred_exact).mean()),
                               "exact_macro_f1": f1acc(y_n, pred_exact)[0], "sim_minutes": (time.time() - t0) / 60})
            print(f"{bname} seed {seed}: noisy Macro-F1 = {f1n:.4f} (exact {noise_rows[-1]['exact_macro_f1']:.4f}), "
                  f"agreement {noise_rows[-1]['agreement_with_exact']:.3f}")
    transp = pd.DataFrame(tr_rows); transp.to_csv(OUT/"tables/table_transpile_resources.csv", index=False); display(transp)
    noise = pd.DataFrame(noise_rows); noise.to_csv(OUT/"tables/table_noise_per_seed.csv", index=False)
    if len(noise):
        nsum = noise.groupby("backend")[["macro_f1", "accuracy", "agreement_with_exact", "exact_macro_f1"]].agg(["mean", "std"])
        nsum.to_csv(OUT/"tables/table_noise_mean_std.csv"); display((100 * nsum).round(2))

    # ---------- latency breakdown per wafer ----------
    m = load_stage2("QWaferMamba_E2E", SEEDS[0]); sync = torch.cuda.synchronize if DEVICE.type == "cuda" else (lambda: None)
    x1 = XG[torch.as_tensor(te[:1], device=DEVICE)].float()
    with torch.no_grad():
        for _ in range(10): m(x1)
        sync(); t0 = time.time()
        for _ in range(100): h1 = encoder_latent(m.backbone, x1)
        sync(); t_enc = (time.time() - t0) / 100 * 1e3
        a1 = np.pi * torch.tanh(m.bridge(h1.float())); sync(); t0 = time.time()
        for _ in range(100): m.qlayer(a1)
        sync(); t_q = (time.time() - t0) / 100 * 1e3
    lat = [{"component": "SS2D encoder + bridge (GPU, batch 1)", "ms_per_wafer": t_enc},
           {"component": "quantum layer, exact statevector simulation (GPU, batch 1)", "ms_per_wafer": t_q}]
    for r_ in tr_rows:
        lat.append({"component": f"quantum layer on {r_['backend']}: {NOISE_SHOTS} shots x (circuit {r_['circuit_duration_us']:.1f} us + delay {REP_DELAY_US:.0f} us), estimate",
                    "ms_per_wafer": r_["qpu_time_per_wafer_ms_1024_shots"]})
    lat.append({"component": "compilation, job submission, queue and network transfer", "ms_per_wafer": np.nan})
    latdf = pd.DataFrame(lat); latdf.to_csv(OUT/"tables/table_latency_breakdown.csv", index=False); display(latdf)
    del m; torch.cuda.empty_cache()

# ## Statistical analysis (R1-7, R2-4)
# Unit of replication: one optimisation seed (random initialisation, data order and augmentation) on one fixed lot-disjoint partition; the five seeds share the same test wafers, so seed-level tests describe training randomness, not sampling of new lots (partition variability is reported separately in Cell 19). The primary hypothesis is QWafer-Mamba versus its same-schedule classical control on test Macro-F1 (paired t-test over seeds, α = 0.05). All other comparisons form a secondary family with Holm correction. Equivalence to a baseline is tested with two one-sided tests (TOST) against a margin of ±1 Macro-F1 point. A hierarchical bootstrap that resamples seeds and test wafers gives a 95% interval for the mean difference that includes both sources of uncertainty.

# CELL 21 — PRIMARY HYPOTHESIS, HOLM CORRECTION, TOST EQUIVALENCE, HIERARCHICAL BOOTSTRAP
from scipy.stats import t as tdist

def fast_f1(y, p):
    cm = np.bincount(y * NCLS + p, minlength=NCLS * NCLS).reshape(NCLS, NCLS).astype(float)
    tp = np.diag(cm); pr = np.divide(tp, cm.sum(0), out=np.zeros(NCLS), where=cm.sum(0) > 0)
    rc = np.divide(tp, cm.sum(1), out=np.zeros(NCLS), where=cm.sum(1) > 0)
    return float(np.mean(np.divide(2 * pr * rc, pr + rc, out=np.zeros(NCLS), where=(pr + rc) > 0)))

PRED = {k: np.stack([RES[("lot", k, s)]["prob_te"].argmax(1) for s in SEEDS]) for k in
        {kk[1] for kk in RES if kk[0] == "lot"}}
B_KIND = "QWaferMamba_E2E"
FAMILY = [("WaferMamba_FT", "primary")] + [(k, "secondary") for k in
          ["WaferMamba", "ResNet18", "CNN", "EfficientNetB0", "MobileViTXXS"] + HEAD_KINDS + ABL_KINDS if k in PRED]

def hier_boot(a, b, n=N_HBOOT, seed=0):
    rng = np.random.default_rng(seed); out = np.empty(n); S_, N_ = len(SEEDS), len(Yte)
    for i in range(n):
        ss = rng.integers(0, S_, S_); jj = rng.integers(0, N_, N_); y = Yte[jj]
        out[i] = np.mean([fast_f1(y, PRED[b][s][jj]) - fast_f1(y, PRED[a][s][jj]) for s in ss])
    return np.percentile(out, [2.5, 97.5])

rows = []
for a, fam in FAMILY:
    d = np.array([fast_f1(Yte, PRED[B_KIND][s]) - fast_f1(Yte, PRED[a][s]) for s in range(len(SEEDS))])
    n = len(d); mu, sd = d.mean(), d.std(ddof=1) if n > 1 else 0.0; se = sd / np.sqrt(n) if n > 1 else np.nan
    p = ttest_1samp(d, 0).pvalue if n > 1 else np.nan
    tq95, tq90 = (tdist.ppf(.975, n - 1), tdist.ppf(.95, n - 1)) if n > 1 else (np.nan, np.nan)
    p_tost = max(ttest_1samp(d, -EQUIV_MARGIN, alternative="greater").pvalue,
                 ttest_1samp(d, EQUIV_MARGIN, alternative="less").pvalue) if n > 1 else np.nan
    lo, hi = hier_boot(a, B_KIND)
    rows.append({"comparison": f"QWafer-Mamba − {a}", "family": fam, "mean_delta_macro_f1": mu, "sd_over_seeds": sd,
                 "ci95_low_t": mu - tq95 * se, "ci95_high_t": mu + tq95 * se, "ci90_low_t": mu - tq90 * se, "ci90_high_t": mu + tq90 * se,
                 "p_paired_t": p, "p_TOST_margin": p_tost, "equivalent_within_margin": bool(p_tost < .05) if n > 1 else None,
                 "hboot95_low": lo, "hboot95_high": hi})
st = pd.DataFrame(rows)
sec = st[st.family == "secondary"].sort_values("p_paired_t"); m_ = len(sec); adj, run = [], 0.0
for i, p in enumerate(sec.p_paired_t.values):                         # Holm step-down
    run = max(run, min(1.0, (m_ - i) * p)); adj.append(run)
st.loc[sec.index, "p_holm"] = adj
st.loc[st.family == "primary", "p_holm"] = st.loc[st.family == "primary", "p_paired_t"]
st.to_csv(OUT/"tables/table_stats_primary_holm_tost.csv", index=False)
show = st.copy()
for c in [c for c in st.columns if c.startswith(("mean", "sd", "ci", "hboot"))]: show[c] = 100 * show[c]
display(show.round(3))
print("Equivalence margin = ±%.1f Macro-F1 points; hierarchical bootstrap replicates = %d" % (100 * EQUIV_MARGIN, N_HBOOT))

# CELL 22 — PARAMETER ACCOUNTING: STORED vs ACTIVE (R1-4)
def accounting(kind):
    m = BUILD[kind]() if kind in BUILD and kind not in QKINDS else build_stage2(kind)
    stored = nparams(m)
    unused = nparams(m.backbone.head.cls) if hasattr(m, "backbone") else 0     # retained stage-1 classifier, not on the prediction path
    if hasattr(m, "backbone"):
        head = stored - nparams(m.backbone)
        dec = f"{head} (bridge {nparams(m.bridge)} + " + (f"θ {m.qlayer.theta.numel()} + scale/bias {1 + NCLS})" if kind in QKINDS
                                                         else f"output layer {nparams(m.out)})")
    elif hasattr(m, "head"):
        head = nparams(m.head.cls); dec = f"{head} (linear classifier)"
    trainable_q = m.qlayer.theta.numel() if kind in QKINDS and kind != "Q_Frozen" else 0
    return {"model": kind, "stored_params": stored, "unused_params": unused, "active_params": stored - unused,
            "decision_layer_params": dec, "trainable_quantum_theta": trainable_q}
kinds = MODELS + ["WaferMamba_FT"] + [k for k in STAGE2_KINDS if k not in ("QWaferMamba_E2E", "WaferMamba_FT")] + ["QWaferMamba_E2E"]
acc = pd.DataFrame([accounting("WaferMamba" if k == "WaferMamba_FT" else k) | {"model": k} for k in kinds])
acc.to_csv(OUT/"tables/table_param_accounting.csv", index=False); display(acc)
print("Convention: active = parameters used to produce a prediction; stored = all parameters in the checkpoint. "
      "Baselines have no unused parameters, so stored = active for them.")

# CELL 23 — EXPORT: summary.json, artefacts (R1-8), results ZIP and checkpoint ZIP
import subprocess, platform
# per-wafer test predictions of every run
prow = []
for (sp, kind, seed), r in RES.items():
    te_ = SPLITS[sp][2]; P = r["prob_te"]
    d = pd.DataFrame(P, columns=[f"p_{c}" for c in DEFECTS])
    d.insert(0, "pred", np.array(DEFECTS)[P.argmax(1)]); d.insert(0, "true", np.array(DEFECTS)[Y[te_]])
    d.insert(0, "wafer_index", te_); d.insert(0, "seed", seed); d.insert(0, "model", kind); d.insert(0, "protocol", sp)
    prow.append(d)
pd.concat(prow).to_csv(OUT/"artifact/test_predictions_all_runs.csv.gz", index=False, float_format="%.6f")
# training histories
hrow = [dict(protocol=sp, model=kind, seed=seed, **h) for (sp, kind, seed), r in RES.items() for h in r["hist"]]
pd.DataFrame(hrow).to_csv(OUT/"artifact/training_histories.csv", index=False)
# configuration and environment
CFG_KEYS = ["SEEDS", "EPOCHS", "PATIENCE", "WARMUP", "BATCH", "IMG_SIZE", "LATENT", "NCLS", "LR_MAIN", "WD", "LABEL_SMOOTH",
            "IMBALANCE", "MODELS", "RANDOM_SPLIT_MODELS", "QUBITS", "QDEPTH", "RES_SEED", "N_BOOT", "SPLIT_SEED", "HQ_QUBITS",
            "HQ_REPS", "FT_EPOCHS", "FT_PATIENCE", "FT_HEAD_WARMUP", "FT_CW_POWER", "LR_BACKBONE", "LR_QUANTUM", "VQC_QUBITS",
            "VQC_REPS", "VQC_PER_CLASS", "VQC_MAXITER", "HEAD_KINDS", "ABL_KINDS", "STAGE2_KINDS", "PARTITION_SEEDS",
            "PARTITION_MODELS", "PARTITION_STAGE2", "SHOTS_LIST", "SHOT_REPEATS", "NOISE_SHOTS", "NOISE_N", "FAKE_BACKENDS",
            "REP_DELAY_US", "EQUIV_MARGIN", "N_HBOOT", "DEFECTS"]
CONFIG = {k: globals()[k] for k in CFG_KEYS if k in globals()}
(OUT/"artifact/config.json").write_text(json.dumps(CONFIG, indent=2, default=str))
env = subprocess.run("pip freeze", shell=True, capture_output=True, text=True).stdout
(OUT/"artifact/environment.txt").write_text(f"python {platform.python_version()}\ntorch {torch.__version__}\n"
                                            f"cuda {torch.version.cuda}\ngpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}\n\n" + env)

summary = {
    "dataset": "WM-811K", "task": "8-class defect-pattern classification (none excluded)", "version": "v4 (IJIES revision)",
    "n": int(len(Y)), "split_sizes_lot": {k: int(len(v)) for k, v in zip(["train", "val", "test"], SPLITS["lot"])},
    "seeds": SEEDS, "epochs_stage1_max": EPOCHS, "imbalance": IMBALANCE, "partitions": [p for _, _, p in PARTITIONS],
    "main_results_mean": mean.round(5).to_dict(orient="index"),
    "main_results_std": std.round(5).to_dict(orient="index"),
    "quantum_verdicts": VERDICTS,
    "stats_primary_and_holm": st.round(6).to_dict(orient="records"),
    "quantum_layer": {"qubits": HQ_QUBITS, "encoding": "RY angle encoding", "ansatz": f"RealAmplitudes reps={HQ_REPS}, reverse_linear",
                      "trainable_theta": len(HTHETA), "readout": "<Z_q> per qubit, one qubit per class",
                      "optimizer": "AdamW, exact gradients (verified equal to parameter-shift)",
                      "stage2_epochs_max": FT_EPOCHS, "stage2_head_warmup_epochs": FT_HEAD_WARMUP, "stage2_class_weight_power": FT_CW_POWER,
                      "lr_backbone": LR_BACKBONE, "lr_quantum": LR_QUANTUM},
    "vqc_cobyla": {"qubits": VQC_QUBITS, "ansatz": f"RealAmplitudes reps={VQC_REPS}, reverse_linear",
                   "trainable_theta": len(THETA), "readout": "P(q0,q1,q2) -> 8 classes", "optimizer": "COBYLA",
                   "maxiter": VQC_MAXITER, "train_subset_per_class": VQC_PER_CLASS},
    "reservoir": {"qubits": QUBITS, "layers": QDEPTH, "outputs": int(OBS.shape[1]),
                  "pca_variance_kept": {str(k): v for k, v in pca_var.items()}},
    "figure_seeds": FIG_SEEDS,
}
(OUT/"summary.json").write_text(json.dumps(summary, indent=2, default=str))
zp = shutil.make_archive(str(ROOT/"QWafer_Mamba_v4_results"), "zip", OUT)
cz = shutil.make_archive(str(ROOT/"QWafer_Mamba_v4_checkpoints"), "zip", RUNS)
print(json.dumps(VERDICTS, indent=2))
print("Results ZIP (send this back):", zp, f"({os.path.getsize(zp)/1e6:.1f} MB)")
print("Checkpoint ZIP (keep in Drive, for the GitHub release):", cz, f"({os.path.getsize(cz)/1e6:.1f} MB)")
try:
    from google.colab import files; files.download(zp)
except Exception: pass

# ## What to send back
# Upload `QWafer_Mamba_v4_results.zip` (tables, figures, `artifact/` folder, summary.json; it downloads automatically at the end and is also saved in `MyDrive/QWafer_Mamba_v2`). Keep `QWafer_Mamba_v4_checkpoints.zip` in Drive: it holds every checkpoint and per-run prediction file and goes into the public code release requested by the editor and Reviewer 1 (comment 8).
# 
# **Interpretation rule:** a quantum component is claimed as a contribution only if it beats its classical control consistently across seeds (bootstrap CI above zero in most seeds and a significant paired test). Cell 15 prints the verdicts; Cell 21 adds the Holm-corrected family, the TOST equivalence test and the hierarchical bootstrap; Cells 12, 19 and 20 give the head controls, ablations, partitions and noise results.
