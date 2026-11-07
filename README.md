# QWafer-Mamba

Code and machine-readable results for the paper *QWafer-Mamba: A Compact Hybrid Quantum-Classical Network for Stable and Efficient Wafer Map Defect Recognition* (International Journal of Intelligent Engineering and Systems, revised version).

QWafer-Mamba couples a four-direction selective state-space encoder (WaferMamba) with an eight-qubit RealAmplitudes circuit that serves as the decision layer, trained end to end. The repository reproduces every table of the revised paper: five backbones, nine Stage 2 decision layers (linear, MLP, bounded tanh and trigonometric heads, the quantum layer and four circuit ablations), the quantum reservoir and COBYLA baselines, four lot-disjoint partitions, finite-shot and IBM noise-model evaluation, and the statistical analysis (primary paired test, Holm correction, TOST, hierarchical bootstrap).

## Contents

| File | Content |
|---|---|
| `QWafer_Mamba_WM811K_v4.ipynb` | Complete pipeline (Google Colab, T4 GPU): data preparation, lot-aware splits, all models, quantum circuits with Qiskit verification, training, evaluation, statistics and export |
| `qwafer_mamba_v4.py` | The same code exported as a plain Python script (the custom four-direction selective scan is in Cell 6, the quantum layer in Cell 11, statistics in Cells 15 and 21) |
| `split_assignments.csv.gz` | Wafer-level assignment (train, validation, test) with lot identifiers for all eight protocols (`lot`, `lotP101`, `lotP202`, `lotP303` and the size-matched random splits) |
| `test_predictions_all_runs.csv.gz` | Per-wafer test predictions and class probabilities of every run (protocol, model, seed) |
| `training_histories.csv` | Per-epoch training loss and validation metrics of every run |
| `table_*.csv` | All result tables of the paper (29 files), for example `table_main_results_mean_std.csv`, `table_stage2_heads_ablation_mean_std.csv`, `table_partitions_per_run.csv`, `table_shots_mean_std.csv`, `table_noise_mean_std.csv`, `table_transpile_resources.csv` and `table_stats_primary_holm_tost.csv` |
| `summary.json` | Summary of the main results and settings |
| `config.json`, `environment.txt` | Full configuration (seeds, epochs, learning rates, circuit settings) and exact package versions |

Per-seed checkpoints (`QWafer_Mamba_v4_checkpoints.zip`, about 670 MB) are attached to the GitHub release of this repository.

## Reproducing the results

1. Open `QWafer_Mamba_WM811K_v4.ipynb` in Google Colab and select a T4 GPU.
2. Run all cells. Cell 3 downloads WM-811K from Kaggle (`qingyi/wm811k-wafer-map`, a Kaggle API token is required); the dataset is not redistributed here and remains subject to its original terms.
3. Finished runs are saved to Google Drive and reloaded after a disconnect, so the run (about 11 h on a T4) can be completed over several sessions.
4. To reuse the released checkpoints, unzip them into `MyDrive/QWafer_Mamba_v2/runs_v4/`; all training cells then load the saved runs and only the evaluation is recomputed.

Seeds: 42, 7, 123, 2024, 31. Lot-partition seeds: 42 (primary), 101, 202, 303.

## Licence

Code: MIT licence. WM-811K: see the original data provider.
