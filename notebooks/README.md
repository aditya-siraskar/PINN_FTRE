# Notebooks (Google Colab, phase-wise)

Each notebook runs one stage of [`PLAN.md`](../PLAN.md) in Colab. Click a badge, switch to a **T4 GPU**
(Runtime → Change runtime type), and run the setup cell first.

**No Colab GPU left?** Every notebook also runs on **Kaggle** (free T4, 30 h/week): open it with
`https://kaggle.com/kernels/welcome?src=https://github.com/aditya-siraskar/PINN_FTRE/blob/main/notebooks/<name>.ipynb`,
set Settings → Accelerator **GPU T4 x2** and Internet **on**, then **Save Version → Save & Run All (Commit)**.
The setup cell detects Kaggle and works in `/kaggle/working`; download the outputs from the version's **Output** tab
(notebook 08 writes `stage7_outputs.zip`).

| # | Phase | Stage | Notebook | Status |
|---|---|---|---|---|
| 00 | all | setup + tests | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-siraskar/PINN_FTRE/blob/main/notebooks/00_setup_and_tests.ipynb) `00_setup_and_tests` | ready |
| 01 | 1 · validation | 2a ground truth (exact / FD / MC) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-siraskar/PINN_FTRE/blob/main/notebooks/01_phase1_stage2a_reference.ipynb) `01_phase1_stage2a_reference` | ready |
| 02 | 1 · validation | 2b 1-D PINN | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-siraskar/PINN_FTRE/blob/main/notebooks/02_phase1_stage2b_pinn1d.ipynb) `02_phase1_stage2b_pinn1d` | ready |
| 03 | 1 · validation | 2c ablations (3 seeds) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-siraskar/PINN_FTRE/blob/main/notebooks/03_phase1_stage2c_ablations.ipynb) `03_phase1_stage2c_ablations` | ready |
| 04 | 2 · multi-asset | 3 multi-asset ground truth | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-siraskar/PINN_FTRE/blob/main/notebooks/04_phase2_stage3_multiasset_reference.ipynb) `04_phase2_stage3_multiasset_reference` | ready |
| 05 | 2 · multi-asset | 4 d-dim PINN, scaling | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-siraskar/PINN_FTRE/blob/main/notebooks/05_phase2_stage4_pinn_nd.ipynb) `05_phase2_stage4_pinn_nd` | done (d = 10 reported as breakdown; not rerun) |
| 06 | 3 · real-time | 5 amortised, latency (headline) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-siraskar/PINN_FTRE/blob/main/notebooks/06_phase3_stage5_realtime_amortised.ipynb) `06_phase3_stage5_realtime_amortised` | done (~45 min on a T4) |
| 07 | 4 · empirical | 6 real data, backtests | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-siraskar/PINN_FTRE/blob/main/notebooks/07_phase4_stage6_real_data.ipynb) `07_phase4_stage6_real_data` | done (~23 min training per market on a T4) |
| 08 | 4 · empirical | 7 physics vs data, release | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aditya-siraskar/PINN_FTRE/blob/main/notebooks/08_phase4_stage7_ablation_release.ipynb) `08_phase4_stage7_ablation_release` | done (~2 h on a Kaggle T4) |

Run order: 00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08.

## How the notebooks work
- **Setup cell:** mounts Google Drive and clones the repo to `MyDrive/PINN_FTRE` (or `git pull` if it is already there), then runs `pip install -e .`. Figures (`figures/`) and results (`results/`) are written inside that folder, so they **survive Colab disconnects**. Set `USE_DRIVE = False` for a throwaway run.
- **Logic stays in the library.** Notebooks call `scripts/*.py` and `pinn_tailrisk/`, then show the figures with `show_figures(stage)` next to a pass/fail checklist. To change the method, edit the library, not the notebook.

## Syncing Colab results back to GitHub
Colab edits the Drive copy. To push from it, run in a cell:
```
!git -C "/content/drive/MyDrive/PINN_FTRE" add figures results/**/*.json results/**/*.csv
!git -C "/content/drive/MyDrive/PINN_FTRE" commit -m "Stage X results"
!git -C "/content/drive/MyDrive/PINN_FTRE" push https://<GITHUB_TOKEN>@github.com/aditya-siraskar/PINN_FTRE main
```
Use a GitHub personal access token, stored with Colab's 🔑 Secrets panel, never pasted into the notebook. Or download the `figures/` folder from Drive and commit it on your laptop.
