# Adaptive Audiovisual Integration in Recurrent Neural Networks

Code for the MSc dissertation of the same name (Joseph Zeitoun, Imperial I-X, September 2026).
Supervisors: Dr Marcus Ghosh (Imperial) and Dr Pip Coen (UCL).

Two mouse studies report different rules for combining sight and sound. In a detection task the
auditory channel dominates \[Song et al., 2017]; in a localisation task neither sense wins and the
two are weighted by reliability \[Coen et al., 2023]. This project asks whether one small gated
recurrent network, trained on synthetic versions of both tasks, can reproduce either strategy, and
what its internal dynamics look like when it does.

---

## Start here

**`Notebooks_v8/` contains the work described in the report.** Every figure, table and number in the
submitted dissertation comes from that folder. The earlier folders (`Notebook_v1` through
`Notebooks_v7`) are the development history, kept because `PROJECT_NOTEBOOK_LOG.md` walks through
them version by version and records what changed and why. Nothing in v1–v7 is cited in the report.

`PROJECT_NOTEBOOK_LOG.md` is the long-form companion to this repository: what each notebook does,
what changed between versions, the numbers each version produced, and where results were wrong and
how that was diagnosed.

---

## Setup

Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Everything runs on CPU. No GPU is needed and none was used. All seeds are fixed, so a clean run
reproduces the reported numbers.

---

## Running the notebooks

Run `01_generate_trials_v8.ipynb` first — it writes the trial datasets that every other notebook
loads. After that the notebooks are independent and can be run in any order, although the numbering
follows the order of the report.

| Notebook | What it does | Report |
|---|---|---|
| `01_generate_trials_v8` | Builds the twelve subtasks and writes `generated_trials_v8/` | §3.1, Fig. 1, Table 1 |
| `02_curriculum_experiments_v8` | Sequential vs interleaved training | §3.1, Fig. 2 |
| `03_hidden_unit_sweep_v8` | Accuracy across 1–8 hidden units | §3.2, Fig. 3 |
| `03b_rnn_vs_gru_sweep_v8` | GRU vs plain tanh RNN | §3.2, Fig. 3–4 |
| `04_interpretability_2node_v8` | Decision regions, trajectories, fixed points, input drive | §3.4, Fig. 7 |
| `05_statespace_analysis_v8` | State-space analysis of the two-unit GRU | §3.4 |
| `05_state_space_figures_v8` | Figure-quality versions of the same five plots | §3.4 |
| `06_masked_gru_v8` | Masked GRU (self-connections only) | §3.2 |
| `07_conflict_trained_v8` | Auditory-first localisation labels | §3.4 |
| `08_pca_dynamics_5unit_v8` | PCA state space, five-unit network | §3.4, Fig. 8 |
| `09_masked_sweep_v8` | Masked vs standard across hidden sizes | §3.2, Fig. 3 |
| `10_masked_conditions_v8` | Baseline, no-noise and conflict-trained conditions | §3.5, Fig. 12 |
| `11_pca_per_seed_v8` | PCA state space per seed | §3.4 |
| `12_noise_robustness_v8` | Test-noise sweep without retraining | §3.5, Fig. 13 |
| `13_task_aligned_space_v8` | **Task-aligned (where/whether) space** — the primary interpretability method | §3.4, Fig. 9–11, Table 2 |
| `14_training_noise_sweep_v8` | Training-noise sweep, 0 to 2.0 | §3.5, Fig. 14 |
| `15_noise_geometry_v8` | Task-space geometry against training noise | §3.5, Fig. 15 |

Two notebooks carry a `05` prefix. `05_statespace_analysis_v8` is the analysis; `05_state_space_figures_v8`
regenerates the same five plots at figure quality for the report.

Most notebooks have had their outputs cleared to keep the repository small. The figures they produce
are saved to disk and are committed — see below.

---

## Layout

```
Notebooks_v8/
  01–15_*.ipynb              analysis notebooks
  cell_Ag_curve.py           standalone cell: A(g) reliability-weighting curve (§3.3, Fig. 6)
  generated_trials_v8/       train.npz, test.npz written by notebook 01
  trained_models_v8*/        trained weights and sweep results (.pt, .pkl)
  task_space_v8/             task-aligned figures + decoder_grouped_cv_masked.csv
  interp_v8/, statespace_v8/, pca_v8/, pca_seeds_v8/,
  masked_v8/, masked_sweep_v8/, masked_conditions_v8/,
  conflict_trained_v8/, noise_robustness_v8/,
  noise_training_v8/, noise_geometry_v8/     figure outputs per notebook

Notebook_v1 … Notebooks_v7   development history (not cited in the report)
PROJECT_NOTEBOOK_LOG.md      full narrative log of the project
```

---

## Method summary

Each trial is 50 timesteps of 50 ms. The stimulus appears at steps 10–20; Gaussian noise of standard
deviation 0.1 runs throughout, and stimulus intensity is drawn uniformly from 1.0 to 3.0. Four input
channels — auditory left, auditory right, visual left, visual right — feed a single-layer GRU of 1 to
8 hidden units, and one linear readout produces four classes: no detection, detection, right, left.
There is no context cue, so the network must infer the task from the stimulus.

Twelve subtasks share that readout. The multisensory detection conflict is held out from training
entirely and used to test for auditory dominance. Localisation conflicts are trained with random
left/right labels, so any consistent conflict behaviour has to emerge from the non-conflict trials.

Training uses Adam at a learning rate of 1e-3, batch size 64, cross-entropy at every timestep,
50 epochs, and five seeds per condition. Accuracy is read from the final timestep.

Interpretability uses two methods. In two-unit networks the whole hidden state is a plane, so
decision regions, trajectories and fixed points are drawn directly; fixed points come from minimising
`q(h) = ½‖F(h) − h‖²` with L-BFGS-B and are classified from the Jacobian eigenvalues. Across seeds and
sizes, activity is projected onto task-defined axes — a `where` decoder (left vs right) and a
`whether` decoder (detected vs not) — and the angle between them measures how separately the two
tasks are coded.

---

## Reproducing the reported numbers

Run `01` first, then the notebook listed against the figure you want in the table above. Seeds are
fixed throughout. Where a result depends on initialisation, the notebooks report all five seeds
rather than a selected one — this matters, because several results in the report vary substantially
across seeds.

---

## Licence and attribution

Built with PyTorch, NumPy, SciPy, scikit-learn and Matplotlib, used under their own licences. The
behavioural targets come from published mouse studies conducted under their own ethical approvals;
no animals were used and no new biological data were collected here. All trials in this repository
are synthetic.

### References

Coen, P., Sit, T.P.H., Wells, M.J., Carandini, M. and Harris, K.D. (2023) Mouse frontal cortex
mediates additive multisensory decisions. *Neuron*, 111(15), pp. 2432–2447.

Song, Y.-H., Kim, J.-H., Jeong, H.-W., Choi, I., Jeong, D., Kim, K. and Lee, S.-H. (2017) A neural
circuit for auditory dominance over visual perception. *Neuron*, 93(4), pp. 940–954.
