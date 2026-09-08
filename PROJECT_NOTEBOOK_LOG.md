# Project Notebook Log — Adaptive Audiovisual Integration in Recurrent Neural Networks

**Author:** Joseph  
**Supervisors:** Dr Marcus Ghosh (Imperial), Dr Pip Coen (UCL)

**What this is:** a guide to the code in this repository. It walks through every notebook, version by
version, and records what each one does, what changed between versions and why, the numbers each version
produced, and where results were near chance, surprising or wrong and how that was diagnosed. Read it
alongside the notebooks: sections 1 to 7 follow the versions in order, and sections 8 to 12 summarise the
arc, the recurring methodological themes, the known caveats, the figures and the notebooks themselves.

> **Note on the numbers:** every numeric value below is quoted directly from the notebook outputs.

---

## 0. The scientific question and the two reference paradigms

The project models **multisensory (audiovisual) integration** in mice using small recurrent neural networks, drawing on two experimental paradigms:

- **Detection task — Song et al. (2017):** is a stimulus present? In the mouse literature the animal is **auditory-driven** (auditory presence signals "detected"). This is built into the task as the project matures: visual-only stimuli come to count as *not detected*.
- **Localisation task — Coen et al. (2023):** is the stimulus on the left or the right? Conflict trials (audio one side, visual the other) probe **reliability-weighted integration** — the animal tends to follow whichever cue is stronger, and Coen's own accumulator model treats conflict trials with a random-reward design.

The overarching plan across versions: (1) build synthetic trial datasets for the two tasks; (2) train RNNs and confirm they solve them; (3) unify both tasks in one network; (4) probe how the network resolves **conflict**; (5) open the trained network as a **dynamical system** to explain *how* it integrates the two modalities.

**The 4-class label scheme (stabilises from v2 onward):** `0 = no detection`, `1 = detection`, `2 = right`, `3 = left`.
**Fixed stimulus geometry (constant across all versions):** 2.5 s trials, 50 timesteps of 50 ms; stimulus on from 0.5–1.0 s (timesteps 10–20); 4 input channels `[aud_L, aud_R, vis_L, vis_R]`; baseline Gaussian noise std 0.1; stimulus intensity drawn uniformly from [1.0, 3.0]. The network must **hold its answer from the stimulus window to a readout at the final timestep — a ~30-step memory delay.** Loss is applied at every timestep (label broadcast across time); the prediction is read at the last timestep.

---

## 1. Version 1 — first datasets and "can it even learn?"

**Folder:** `Notebook_v1/` — `01_generate_trials.ipynb`, `02_train_rnns.ipynb`
**Framing:** Phase 1 (generate two *separate* single-task datasets) + Phase 2 (train two *separate* single-task RNNs). Unifying the tasks is explicitly deferred.

### 1.1 `01_generate_trials` — two separate datasets
- 3000 trials per task, 70/30 train/test split, seed 42.
- **Detection** (binary present/absent): 4 trial types sampled by probability. Realised counts: absent 741, auditory_only 743, visual_only 747, multisensory 769 → **labels absent 741 / present 2259 (~25% / 75%)**. This imbalance matters below.
- **Localisation** (binary left/right): 8 types including two **conflict** types. Ground-truth label rule = **"stronger modality wins"** (the modality with the higher sampled intensity determines the side). Realised labels roughly balanced (left 1484 / right 1516).

**Design doubts recorded at the time:** whether 50 ms timesteps is the right resolution; whether the [1.0, 3.0] intensity range gives enough variation to elicit reliability weighting; and whether "stronger modality wins" is even the right ground truth for conflict trials, or whether a reliability-weighted scheme is needed.

### 1.2 `02_train_rnns` — the tasks are (almost) trivially solvable
- Model `TinyGRU`: `nn.GRU(4, 8)` → `nn.Linear(8, 1)`; Adam lr 1e-3, 50 epochs, seed 42.
- **Detection final test acc = 1.000; Localisation final test acc = 0.997.** Per-type detection = 1.000 everywhere; localisation = 1.000 on all non-conflict types, with conflict types at **0.979 / 0.990**.
- **Conflict behaviour analysis:** on conflict trials the network followed audio vs visual roughly 50/50 (`audL_visR`: 53.6% audio / 46.4% visual; `audR_visL`: 49.0% / 51.0%). Read as consistent with **reliability-weighted integration** rather than a fixed dominance — the intended "Coen-like" signature.

> **Why it matters.** Two lessons seeded the whole project here. (1) The detection network sat at exactly **75% for the first ~10 epochs** — it had learned the majority class ("always say present") because the labels were 75% present. That is the classic *majority-class trap*, and it is the direct motivation for the exact class-balancing introduced in v2. (2) **1.000 and 0.997 accuracy means the tasks are almost trivial** for even an 8-unit GRU. If everything is solved perfectly, there is no interesting integration behaviour to study. This "it's too easy" realisation is what pushes every later version toward *harder, more diagnostic* task designs and toward studying **conflict trials** — the one place the network's behaviour is not pinned by the labels.

---

## 2. Version 2 — exact balance, held-out conflict, and the first unified network

**Folder:** `Notebooks_v2/` — `01_generate_trials_v2.ipynb`, `02_train_rnns_v2.ipynb`

### 2.1 What changed from v1 (the fixes)
- **Exact, deterministic class balance** (counts, not probabilities). Detection now **50/50** absent/present — directly fixing the v1 majority-class problem.
- **Two localisation variants:** `localisation_with_conflict` (conflicts in train+test, as v1) vs `localisation_no_train_conflict` (conflicts **test-only**) — the latter closer to how Coen et al. analysed mice (conflict trials never deterministically rewarded).
- **Conflict left/right forced 50/50** (no longer left to random intensity draws).
- **First `unified_task` dataset** with the four-class label scheme, conflicts held out for test.

### 2.2 Results — the first genuinely hard regime appears
Hidden size 8, 50 epochs, seed 42. Headline final test accuracies:

| Network | Final test acc |
|---|---|
| Detection | **1.000** |
| Localisation (conflict in train) | **0.996** |
| Localisation (no train conflict) | **0.947** |
| Unified (4-class) | **0.762** |

- **Localisation B (conflicts test-only)** shows a train/test gap: **train 1.000, test 0.947**, with test loss *rising* over training — mild overfitting / a generalisation gap on the held-out conflict trials.
- **Unified** overfits clearly: **train 1.000 but test 0.762**, test loss climbing from ~0.95 to ~1.21. The 4-class unified task is the first non-trivial problem.
- **Conflict comparison A vs B:** the conflict-trained net splits ~50/50 audio/visual; the non-conflict-trained net leans slightly audio (0.61 audio on `audL_visR`). First evidence that *what the network is trained on changes its integration strategy.*

### 2.3 Hidden-unit sweep (1–8 units, single seed)

| h | Detection | Localisation | Unified |
|---|---|---|---|
| 1 | 1.000 | 0.929 | **0.463** |
| 2 | 1.000 | 0.936 | **0.907** |
| 3 | 1.000 | 0.928 | **0.573** |
| 4 | 1.000 | 0.947 | 0.750 |
| 5–8 | 1.000 | 0.936–0.964 | 0.757–0.799 |

> **Why it matters.** Detection is perfect at **every** size, even **1 hidden unit** — confirming it is essentially trivial. The unified curve is wildly **non-monotonic** (0.463 → 0.907 → 0.573 across h = 1, 2, 3). A capacity curve should not zig-zag like that. The correct reading is **not** "2 units is optimal"; it is "a **single seed is too noisy to trust** — these swings are optimisation luck, not capacity." That diagnosis is exactly what motivates the **multi-seed sweep** introduced in v4. (Also flagged: a doc/code mismatch — the summary text says "3000 trials" for the unified set while the code produces 3300.)

---

## 3. Version 3 — curriculum learning, and the discovery of pattern-matching + catastrophic forgetting

**Folder:** `Notebooks_v3/` — `01_generate_trials_v3.ipynb`, `02_curriculum_experiments_v3.ipynb`

### 3.1 Data changes
- **Separate, independently generated train and test sets** (no split) → guaranteed balance, no sampling bias. Test = exactly 100 trials × 12 types = 1200. Train balanced across **non-conflict types only**; **conflict trials excluded from training** (test-only probe), following Song/Coen (conflict never deterministically rewarded).

### 3.2 The experiment: three curricula
"Curriculum" here = the **order** subtasks are presented, at fixed total budget: **Loc→Det** (25 epochs each), **Det→Loc**, and **Mixed** (50 epochs interleaved).

| Curriculum | Overall | Non-conflict | Conflict |
|---|---|---|---|
| Loc→Det | 0.475 | 0.556 | 0.070 |
| Det→Loc | 0.623 | 0.593 | **0.775** |
| Mixed | 0.833 | **1.000** | **0.000** |

> **Why it matters — two headline diagnoses.**
> **(1) Pattern-matching, not integration.** The Mixed network scores **100% on non-conflict but 0% on conflict**. It has learned the closest *surface pattern*: a conflict trial has all-bilateral input, which looks like a detection trial, so it labels conflicts "detection." The notebook states it plainly: "the network has learned pattern-matching rather than integration." This is the single most important negative result of the project — it reframes the whole question from "can it score high?" to "can we *force* it to integrate rather than memorise surface statistics?"
> **(2) Catastrophic forgetting.** Both sequential curricula collapse at the phase boundary: whichever task is trained *second* overwrites the first (loss spikes to ~5 at each phase-2 start; the phase-1 subtasks fall to 0.000). Only the task trained last survives. Below-chance conflict numbers (e.g. 0.070) are read correctly as "a confident-but-wrong rule," i.e. an artefact of collapse, not integration.
> The one intriguing signal: **Det→Loc reaches 0.775 on conflicts it never saw** — the one condition where above-chance integration *emerges*. This is what makes the "add conflicts to training" idea (v4) worth trying.

---

## 4. Version 4 — the key intervention (random-label conflict training) + the first proper multi-seed sweep

**Folder:** `Notebooks_v4/` — `01_generate_trials_v4.ipynb`, `02_curriculum_experiments_v4.ipynb`, `03_hidden_unit_sweep_v4.ipynb`

### 4.1 The intervention: conflicts in training with RANDOM labels (Coen 2023 style)
Conflict trials are now **in the training set, labelled 50/50 left/right at random**, uncorrelated with which modality is stronger. The reasoning: this means the network **cannot** learn "conflicts are detection" (they're labelled L/R) *and* **cannot** learn "follow the stronger modality" (labels are random w.r.t. intensity). "Whatever integration behaviour emerges on the test set is therefore something the network **discovered, not something it was told.**" Test labels keep the deterministic "stronger-wins" rule. Train is now 4800 trials, perfectly balanced 1200/1200/1200/1200.

### 4.2 Curriculum results — the intervention works (for Mixed)

| Curriculum | Overall | Non-conflict | Conflict |
|---|---|---|---|
| Loc→Det | 0.349 | 0.419 | **0.000** |
| Det→Loc | 0.635 | 0.598 | **0.820** |
| Mixed | **0.927** | **1.000** | **0.560** |

> **Why it matters.** The central intervention succeeded: Mixed-condition conflict accuracy rose **0.000 (v3) → 0.560 (v4)** while non-conflict stayed at 1.000. Because conflict labels were random in training, a test-set conflict score of 0.560 (above the 0.5 binary chance) can only mean the network **discovered reliability weighting on its own** — the cleanest possible evidence of genuine integration. Sequential curricula still collapse (Loc→Det actually worsens to 0.349) — catastrophic forgetting is not a tuning issue, it is intrinsic to sequential training. This cements **unified/mixed training as the only viable route** for the rest of the project.

### 4.3 First multi-seed hidden-unit sweep (1–8 units × 5 seeds × 3 tasks = 120 nets)

| h | Detection | Localisation | Unified |
|---|---|---|---|
| 1 | 1.000 ± 0.000 | 0.945 ± 0.038 | **0.430 ± 0.078** |
| 2 | 1.000 | 0.923 | 0.821 ± 0.090 |
| 3 | 1.000 | 0.917 | 0.870 ± 0.108 |
| 4 | 1.000 | 0.898 | 0.943 ± 0.023 |
| 5 | 1.000 | 0.901 | **0.950 ± 0.023 (peak)** |
| 6–8 | 1.000 | 0.883–0.890 | 0.938–0.947 |

> **Why it matters.** With error bars the picture finally makes sense where v2's single seed was chaos: **variance is huge at low capacity and collapses at high capacity** (unified std 0.108 at h=3 → ~0.012 at h≥6). The unified task needs **≥4 units** to saturate (~0.94–0.95); the chosen size 8 is comfortably in the safe regime. The remaining zig-zag is traced to **specific outlier seeds** (e.g. unified h=3 has one seed at 0.654 among ~0.92s) — i.e. *identifiable* bad-luck runs, not a real capacity dip. This is the moment the project learns to separate "signal" from "seed luck" quantitatively.
> **Engineering insight (also flagged):** a benchmark found **CPU 5.8 s vs MPS 113.6 s** — the Apple GPU was ~20× *slower* for these tiny GRUs, blowing the sweep out to ~19.6 hours. Later notebooks force CPU. A nice example of empirically checking an assumption ("GPU = faster") instead of trusting it.

---

## 5. Version 5 — sharpening the task to *demand* auditory dominance

**Folder:** `Notebooks_v5/` — `01_generate_trials_v5.ipynb`, `02_curriculum_experiments_v5.ipynb`, `03_hidden_unit_sweep_v5.ipynb`

### 5.1 Data changes (the corrected trial table)
Two science-changing edits that bake the biology into the labels:
- **`det_visual_only` → class 0 (no detection).** Visual-only now counts as *not detected*, making detection an **auditory** detection task (Song et al.). Auditory dominance is now part of the task definition, not just hoped for.
- **`det_multisensory` (all four channels) becomes a test-only conflict with a random 0/1 label.** It is the Song-style detection conflict (audio says "detected", vision says "not"). Its nominal label is 1, so **accuracy on this row reads directly as the auditory-dominance rate.**

Localisation conflict design unchanged (random labels in train, stronger-wins at test). Training balanced to 1200/class (4800 total); test 100/type (1200).

### 5.2 Curriculum results (HIDDEN_SIZE = 5)

| Curriculum | Overall | Mean non-conflict subtask | Auditory-dominance rate |
|---|---|---|---|
| Loc→Det | 0.250 (**= chance**) | 0.333 | 0.000 |
| Det→Loc | 0.340 | 0.342 | 0.000 |
| Unified | **0.876** | **0.999** | 0.000 |

> **Why it matters.** Unified is clean and strong (0.999 on trained subtasks). But the intended headline readout — **auditory dominance on `det_multisensory` — is exactly 0.000 in all three curricula.** The network *never* calls the matched audio+visual trial "detected"; it is completely **visual-dominant** (says "no detection"), the *opposite* of the hypothesis the task was redesigned to demonstrate. This is a genuine, surprising null result and a major open question — and it is the direct motivation for v6/v7's `07_conflict_trained` experiment, which *teaches* the dominance rule explicitly to see whether the architecture even can represent it.
> The two sequential curricula collapse to chance (0.250, 0.340) — forgetting again.

### 5.3 Hidden-unit sweep (unified only, 1–8 × 5 seeds)

| h | Overall | Det group | Loc group |
|---|---|---|---|
| 1 | 0.528 ± 0.092 | 0.419 | 0.582 |
| 2 | 0.765 ± 0.095 | 0.900 | 0.698 |
| 3 | **0.859 ± 0.028 (elbow)** | 0.802 | 0.887 |
| 4–8 | 0.850–0.875 | 0.750–0.793 | 0.900–0.916 |

Recommendation printed: **HIDDEN_SIZE = 6** (smallest within 1% of best mean 0.874).

> **Why it matters.** The **elbow is at 3 units** — the minimum for stable good performance; beyond that it is a flat plateau within noise. Two subtleties worth noting: (i) the "detection group" ceiling of **~0.75 is an artefact** — that 4-item group contains the never-correct test-only `det_multisensory` conflict, so ~3/4 correct is the maximum possible, not a detection failure; (ii) the recommendation says **6** but notebook 02 was actually run at **5** — a discrepancy worth noting (both are on the plateau, so it barely affects results, but honesty about it is the point). Also a leftover "set HIDDEN_SIZE in *02...v5*" message referencing the wrong version.

---

## 6. Version 6/7 — the definitive dataset, model choice, and the dynamical-systems payoff

**Folder:** `Notebooks_v6/` (notebooks numbered 01–08; the 06/07/08 files carry a `_v7` suffix and their outputs also live in `Notebooks_v7/` subfolders). This is the mature phase: the task design is frozen and the effort shifts to **model selection** and **mechanistic interpretability**.

### 6.1 `01_generate_trials_v6` — freeze the task, add intensity logging
Identical task design to v5, with **one addition**: the test set now records the two per-trial intensities on every localisation-conflict trial (`aud_int`, `vis_int`), so conflict accuracy can be broken down by **how much the stronger cue wins by**. This one logging change is what enables the reliability-weighting ("intensity-gap") analyses later. The 12-subtask table, balancing (1200/class, 4800 train; 1200 test) and label rules are exactly as in v5.

### 6.2 `03_hidden_unit_sweep_v6` — reconfirm capacity
Unified GRU, 1–8 units × 5 seeds. Same curve as v5: 0.528 (h1) → 0.765 (h2) → **0.859 (h3, the knee)** → plateau ~0.85–0.87. Recommends 6; NB02 uses 5, the same discrepancy carried over from v5. Detection-group ~0.75 plateau is again the `det_multisensory` artefact, not a failure.

### 6.3 `03b_rnn_vs_gru_sweep_v6` — should we switch to a plain RNN for interpretability?
**Motivation:** a vanilla (Elman) RNN has no gates, so its dynamics are far easier to analyse. If a plain RNN trains as well as the GRU, it is the better choice for the interpretability work. Identical setup, `cell ∈ {GRU, RNN(tanh)}`, 1–8 units × 5 seeds = 80 nets.

| h | GRU mean | RNN mean | gap |
|---|---|---|---|
| 1 | 0.528 | 0.298 | 0.230 |
| 2 | 0.765 | 0.377 | **0.388** |
| 3 | 0.859 | 0.687 | 0.172 |
| 4 | 0.855 | 0.773 | 0.082 |
| 5 | 0.853 | 0.801 | 0.052 |
| 7 | 0.850 | 0.836 | 0.014 |
| 8 | 0.859 | 0.774 | 0.084 |

Verdict (printed): plateau gap (h≥3) GRU 0.858 vs RNN 0.778 = **0.080**; RNN only reaches GRU-plateau at **7 units**, with much higher seed variance (std up to 0.193 at h=3) and non-monotonic behaviour (peaks at h=7, drops at h=8). **"RNN lags noticeably → not a straight swap for the GRU."**

> **Why it matters.** This is the reason the **GRU is kept and the 2-unit interpretability is done on the GRU** — the plain RNN's ~30-step memory demand is too much for it at small sizes (exactly the a-priori worry, confirmed empirically). A clean example of testing a convenient hypothesis ("RNN is good enough, and easier to analyse") and letting the data veto it. The gap lands *exactly* on the 0.080 threshold separating "close enough" from "not a straight swap" — an honest, borderline call rather than a forced one.

### 6.4 `02_curriculum_experiments_v6` — the main behavioural result (5 seeds, HIDDEN_SIZE = 5)
Now run across 5 seeds with mean ± std bands and richer conflict readouts.

**Overall test accuracy:** Loc→Det 0.281 ± 0.097; Det→Loc 0.379 ± 0.102; **Unified 0.853 ± 0.019.**

**Separated readouts (the important table):**

| Curriculum | Non-conflict acc | Loc-conflict (stronger-side agreement) | Det multisensory (aud-dominance) |
|---|---|---|---|
| Loc→Det | 0.313 ± 0.086 | 0.010 ± 0.020 | 0.534 ± **0.452** |
| Det→Loc | 0.384 ± 0.110 | 0.549 ± 0.118 | 0.000 ± 0.000 |
| **Unified** | **1.000 ± 0.000** | **0.620 ± 0.112** | **0.000 ± 0.000** |

> **Why it matters — how to read every one of these numbers.**
> - **Unified non-conflict = 1.000:** perfect on the 9 subtasks it was actually trained to do. This is the "performance" number.
> - **Unified loc-conflict = 0.620 (> 0.5):** the network agrees with the stronger cue on conflict trials **more often than chance, despite conflict labels being random in training** → reliability weighting, discovered not taught. This is the positive scientific result. The intensity-gap breakdown makes it mechanistic: accuracy **rises as the intensity gap grows** (easy when one cue clearly dominates, near-chance on near-ties) — the visual signature of reliability weighting.
> - **Unified det-multisensory aud-dominance = 0.000:** firmly **visual-dominant** on the detection conflict — the persistent, reproducible null from v5, now confirmed across 5 seeds with zero variance. A real property of the network, not noise.
> - **The sequential curricula are artefacts:** they collapse (0.28, 0.38 ≈ chance), so their conflict bars are "a **forgetting artefact, not an integration strategy**." Two clear illustrations of luck versus signal: **loc→det loc-conflict = 0.010** is *below* chance (a collapsed net systematically predicting the wrong side — anti-signal); and **loc→det det-multisensory = 0.534 ± 0.452** has a std so enormous it is essentially **bimodal across seeds** (some seeds all-detect, some all-not) — pure symmetry-breaking, i.e. coin-flip luck, which is exactly why it is dismissed rather than reported as "~0.5 dominance." A per-seed right/left-split figure was added specifically to check whether any side preference "is real or just symmetry-breaking."

### 6.5 `04_interpretability_2node_v6` — opening the 2-unit network as a dynamical system
With only **2 hidden units**, the entire network state is a point in a plane, so the computation can be drawn directly (no dimensionality reduction). Toolkit and citations: **fixed points + Jacobian-eigenvalue stability (Sussillo & Barak 2013); phase-portrait / low-D flow view (Mante et al. 2013); tiny-RNN dynamical regression (Ji-An et al. 2025); seed-convergence of dynamical structure (Maheswaranathan et al. 2019).**

- Trained seeds 0–5, kept the best by non-conflict accuracy (a 2-unit net is init-sensitive: seeds 0/4/5 reach ≥0.99, seeds 2/3 stall at ~0.66). Hand-coded numpy GRU verified against PyTorch to `2.8e-07`.
- **Baseline (zero-input) fixed points:** a **stable attractor** at h=[-0.959, -0.858] (|eig| 0.792, 0.560) and a **saddle** at h=[-0.900, 0.129] (|eig| 0.912, 1.105) — **both in the "no det" region**. The resting/no-detection memory state, with the saddle as the boundary the stimulus must cross.
- **Effective input-drive vectors** (one-step displacement per channel): at the origin, aud_L pushes hardest (|dh| = 1.089), the "L" channels push +unit-1 and "R" channels push −unit-1 → **unit 1 is essentially the left/right localisation axis**; auditory has larger gain than visual (aud_L 1.089 vs vis_L 0.720). Once the state has committed to a side, **same-side input saturates** (e.g. aud_L |dh| = 0.011 at the aud-L attractor) while the **opposing cue keeps strong leverage** (aud_L |dh| = 2.003 at the visual-R attractor).

> **Why it matters.** This is the mechanistic heart of the report. Integration is **additive at the origin** (both modalities push the same localisation axis, auditory with more gain) and becomes **winner-take-all via attractor dynamics** — a committed decision resists same-direction input but remains pushable by the competing cue, which is precisely how conflict gets resolved by relative intensity (reliability weighting). A subtle but important caveat: at zero input **only "no det" has its own attractor** — the det/right/left decisions are input-conditioned/transient and relax back toward rest once the stimulus ends, so "memory" here is more nuanced than "one attractor per class."

### 6.6 `05_statespace_analysis_v6` — averaged trajectories and cross-seed structure
Same model, all 5 seeds kept. Five figures in the 2D plane: (1) average trajectory per subtask (the "Figure 1" style), (2) conflict trajectories split by output label, (3) loc-conflict trajectories coloured by intensity gap, (3b) average conflict trajectory per **signed** gap band, (4) the same conflict subtask averaged across seeds, (5) the 5 seeds' readout decision boundaries overlaid.

> **Why it matters.** Two points on honest reporting. (i) **Figure 3/3b operationalise reliability weighting visually** — "a systematic shift of the endpoint with the intensity gap is the reliability-weighting effect made visual." (ii) The **cross-seed caveat (Maheswaranathan et al.):** each network's two hidden axes are internal, so different seeds may represent the *same* computation in **rotated/reflected coordinates** — "tight agreement is strong evidence of shared structure; apparent disagreement may be a change of coordinates, not a real difference." This is why the long-term plan is to compare **dynamical structure, not raw axes**, once the network scales beyond two units — and it directly sets up the PCA approach in NB08. (Note for interpretation: figures 4/5 include the weak seeds 2/3, so some of the visible spread is genuine training failure, not just coordinate change.)

### 6.7 `06_masked_gru_v7` — "can two units that can't talk still integrate?"
A hand-written 2-unit GRU whose hidden-to-hidden matrix is multiplied by a fixed binary mask on every forward pass, **keeping self-connections and zeroing unit↔unit connections in all three gates** (mask = 2×2 identity tiled three times, because PyTorch stacks reset/update/candidate blocks). Verified: max masked off-diagonal = **0.0e+00** every seed; numpy update matches torch to < 1e-4.

- Per-seed non-conflict: seed 0 **1.000**, seed 3 0.999, seed 4 0.998 — but **seed 1 0.746, seed 2 0.687**.
- Baseline dynamics: **exactly one** stable attractor at h=[-0.644, -0.896] in "no det" (|eig| 0.964, 0.677) — a much simpler skeleton than the fully-connected net.

> **Why it matters.** The answer is **yes, but fragile.** Two units that literally cannot exchange information still solve the task on the good seeds (seed 0 is perfect on all 9 non-conflict subtasks) — integration is done **per-unit via shared inputs plus the linear readout**, not via recurrent cross-talk. But removing cross-unit communication makes training **bimodally unreliable** (2 of 5 seeds collapse to ~0.69–0.75). Reporting this cleanly depends on the "keep all seeds, no quality selection" policy — cherry-picking the good seed would have hidden the real cost of the constraint.

### 6.8 `07_conflict_trained_v7` — teaching auditory dominance
A controlled experiment: **change exactly one thing** — relabel the 600 loc-conflict *training* trials so the **audio side always wins** (`audL_visR → left`, `audR_visL → right`); everything else identical. Standard 2-unit GRU.

- **Audio-follow rates ≈ 0.99–1.00** on 4 of 5 seeds, with non-conflict accuracy still ~0.97–1.0 → the dominance rule is **learned cleanly at no cost** to other tasks (echoing Song et al. mouse auditory dominance).
- Dynamics **enrich dramatically**: from one attractor to **three stable attractors** (no-det, right, left corners) + **two saddles** acting as decision separatrices that route conflict trials to the audio side.

> **Why it matters.** This closes the loop on the v5/v6 null result: the unified network was visual-dominant (aud-dominance 0.000), and this experiment proves the **architecture can represent auditory dominance perfectly well when taught** — so the null is about *what the task rewards*, not a capacity limit. Mechanistically it shows a learned behavioural rule literally **reshaping the state-space** (1 → 3 attractors + saddles). The honest outlier: **seed 2 follows audio 0.99 on one conflict direction but only 0.04 on the other** — an asymmetric single-seed breakdown, again surfaced only because all seeds are kept.

### 6.9 `08_pca_dynamics_5unit_v7` — scaling up and visualising honestly
At 5 hidden units the state is no longer directly plottable, so: run the test set, record hidden activity (neurons × time × trials), fit PCA on the pooled activity, keep the top 2 PCs, and redraw the 2-unit figures in PCA space.

- **All 5 seeds ~1.000 non-conflict** (0.999 on one) — **no near-chance seeds** at all, in sharp contrast to the 2-unit masked net. More units → far more robust to seed.
- Top-2 PCA explained variance: seed 0 **0.72** (PC1 0.44, PC2 0.28); across seeds 0.72–0.86.
- Conflict resolution (untrained, random labels): `audL_visR` splits 60/40 right/left; `audR_visL` ~65/35 — a modest, direction-independent **rightward bias**, not a clean audio/vision rule.
- **`det_multisensory` = 0.000** on seed 0 — the held-out detection-conflict is answered entirely "wrong" (mapped consistently to another class).

> **Why it matters.** Three points. (1) **Capacity buys robustness:** 5 units removes the seed-fragility that plagued the 2-unit nets. (2) **Honest visualisation:** two PCs capture only 72–86% of the variance, so it is explicitly flagged that **~14–28% of the dynamics is off-plane** and the region backdrop is a mean-slice approximation — "indicative, not exact." Seeds are shown as **small multiples in their own PC frames** (not overlaid) because each frame is only defined up to sign/rotation — the principled fix to the raw-axis seed-comparison problem raised in NB05. (3) The **`det_multisensory` = 0.000** is a clean **out-of-distribution generalisation failure**: the otherwise-flawless larger network systematically mis-maps a subtask it never saw — a nice counterpoint to the conflict-trained 2-unit net, which got the same held-out subtask *right* (1.000).

---

## 7. Version 8 — comparable task-defined axes, and the noise result that reversed

**Folder:** `Notebooks_v8/` (notebooks 01 to 15 plus `generated_trials_v8/` and per-analysis output
folders). The task design stays frozen from v6. Two things change in v8, and both are methodological
rather than task changes.

**First**, the interpretability moves out of the network's own coordinate system. Every figure up to v7 was
drawn on axes belonging to the network: hidden unit 1 versus 2, or PC1 versus PC2. Those axes differ
between seeds and between model sizes and are only defined up to rotation and reflection, which is exactly
the caveat flagged in v6 NB05 and only partly patched by the small-multiples PCA of NB08. v8 replaces them
with axes defined by the **task**, so networks of any size can be plotted in one shared space.

**Second**, the noise question is tested properly, at a scale where noise can actually compete with the
signal. This reverses an earlier conclusion.

### 7.1 `09_masked_sweep_v8` — do you ever need cross-unit connections?

A follow-up to the v7 masking experiment: sweep the masked model across hidden sizes, not just at
two units, to see whether the constraint costs anything as capacity grows. Masked and standard GRU, 1 to 8
units, 5 seeds each, 80 networks. Mask enforced on every forward pass in both training and testing;
verified maximum off-diagonal magnitude **exactly 0.0e+00** across every trained model at every size.

**Result: the masked model matches the standard GRU at every network size.** Cross-node connections are not
needed at any network size for the same accuracy, and a significance test on the two curves would find no
significant difference between them.

Two nuances that belong in the caption rather than being smoothed away. At **two units** the masked model
has a slightly **higher mean** but **much higher variance across seeds**, consistent with the bimodal
seed-fragility found in v7 NB06 (two of five seeds collapsing to about 0.69 to 0.75). From **three units
up** it is stable, and in fact more stable than the standard GRU. The per-subtask panel again shows poor
`det_multisensory`, which is the never-trained held-out subtask, so that is a generalisation gap and not a
failure of the masking.

> **Why it matters.** This is the promotion of a single-size curiosity into a general
> architectural claim. v7 showed two isolated units can still solve the task; v8 shows recurrent cross-talk
> is unnecessary **at every capacity tested**, which means the integration in this task is done by shared
> input drive plus the linear readout rather than by units exchanging information. The honest framing is
> "no significant difference", not "masked is better", and the two-unit variance is the price of the
> constraint rather than a point in its favour.

### 7.2 `10_masked_conditions_v8` — three conditions, five seeds, one variable at a time

Masked 2-unit GRU under three conditions, five seeds each, each rendered as a row of five per-seed panels
showing readout decision regions, averaged trajectory per subtask, and baseline fixed points classified by
stability. Trials are generated in-notebook so the noise can be switched off, and everything else is held
identical so only the intended variable moves.

| condition | noise | conflict training labels |
|---|---|---|
| baseline | 0.1 | random |
| no_noise | 0.0 | random |
| conflict_trained | 0.1 | audio wins |

**Findings.** Removing the noise changed almost nothing (Joseph: *"It's very similar, just slight changes...
overall, the difference is not that significant."*). Conflict training produced a markedly cleaner geometry
(*"much cleaner in terms of the lines, much better defined... It's the only one where we actually have four
square boxes straight away"*, and *"the X's are still twisted, which makes sense because the boundaries are
at the corners"*), consistent with the 1-attractor to 3-attractor enrichment found in v7 NB07. And the
**decision boundaries differ substantially between seeds**, which was not the case for the unmasked GRU,
suggesting masking widens the set of solutions reached.

**Comparison chain to state explicitly:** baseline to no_noise isolates the noise; no_noise to
conflict_trained isolates the audio-wins labels. Do not compare baseline directly against conflict_trained
if more than one variable differs between them.

> **Why it matters.** At the time this looked like a clean refutation of the noise
> hypothesis, and it was written up that way. Section 7.7 shows why that reading was premature: the test
> compared 0.1 against 0.0 when stimulus intensity is drawn from [1.0, 3.0], so both conditions were
> effectively "no noise". The lesson is about **effect size relative to the signal**, not about the
> hypothesis: a null result is only informative if the manipulation was large enough to matter.

### 7.3 `11_pca_per_seed_v8` — the PCA figure, replicated across seeds

v6 NB08 produced the PCA state-space figure for seed 0 only. The next step: replicate the whole figure once
per seed, five panels, so the geometries can be compared. Each seed is fitted its **own** PCA frame on its
own pooled hidden activity, because a PC frame is defined only up to sign and rotation and cannot be shared.

**The question it settles.** The initial assumption was that the difference between the **cross** geometry
(small networks) and the **C** geometry (the 5-unit PCA) was a difference between **model sizes**. The more
interesting alternative is that the task admits **multiple solutions**, and that a network lands on one or
the other depending on initialisation. That is what this figure tests.

The notebook also prints per-seed geometry metrics computed in the **full hidden space**, so they are
independent of the projection: the angle between the detection and localisation coding axes, the effective
dimensionality (participation ratio) of the endpoint cloud, and the correlation of the angle with
`det_multisensory` accuracy. It additionally includes a 3-unit network drawn directly in 3D with a
participation ratio, to answer the dimensionality question numerically rather than by eye.

### 7.4 `12_noise_robustness_v8` — one trained network, many test noise levels

Load the trained models and test them across a range of noise levels: test noise on the x-axis, test
accuracy on the y-axis, one line per model. Four
conditions at two units (standard and masked, each trained at noise 0.1 and 0.0), five seeds each, tested
across noise 0 to 1.0 with **no retraining**. The design point: test trials are generated **once, clean**,
and noise is added on top at each level, so the underlying trials and their intensities are identical along
the whole x-axis and only the noise varies. Reports two panels plus an area-under-curve and a half-fall
summary per condition. This is a nice-to-have rather than a load-bearing result.

### 7.5 `13_task_aligned_space_v8` — the *where* versus *whether* space (the new main method)

**The proposal, and the reason it is the right move.** Instead of plotting the state in neural activity
space, ask the activity two task questions: *where* (left or right) and *whether* (detected or not). Those
questions mean the same thing in every network regardless of size, and in principle for recorded mouse
neurons too, so the resulting axes are comparable in a way hidden-unit and PCA axes never were.

**Method.** Take a trained network, run the full test set, record hidden states as trials by time by
neurons. Fit a logistic decoder predicting left versus right **on localisation trials only**, and a second
predicting detection versus no detection **on detection trials only**. Each decoder's weight vector is a
direction in activity space. Project every hidden state onto both to obtain `z_where` and `z_whether`, then
redraw every earlier figure in that plane.

**Three implementation decisions that make the space trustworthy.**

1. **Conflicts are excluded from the fits entirely**, along with the held-out `det_multisensory`. The
   decoders never see a conflict trial, so when the conflicts are projected, where they land is a genuine
   out-of-sample result rather than a restatement of the fit. This is the same logic as the v4 random-label
   conflict design: the finding counts because the network was not told the answer.
2. **Axis signs are pinned** so positive always means right and positive always means detection. Logistic
   regression takes its sign from internal class ordering, so without this the axes flip arbitrarily between
   seeds and the cross-seed comparison, which is the entire purpose of the method, becomes meaningless.
3. **One fixed axis pair, pooled across timesteps**, rather than a new pair at each timestep, so the space
   does not move underneath the trajectories. Weights are unit-normalised so magnitudes are comparable
   across networks.

**Why each decoder gets only some trials.** Each answers one question and needs trials where that question
is well posed. A detection trial such as `det_auditory_only` plays sound on both sides at once, so it has no
left or right label and would only add noise to the *where* fit. The localisation trials are all trials
where something is present, so they offer no no-detection case to contrast against for the *whether* fit.
Fitting and projecting are separate steps: six subtask types define the *where* axis and three define the
*whether* axis, but **all twelve subtasks are projected and plotted**.

**Two measurements that replace describing shapes by eye.**

- **Angle between the axes.** 90 degrees means *where* and *whether* occupy independent directions, a
  factorised code, the **cross**. Near 0 degrees means they are stacked along a single direction, a
  collinear code, the **C**.
- **Alignment to the hidden-unit axes** (two units only). 0 degrees means one unit per task, 45 degrees
  means both units carry both tasks. This quantifies the "rotation" of the solution.

**Results, masked 2-unit GRU, five seeds. Decoder accuracy (3-fold cross-validated) is 0.999 to 1.000
everywhere, so both axes are genuine directions and not artefacts.**

| seed | angle between axes | align where | align whether |
|---:|---:|---:|---:|
| 0 | 82.4° | 43.5° | 36.0° |
| 1 | 62.5° | 5.5° | 22.0° |
| 2 | 29.5° | 39.6° | 10.1° |
| 3 | 84.8° | 37.0° | 42.2° |
| 4 | 7.3° | 39.6° | 43.1° |

Mean angle **53.3° with standard deviation 30.4°**. Mean alignment: where **33.1°**, whether **30.7°**.

**Single-seed size comparison (seed 0 only):** 2 units 82.4°, 3 units 83.4°, 5 units 83.6°, with decoder
accuracy 0.999 to 1.000 throughout.

> **Why it matters — four separate points.**
>
> **(1) The variance lives across seeds, not across sizes.** At two units the angle spans almost the entire
> available range (7° to 85°, sd 30.4°), while across sizes at a fixed seed it barely moves (82.4, 83.4,
> 83.6). This is the direct answer to the open question: the cross versus C
> distinction is **multiple solutions to the same task, selected by initialisation**, not an effect of model
> size.
>
> **(2) But that size comparison is confounded.** All three sizes used seed 0,
> which happens to be a cross seed, so size and seed are entangled. All that can honestly be claimed is
> that seed 0 gives an orthogonal solution at every size. Running every size at every seed would
> resolve this, with a variance decomposition (spread across seeds within a size, versus spread of the
> per-size means) and a scatter of angle against `det_multisensory` accuracy.
>
> **(3) The mean of 53.3° is not a useful summary.** With values at 7, 29, 62, 82 and 85 the distribution is close to
> bimodal and essentially no seed sits near 53. The mean describes a network that does not exist. Report the
> range and the split. This is the same discipline applied in v6 to `det_multisensory = 0.534 ± 0.452`.
>
> **(4) Both geometries decode perfectly, so the task does not require factorisation.** Seed 4 has its two
> axes only 7.3 degrees apart yet both decoders still reach 1.000. That is not a contradiction: the two
> decoders are fit on **disjoint trial sets**, so all four conditions can be strung along a single
> direction (no-detection, left, right, detected) and a four-way readout can still carve four regions out of
> a line. The consequence is the interesting part. **Factorisation buys nothing in training accuracy**, so
> if it buys anything it must show up elsewhere, which generates the prediction in 7.9.
>
> Two further observations. Alignment is largely **independent** of the angle: seed 4 has near-parallel axes
> yet alignment near 40 degrees, meaning its single shared direction is itself diagonal to the unit axes.
> And specialisation does occur occasionally but not reliably: seed 1's *where* axis at 5.5 degrees and seed
> 2's *whether* axis at 10.1 degrees did land close to a single unit.

**The decoder thresholds also give an interpretable backdrop.** `z_where = -b_where` separates left from
right and `z_whether = -b_whether` separates no detection from detection, so the four quadrants they cut out
literally **are** the four task conditions. That replaces the readout's argmax regions in an arbitrary
neural basis with a frame a reader can interpret without knowing anything about the network.

### 7.6 The 3D figure, rebuilt

The original 3D plot (three hidden units, one cube, all twelve subtasks) was reported as unreadable, and
the diagnosis is specific: colour was asked to carry twelve categories in a medium where occlusion and
perspective already distort apparent hue and line weight, and the palette had been deliberately grouped
(all detection orange, all localisation blue) because that read well in 2D. Categorical colour separates
about six or seven classes in a clean 2D plot and fewer in 3D, so it collapsed to "orange versus blue".

The redesign stops asking colour to do all the work, and gives the third axis a meaning:

- **Figure 6, the main version.** Small multiples (Detection, Localisation, Conflicts) on identical axes and
  an identical camera angle, so each panel holds three to six lines. The axes are `z_where` by `z_whether`
  by **time**, so two of the three carry task meaning, unlike a plot of three hidden units where no axis
  means anything on its own. Endpoints are labelled directly instead of via a legend, faint wall projections
  restore the depth cues a static render loses, and the stimulus on and off points are marked on the time
  axis.
- **Figure 6b.** The same panels coloured by timestep, which shows *when* the network commits, information
  that was not visible at all in the original.
- **Figure 6c.** A plotly version in which clicking a legend entry isolates a trace, so the reader removes
  what they do not want to see rather than guessing in advance.
- **Figure 6d.** The 3-unit network in its own activity space, kept for one narrow purpose only: to show
  whether three units genuinely use all three dimensions or collapse onto a plane. A participation ratio is
  printed so that question is answered numerically and the plot only has to support it.

> **Why it matters.** A visualisation-design decision
> rather than a cosmetic fix. Adding time as the third axis makes trajectories separate as they evolve
> instead of tangling, but it also means the figure is no longer a pure state-space picture, which is
> precisely why 6d is retained. The trade-off is named rather than hidden.

### 7.7 `14_training_noise_sweep_v8` — the correction: noise does shape the solution

**This section supersedes the earlier conclusion that the noise hypothesis was refuted.**

**Why the original test was not a test.** The v8 NB10 comparison used training noise **0.1 versus 0.0**. But
stimulus intensity is drawn from **[1.0, 3.0]**, so a noise standard deviation of 0.1 is roughly a thirtieth
of a typical signal. The two conditions were both effectively "no noise", so finding no difference between
them carried very little information. The analogy is testing whether temperature affects a reaction by
comparing 20°C with 20.5°C.

**The proper test.** Networks are **trained from scratch** at noise 0, 0.1, 0.25, 0.5, 1.0, 1.5 and 2.0,
masked and standard, five seeds each, 70 networks. At 1.0 the noise is comparable to the weakest stimulus;
at 2.0 it exceeds most of them. Crucially the notebook separates two questions that the earlier comparison
had run together:

- **How hard is the task at this noise level?** Accuracy against training noise, measured both on a matched
  test set at the same noise and on a clean test set with the noise removed. Accuracy falling is expected
  and not itself interesting.
- **Does the network find a different solution?** The two geometry measures from NB13. This is the
  hypothesis.

**Results.** The angle between the *where* and *whether* axes **rises with training noise**, from roughly 60
degrees to nearly 80 on the mean line. The **alignment to the hidden-unit axes also rises**, so higher noise
makes the network share both tasks across both units rather than specialising. And the **spread across seeds
shrinks**: at noise 0 the error bar is very wide (individual seeds span roughly 7 to 82 degrees), while by
noise 2.0 the spread is much narrower and the seeds converge high. There is a positive correlation between
training noise and alignment.

> **Why it matters — the most important reversal in the project.** The correct framing
> is not "we tested the hypothesis and it was wrong". It is: **the hypothesis was untestable at the original
> noise level because the manipulation was negligible relative to the signal; tested across a range where
> noise competes with the signal, noise does shape the learned solution.** And the interesting part is not
> merely that the angle rises. It is that the **across-seed spread collapses**. At low noise, initialisation
> decides which of several solutions the network finds (7.5); at high noise that multiplicity disappears
> and the networks converge on one organisation. So noise acts as a **selection pressure among solutions**,
> which ties directly into the multiple-solutions finding and turns two separate observations into one
> story. The methodological lesson: a null result is only evidence when the
> manipulation was large enough to have produced an effect.
>
> **Caveat to keep.** At very high noise a network may simply fail the task, in which case the decoders are
> fitting almost nothing and the reported angles no longer describe a real solution. Decoder accuracy is
> printed alongside every geometry number for exactly this reason, and the geometry panels should only be
> read where accuracy is clearly above chance.

### 7.8 `15_noise_geometry_v8` — what the rising line actually looks like

The next step is to see the summary line of 7.7 rendered in the state space: the five-seed task-space
figure at the **lowest** and the **highest** training noise.

**The design point, and the reason this is a separate notebook.** Every network is **trained at its own
noise level but tested and projected on identical clean trials**. In any comparison only one thing should
change: vary the training noise, but test every network at the same noise level. Otherwise a network
trained clean and tested noisy, compared against one trained noisy and tested noisy, differs in two ways at
once and no difference in the trajectories can be attributed to training. This is the same
change-one-variable discipline as v7 NB07 and v8 NB10.

**Only the two extreme noise levels**, not all seven. All seven would be far too crowded, and the summary
line barely moves between training noise 1.0 and 1.5.

**Contents.** Figure 1: five seeds by two noise levels as a 2 by 5 grid, angle printed in each panel title,
expectation being a mixed top row and a consistently orthogonal bottom row. Figure 2: a **seed 4 close-up**,
low versus high, with the two coding axes drawn as arrows from the origin. Figure 3: a paired plot, one line
per seed from low to high noise, for both the angle and the alignment, which shows the **rise** and the
**convergence** in a single picture. Plus a summary table with the per-seed change and the clean accuracy at
both noise levels. Trained networks are cached so the figures can be redrawn without retraining.

**Why seed 4 specifically.** At 7.3 degrees it was the most collinear of the five at low noise, so it has
the most room to change and makes the effect easiest to see: the prediction from 7.7 is that seed 4 should
go from a very low angle to a very high one. Seed 0 started at 82 degrees
and can only gain a few, which makes it a poor illustration. Choosing the most extreme case deliberately, and
saying why, is better practice than picking whichever seed looks best.

### 7.9 The open prediction that links geometry to behaviour

If the four conditions lie along a single direction (the C, low angle), then any unfamiliar input must land
somewhere on that one line and can easily fall into the wrong region, whereas a factorised code (the cross,
high angle) has a separate dimension for each question. This generates two testable predictions:

- **Low-angle seeds should generalise worse on `det_multisensory`**, the subtask that is never trained.
- **Low-angle seeds should be less robust to test noise.**

Both are measurable from data already in hand: `det_multisensory` accuracy per seed comes out of
a per-seed sweep across sizes, and noise robustness from NB12 or NB14. Confirming either would connect the
geometry to behaviour and turn a description of shapes into a mechanistic claim. It may also explain the
long-standing `det_multisensory = 0.000` result of v5 and v6: in a collinear code, an unseen input
combination can push the state anywhere along the line.

---

## 8. The narrative arc — how the versions build on each other

1. **v1:** Build two tasks; discover they are (a) **too easy** (1.000 / 0.997) and (b) label-imbalanced (75% majority trap). → *We need harder, balanced tasks, and conflict trials are the only interesting behaviour.*
2. **v2:** Exact balance; first **4-class unified** network (0.762, overfits); hold conflicts out of training. Single-seed sweep is too noisy to read. → *Multi-seed needed; unified task is the real problem.*
3. **v3:** Curriculum experiments expose the two core phenomena — **pattern-matching** (100% non-conflict, 0% conflict) and **catastrophic forgetting**. → *We must force integration, not memorisation.*
4. **v4:** **Random-label conflict training** forces discovery → conflict accuracy 0.000 → 0.560, non-conflict 1.000; first proper **multi-seed sweep** separates signal from seed luck (unified needs ≥4 units). → *Unified training + discovered reliability weighting is the paradigm.*
5. **v5:** Bake the biology into the labels (visual-only = not-detected; `det_multisensory` as a test-only auditory-dominance probe). Unified reaches 0.999 on trained subtasks, but **auditory dominance = 0.000** everywhere — a real null. → *Why is the network visual-dominant? Can it even represent auditory dominance?*
6. **v6/v7:** Freeze the task; confirm **GRU beats RNN** (keep GRU); reconfirm capacity; then the interpretability payoff — open the 2-unit GRU as a **dynamical system** (attractors, saddles, input-drive axes) explaining additive-then-winner-take-all integration; **mask** the recurrence (integration survives but training gets fragile); **teach** auditory dominance (learned cleanly, 1→3 attractors — proving the v5 null is about reward, not capacity); and **scale to 5 units via PCA** (robust performance, honest projection caveats, an OOD failure on the held-out subtask).
7. **v8:** Promote the masking result from one size to **all sizes** (cross-unit connections are never
   needed for accuracy). Then two methodological turns. First, replace network-defined axes with
   **task-defined axes** (*where* versus *whether*), which for the first time makes cross-seed and
   cross-size comparison valid and reduces the cross-versus-C question to a single number. That number
   varies enormously across seeds but not across sizes, so the geometries are **multiple solutions selected
   by initialisation**, not a capacity effect. Second, retest the noise hypothesis at a scale where noise
   competes with the signal, which **reverses the earlier null**: noise raises the angle, raises the
   sharing across units, and **collapses the across-seed spread**, so noise acts as a selection pressure
   among the available solutions.

---

## 9. Cross-cutting themes

- **Chance / luck vs. real effect — the recurring discipline.** Every conflict/dominance number is judged against its chance level (0.25 for 4-class, 0.5 for binary conflict readouts) and against **cross-seed variance**. Concrete cases: the v2 single-seed sweep zig-zag (0.463→0.907→0.573) dismissed as optimisation luck once v4's error bars appear; `det_multisensory = 0.534 ± 0.452` dismissed as **symmetry-breaking coin-flips** because the std is enormous; `loc-conflict = 0.010` recognised as *anti*-signal from a collapsed network; per-seed left/right-split figures added specifically to test "real bias vs symmetry-breaking."
- **Artefact vs. mechanism.** Sequential-curriculum conflict scores are labelled **forgetting artefacts**; the detection-group 0.75 plateau is labelled an **artefact of an always-wrong held-out subtask**, not a detection failure. No number is reported without asking what could produce it spuriously.
- **Controlled experiments — change one thing.** The conflict-trained network (NB07) changes *only* the conflict labels, so any dynamical difference is attributable to that alone. The RNN-vs-GRU sweep changes *only* the cell type.
- **Discovered vs. instructed behaviour.** The random-label conflict design is the conceptual keystone: reliability weighting counts as a *finding* precisely because the network was given no consistent conflict signal to copy.
- **Honesty about limitations, in-notebook.** PCA off-plane variance, mean-slice region backdrops, sign/rotation ambiguity of hidden axes, the HIDDEN_SIZE 5-vs-6 discrepancy, the "keep all seeds, no quality selection" policy, even the CPU-faster-than-GPU benchmark — all documented rather than hidden.
- **From behaviour to mechanism.** The project deliberately moves from "does it score well?" (v1–v5) to "*how* does it compute the answer?" (v6/v7 dynamical-systems analysis), grounded in the standard interpretability literature (Sussillo & Barak, Mante, Maheswaranathan, Ji-An).
- **Effect size relative to the signal, not just presence or absence of an effect.** The noise reversal
  (7.7) is the clearest case in the project of a null result that was really an underpowered manipulation.
  The original comparison varied noise by about a thirtieth of the stimulus intensity. The general lesson:
  before reporting "no effect", check that the manipulation was large enough to have produced one.
- **Measurement instead of adjectives.** Up to v7, geometries were described by eye ("like an X", "the C
  solution", "more like two right angles"). v8 replaces that with two numbers computed in the full hidden
  space, the angle between coding axes and the alignment to the unit axes, which are comparable across seeds
  and sizes and can be correlated with behaviour. The measurement is what makes the question tractable.
- **Fit on one set, evaluate on another, at every level of the analysis.** The random-label conflict design
  (v4) made reliability weighting a discovery rather than an instruction. The decoder design (v8) applies
  the identical logic one level up: the axes are fitted **without** conflict trials, so where the conflicts
  land is a result and not a consequence of the fit.
- **Confounds named rather than glossed.** The v8 size comparison used one seed, so size and initialisation
  are entangled, and this is stated rather than glossed. Compare the v5/v6 HIDDEN_SIZE 5 versus 6
  discrepancy, which was likewise recorded rather than quietly resolved.
- **Distributions, not means.** The angle mean of 53.3 degrees describes no actual network, because the
  values cluster near the extremes. This is the same reasoning that dismissed
  `det_multisensory = 0.534 ± 0.452` in v6 as symmetry-breaking coin flips.
- **Visualisation as a design problem with trade-offs.** The 3D rebuild (7.6) diagnoses a specific cause
  (twelve categories on one colour channel, in a medium that distorts colour), applies a principled fix
  (small multiples, direct labelling, a meaningful third axis, depth cues), and names what the fix costs
  (it is no longer a pure state-space picture, which is why the neural-space version is retained).

---

## 10. Known discrepancies and caveats

- **HIDDEN_SIZE:** sweeps (v5, v6) recommend **6**; curriculum notebooks were run at **5**. Both sit on the plateau, so conclusions are unaffected.
- **Leftover version labels:** v6's `03` prints "set HIDDEN_SIZE in *02...v5*" — a stale reference.
- **Doc/code mismatch (v2):** unified set described as "3000 trials" but code produces 3300.
- **Weak seeds included:** 2-unit analyses (NB05 figs 4/5) include seeds 2/3 (~0.66 non-conflict); some visible cross-seed spread is real training failure, not just coordinate change.
- **`det_multisensory` scoring:** its "accuracy" is a **dominance readout**, not a correctness metric (nominal label 1 = "auditory wins"); 0.000 means fully visual-dominant, which is a *finding*, not a bug.
- **NB06/07/08 files were saved with cleared outputs** but had been executed; the numbers above were recovered from the saved figures and by faithfully re-running the notebooks against the real data (reproduced values match the figures, modulo one n=65-vs-66 rounding boundary).
- **Superseded conclusion.** The earlier finding that removing noise made no difference refers only to the
  0.1 versus 0.0 comparison, and is superseded by 7.7: noise does shape the solution when tested across a
  range comparable to the signal.
- **Size versus seed confound in NB13.** The 2, 3 and 5 unit angle comparison (82.4, 83.4, 83.6 degrees)
  used **seed 0 only**, so model size and initialisation are entangled and the comparison cannot on its own
  show that size has no effect.
- **The mean angle of 53.3 degrees describes no actual network.** The distribution is close to bimodal, so
  the mean falls in the gap between the two clusters.
- **Linear decoders can only find straight directions.** If a network encoded a variable non-linearly, a
  linear decoder would capture it only partially. This is why cross-validated decoder accuracy is reported
  alongside every geometry number, and at 0.999 to 1.000 it is the justification for the whole approach. If
  it ever falls near chance, the corresponding angles are meaningless.
- **Geometry at very high noise.** Where a network has failed the task, its decoders are fitting little and
  the reported angle does not describe a solution. Read geometry only where accuracy is well above chance.
- **NumPy 2 incompatibility found during validation.** `ndarray.ptp()` was removed in NumPy 2; the code now
  uses `np.ptp(...)`, which works on both.
- **`aud_int` and `vis_int` exist only in the test npz**, not train, so loaders must guard with
  `if "aud_int" in d`. This caused a `KeyError` in NB13 on first run.
- **NB10 condition 3.** If it was run at no noise with audio-wins labels, that is two changes from
  baseline rather than one, and the comparison chain has to go baseline to no_noise to conflict_trained
  rather than baseline to conflict_trained directly.

---

## 11. Figure inventory

- **v6 `04` / `05` (2-unit GRU, raw hidden plane):** decision regions; per-subtask trajectories; baseline flow field + fixed points; input-conditioned flow fields; average trajectory per subtask (`statespace_v6/fig1_all_subtasks.png`); conflict split by output (`fig2_conflicts.png`); intensity-gap gradient (`fig3_gap_gradient.png`); signed-gap bands (`fig3b_gap_bands_bytype.png`); cross-seed averaged conflict (`fig4_...png`); cross-seed decision boundaries (`fig5_boundaries_seeds.png`).
- **v6 `06` (masked 2-unit):** `masked_v6/masked_fig1.png` (average trajectory per subtask — **the 2-unit masked figure**); `masked_v6/masked_flow.png` (single-attractor flow field).
- **v6 `07` (conflict-trained 2-unit):** `conflict_trained_v6/conflict_fig1.png` (trajectories pulled to audio side); `conflict_trained_v6/conflict_flow.png` (3 attractors + 2 saddles).
- **v6 `08` (5-unit, PCA):** `pca_v6/pca_fig1.png` (**average trajectory per subtask in PCA space — the 5-unit figure**); `pca_fig2.png` (conflict split by output); `pca_fig3b.png` (signed-gap bands); `pca_fig5_seeds.png` (per-seed small multiples).
- **v8 `09` (masked sweep):** `masked_sweep_v8/masked_sweep_overall.png` (**masked versus standard at every
  hidden size, the headline architecture figure**); `masked_sweep_persubtask.png`.
- **v8 `10` (three masked conditions):** `masked_conditions_v8/masked_baseline_5seeds.png`,
  `masked_no_noise_5seeds.png`, `masked_conflict_trained_5seeds.png` (each a row of five per-seed panels
  with decision regions, trajectories and fixed points).
- **v8 `11` (PCA per seed):** `pca_seeds_v8/pca_5unit_5seeds.png`; optionally
  `pca_5unit_masked_5seeds.png`; `trajectories_3unit_3d.png`.
- **v8 `12` (noise robustness):** `noise_robustness_v8/noise_robustness.png`.
- **v8 `13` (task-aligned space):** `task_space_v8/taskspace_fig1_subtasks.png` (**average trajectory per
  subtask in task space, the primary interpretability figure**); `taskspace_fig2_conflicts.png`;
  `taskspace_fig3_gap.png` (signed intensity-gap bands); `taskspace_fig4_seeds.png` (**five seeds on
  comparable axes, previously impossible**); `taskspace_fig5_sizes.png` (**2, 3 and 5 units on comparable
  axes**); `taskspace_fig6_3d_panels.png` (where by whether by time, small multiples);
  `taskspace_fig6b_3d_time.png`; `taskspace_fig6d_neural3d.png`.
- **v8 `14` (training noise sweep):** `noise_training_v8/noise_training_accuracy.png`;
  `noise_training_geometry.png` (**the hypothesis test: angle and alignment against training noise**);
  `noise_training_taskspace.png`.
- **v8 `15` (noise geometry):** `noise_geometry_v8/noise_geometry_5seeds.png` (**five seeds at low versus
  high training noise, all tested clean**); `noise_geometry_seed4.png` (the extreme case close-up);
  `noise_geometry_paired.png` (**per-seed change and the convergence**).

---

## 12. Notebook summary

| # | Notebook | What it covers |
|---|---|---|
| 01 to 05 | data generation, curriculum, hidden-unit sweeps, 2-unit interpretability, state space | v6 numbers |
| 06 | masked GRU (self-connections only), 2 units | |
| 07 | conflict-trained 2-unit GRU (audio-wins labels) | |
| 08 | PCA dynamics, 5 units | seed 0 |
| 09 | masked sweep across hidden sizes | |
| 10 | masked GRU, three conditions | |
| 11 | PCA state space, one panel per seed | |
| 12 | noise robustness of trained models | |
| 13 | task-aligned *where* / *whether* space | the geometry table above |
| 14 | training-noise sweep | the noise reversal |
| 15 | noise geometry, low versus high training noise | |

Policy across all notebooks: **train all seeds, print every accuracy, use an explicit `DISPLAY_SEED` knob
defaulting to the first seed by index, and never filter for a good seed.**
Several findings in this project (the v7 masked bimodality, the v7 seed 2 asymmetry, the v8 angle spread)
exist only because no seed was discarded.

---
