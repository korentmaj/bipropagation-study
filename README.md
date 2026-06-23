# Bipropagation: An Independent Reproduction & Decomposition Study

An honest, independent reproduction and component-level decomposition of Dr. Bojan Ploj's **bipropagation**, a greedy, layer-wise, supervised neural-network training method proposed as an alternative to global backpropagation.

This repository tests bipropagation's specific quantitative claims against properly tuned modern backpropagation baselines. It also decomposes the method into its parts to isolate *which* component, if any, delivers a measurable benefit.

---

## TL;DR / Key findings

**The honest thesis.** Ploj's headline quantitative claims (roughly *25x faster* than backpropagation, *~100% reliable*, and *beating backprop*) do **not** reproduce against a tuned modern backprop baseline. The deterministic-initialization demo's reported 100% accuracy **could not be reproduced**; an honest run reaches ~88%. **But** the core intuition is validated: per-layer supervision genuinely helps train deep networks. A deeply-supervised control (per-layer auxiliary heads but a *single global gradient*) isolates the operative mechanism as **per-layer supervision**, *not* the absence of backpropagation, with a small, secondary *locality* effect appearing only on CIFAR-10/CNN.

In short: **Ploj's broader intuition holds; his specific mechanistic and quantitative claims do not.**

### MNIST / MLP (test accuracy vs. depth)

Full-scale run, 30k train / 10k test, seed 0:

| Depth | Vanilla BP | Modern BP | Anchors (Ploj-style) | Local-loss |
|------:|:----------:|:---------:|:--------------------:|:----------:|
| 2  | 0.9587 | 0.9690 | 0.8774 | **0.9708** |
| 4  | 0.9630 | 0.9735 | 0.8743 | **0.9714** |
| 8  | 0.9586 | 0.9627 | 0.8653 | **0.9701** |
| 16 | 0.1135 (collapse) | 0.9513 | 0.8415 | **0.9685** |

The locality-isolation control (30k MNIST, seed 0, 30 epochs) shows the apparent "local-loss beats backprop at depth" result is largely a weak-baseline artifact, and that **deeply-supervised ≈ local-loss** at every depth:

| Depth | Residual BP | Plain BP (ReLU+BN, 30ep) | Deeply-supervised (global grad + aux heads) | Local-loss |
|------:|:-----------:|:------------------------:|:-------------------------------------------:|:----------:|
| 8  | 0.9674 | 0.9691 | **0.9725** | 0.9701 |
| 16 | 0.9351\* | 0.9661 | **0.9684** | **0.9685** |

\*The residual MLP at depth 16 was under-tuned within the epoch budget and is not a load-bearing baseline.

At depth 16, deeply-supervised (0.9684) ≈ local-loss (0.9685): keeping per-layer supervision while *restoring* the global gradient reproduces the result. The benefit comes from per-layer supervision, not from avoiding global backpropagation.

### CIFAR-10 / CNN (mean test accuracy ± std over 3 seeds)

15k train / 10k test, 3 seeds, 12 epochs, no augmentation (held identical across methods):

| Depth (blocks) | E2E backprop | Greedy local-loss | Deeply-supervised |
|---------------:|:------------:|:-----------------:|:-----------------:|
| 3 | 0.528 ±.016 | **0.567** ±.003 | 0.520 ±.022 |
| 6 | 0.643 ±.007 | **0.649** ±.004 | 0.577 ±.018 |
| 9 | 0.557 ±.022 ↓ | **0.626** ±.005 | 0.609 ±.013 |

On CIFAR the plain (non-residual) end-to-end CNN degrades at depth 9 (0.643 → 0.557). Both per-layer-supervised methods are more depth-robust, and here `local` (0.626) edges out `deepsup` (0.609) at depth 9, so *locality* contributes a small, secondary robustness on CNNs that pure deep supervision does not fully capture. The primary mechanism is still per-layer supervision.

### Reliability and speed (MNIST, depth 6, FAST_MODE indicative)

- **Speed:** layer-wise bipropagation is the *slowest* method tested (20.6s), not 25x faster. The deterministic variant is genuinely the fastest (~1.6x vs. modern backprop) but at much lower accuracy.
- **Reliability:** modern backprop has the *lowest* seed-to-seed variance (std 0.0008). Bipropagation is stable but does not beat it.

---

## Methods compared

All methods share one framework, architecture, and data pipeline to avoid infrastructure confounds.

| Method | Description |
|---|---|
| **End-to-end (vanilla) backprop** | Naive init, saturating (tanh) activation, plain SGD. This is the regime where vanishing gradients bite. |
| **Modern backprop** | Adam + He init + BatchNorm. The strong baseline. |
| **Greedy local-loss (layer-wise)** | The greedy bipropagation scaffold, but each layer is trained with a temporary softmax head and cross-entropy (à la Belilovsky 2019 / Nøkland 2019); the head is discarded before the next layer. |
| **Deeply-supervised control** | Per-layer auxiliary classifier heads with a *single global gradient* (Lee 2015). The locality-isolation control: same per-layer supervision, but global backprop is retained. |
| **Anchors (Ploj-style)** | Best-effort reconstruction of Ploj's hand-designed intermediate-target rule: each layer shifts its input toward per-class anchors/prototypes, weights initialized near identity, final softmax readout. |
| **Deterministic centroid-init** | One hidden layer constructed analytically from class-centroid geometry (sparse ±1 units over the 3 most-discriminative features), targets = `0.99·layer_output + 0.01·two_hot(class)`, refined with Adam. |

---

## Repository structure

```
.
├── README.md                       # this file
├── PAPER.md                        # the English paper (authoritative findings & numbers)
├── LICENSE                         # MIT
├── requirements.txt
├── .gitignore
├── experiments/
│   ├── cifar_experiment.py         # CIFAR-10 / CNN decomposition (e2e, local, deepsup)
│   └── mnist_mlp_experiment.py     # MNIST / MLP, all methods (self-contained)
└── archive/
    ├── README.md
    └── ...                         # raw development fragments, kept for provenance
```

---

## How to run

The experiments are self-contained TensorFlow 2 / Keras scripts that each run in a single Colab cell or locally.

### Local

```bash
pip install -r requirements.txt
python experiments/cifar_experiment.py        # CIFAR-10 / CNN
python experiments/mnist_mlp_experiment.py    # MNIST / MLP
```

### Colab

Upload a script (or paste it into a cell) and run. A GPU runtime (e.g. T4) is recommended for the full configs.

### Notes

- **`FAST_MODE` flag.** Each script has a `FAST_MODE` toggle near the top. `True` gives a small, fast indicative smoke run (subset of data, few epochs, 2 to 3 seeds); set it to `False` for the full benchmark reported in the paper.
- **CIFAR-10 download.** `cifar_experiment.py` downloads CIFAR-10 from `cs.toronto.edu` via `tf.keras.datasets` on first run and caches it to disk; subsequent runs reuse the cache.
- All reported numbers come from actual evaluation runs. None are hardcoded.

---

## Limitations

- **Seeds.** Most decisive numbers (full-scale and control runs) are single-seed (seed 0); the multi-seed evidence is currently FAST_MODE / CIFAR only. A fuller protocol (≥10 seeds, 95% CIs, paired Holm-Bonferroni tests) is left for follow-up.
- **Plain, non-residual baselines.** Both testbeds compare against plain baselines that degrade with depth for known optimization reasons. A residual/normalized end-to-end baseline would likely close the depth gap, so the depth-robustness claims are relative to *plain* architectures, not modern residual networks.
- **Iso-compute.** Local-loss sees the data roughly 3-6x more often than a single end-to-end run; a clean accuracy-vs-wall-clock and iso-gradient-step accounting is still outstanding.
- **Reconstruction of Ploj's rule.** The "anchors" method reconstructs an unpublished multi-class target rule (the original `MNIST.m` is auth-walled on ResearchGate). A more faithful target scheme could raise the anchors numbers, though it would not change the per-layer-supervision-not-locality conclusion.

---

## Credit

The **bipropagation method and the underlying intuition, that per-layer supervision can help train deep networks, originate with Dr. Bojan Ploj.** This repository is an independent reproduction and decomposition of his work; the credit for the original idea is his. We thank him for making the method and code public, which is what made this study possible.

Dr. Ploj's repositories:
- [github.com/BojanPLOJ/Bipropagation](https://github.com/BojanPLOJ/Bipropagation)
- [github.com/BojanPLOJ/Deterministic-Bipropagation-Initialization](https://github.com/BojanPLOJ/Deterministic-Bipropagation-Initialization)

Related foundational work this study builds on includes Deeply-Supervised Nets (Lee et al. 2015), greedy layer-wise learning at scale (Belilovsky et al. 2019), local error signals (Nøkland & Eidnes 2019), and Difference Target Propagation (Lee et al. 2015). See `PAPER.md` for the full reference list.

---

## Contributing / further research

Contributions and extensions are warmly welcome. This is intended as an open, honest starting point, not a closed verdict. Particularly valuable directions:

- **Residual / normalized end-to-end baselines.** Does the depth-robustness gap survive against a properly modern baseline?
- **More seeds + confidence intervals.** ≥10 seeds, 95% CIs, paired Holm-Bonferroni tests on the full-scale and control tables.
- **Harder data.** CIFAR-100, Tiny-ImageNet.
- **Other local-learning methods.** Forward-Forward, Difference Target Propagation, synthetic gradients, feedback alignment, as additional points of comparison.
- **A more faithful reconstruction** of Ploj's multi-class intermediate-target rule (ideally from the original `MNIST.m`).
- **Iso-compute accounting.** Accuracy vs. wall-clock and vs. gradient steps.

Open an issue or a pull request.

---

## Citation

```bibtex
@misc{korent2026bipropagation,
  author       = {Korent, Maj},
  title        = {Bipropagation: An Independent Reproduction and Decomposition Study},
  year         = {2026},
  howpublished = {\url{https://github.com/korentmaj/bipropagation-study}},
  note         = {Independent reproduction and decomposition of Bojan Ploj's bipropagation method.}
}
```

---

*This study is offered in a spirit of constructive, respectful scientific scrutiny. The aim is to separate what reproduces from what does not, and to credit the genuine insight at the core of the method.*
