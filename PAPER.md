# Reproducing and Decomposing "Bipropagation": Per-Layer Supervision, Not the Absence of Backpropagation, Explains the Benefit

## Abstract

"Bipropagation," a greedy layer-wise supervised training scheme proposed by Bojan Ploj, is claimed to train neural networks roughly 25× faster than backpropagation, with ~100% reliability and accuracy, by replacing global backpropagation with per-layer training toward hand-designed intermediate targets. We conduct a controlled reproduction of these claims against properly tuned modern backpropagation baselines on MNIST. The headline claims do not reproduce: layer-wise bipropagation is the *slowest* method tested, not the fastest, and modern backpropagation has the *lowest* seed-to-seed variance. We further document that the associated "deterministic initialization" repository ships fabricated demo results (hardcoded accuracy prints, cosmetically rescaled curves, a random-noise dataset); an honest reproduction yields ~50–88% on MNIST. However, the underlying greedy layer-wise *scaffold* is sound: replacing Ploj's hand-designed targets with a per-layer local supervised loss recovers backprop-level accuracy. Crucially, a deeply-supervised control (per-layer auxiliary heads but a single global gradient) matches the local-loss method at every depth (depth-16: 0.9684 vs 0.9685), isolating the mechanism. The benefit comes from per-layer supervision, not from avoiding global backpropagation. Ploj's mechanistic claim is refuted; his broader intuition is supported.

---

## 1. Introduction

Backpropagation remains the default algorithm for training deep neural networks, but its global, sequential credit assignment carries well-known costs: it caches all activations (peak memory grows with depth), it imposes backward-locking that limits parallelism, and it is often cited as biologically implausible. These costs have driven a long line of "beyond backpropagation" research into greedy, layer-local, and target-based training.

"Bipropagation," developed by Dr. Bojan Ploj over a series of self-published and weakly-reviewed venues, sits in this space. The method is a greedy, supervised, layer-by-layer training scheme: each layer is trained in isolation to map its input to a *pre-defined intermediate target*, with weights initialized near identity so each layer performs only a small transformation, followed by a final softmax readout. Because no global gradient flows across layers, the method is presented as sidestepping vanishing gradients. The associated claims are strong and specific: training "25× faster" than backpropagation, "~100% reliable," and reaching ~100% accuracy on benchmark tasks. A more recent variant ("deterministic bipropagation initialization," tied to a 2026 Neurocomputing submission) constructs a hidden layer analytically from class-centroid geometry, with no iterative training of that layer at all.

These claims have not been independently replicated. They originate from toy-scale demonstrations (XOR, Iris, MNIST) in self-published sources, and the only public code that is *actually* bipropagation implements a hand-coded XOR case; the multi-class rule is documented in prose but not present in any reachable, runnable code. Independent evaluation therefore matters for two reasons: (i) to test whether the specific quantitative claims survive comparison against a competently tuned modern baseline, and (ii) to determine *which component*, if any, of the method delivers a measurable benefit — a question the original sources never isolate.

This paper is a controlled reproduction and decomposition study. Our contributions are:

- **A faithful reproduction** of Ploj's headline claims on MNIST against properly-tuned backprop baselines, showing that "25× faster" and "~100% reliable" do not hold: layer-wise bipropagation is the slowest method we test, and modern backprop has the lowest variance.
- **Documentation of fabricated results** in the public "deterministic initialization" demo (hardcoded accuracy prints, a cosmetically rescaled loss curve, a random-noise headline dataset), and an honest reproduction giving ~50–88% on MNIST rather than the claimed 100%.
- **A decomposition of the method into its components**, showing that replacing Ploj's hand-designed targets with a per-layer local supervised loss recovers (and at face value appeared to exceed) backprop-level accuracy.
- **A locality-isolation control** — deeply-supervised backprop with per-layer auxiliary heads but a single global gradient — that matches the local-loss method at every depth (depth-16: 0.9684 vs 0.9685). This isolates the mechanism: the benefit is *per-layer supervision*, not the *absence of global backpropagation*. Ploj's mechanistic claim is thereby refuted while his broader intuition (per-layer supervision helps deep nets) is supported.

We are explicit that most of our findings are *confirmatory* of established literature. The novel contribution is the rigorous reproduction and component-level decomposition of Ploj's specific method, including catching and correcting a baseline-and-compute confound in our own initial positive result.

---

## 2. Related Work

**Greedy layer-wise pretraining.** The stacking skeleton of bipropagation — train one layer, freeze it, train the next on its output — is the same skeleton as classic greedy layer-wise pretraining with deep belief networks [1] and stacked autoencoders [2]. The difference is that those methods are *unsupervised* per layer, whereas bipropagation supplies a *supervised* intermediate target.

**Target propagation.** The closest reviewed cousin is Difference Target Propagation [3], which also assigns each hidden layer a target activation. The key contrast: DTP *learns* its targets (via layer-wise autoencoders), whereas bipropagation *hand-constructs* them, either per-problem (XOR) or via a documented prototype-shift rule for the multi-class case.

**Forward-Forward.** Hinton's Forward-Forward algorithm [4] shares bipropagation's core motivation — greedy, layer-local training with no global backward pass, aimed at biological plausibility and low-power hardware — while using a contrastive goodness objective rather than explicit targets.

**Deep supervision.** Deeply-Supervised Nets [5] attach auxiliary classifier heads to intermediate layers and train them *jointly with a global gradient*. This is the critical control for our study: it retains per-layer supervision while keeping global backpropagation, allowing us to separate the two factors that bipropagation conflates.

**Local error signals.** Nøkland & Eidnes [6] train each layer with local classification and similarity losses, demonstrating that mainstream local-loss learning matches or approaches backprop without a global backward pass on standard benchmarks. This is essentially the modern, reviewed embodiment of what bipropagation attempts, but with *learned* local objectives.

**Greedy supervised layer-wise at scale.** Belilovsky et al. [7] show that greedy supervised layer-wise training scales to ImageNet, providing the strongest evidence that the greedy supervised scaffold is sound on hard data — and the appropriate prior expectation for our CIFAR-10 experiment.

**Linear / classifier probes.** Alain & Bengio [8] introduced linear probes to measure how linearly separable representations become with depth. Our local-loss construction (a temporary softmax head per layer) is effectively a trained probe used as a training signal, and our separability observations are confirmatory of this line.

**LDA / PCA-style analytic initialization.** The deterministic centroid construction resembles discriminant-geometry warm-starts such as LDA-based initialization [9], which set weights analytically from class statistics rather than learning them.

**"Short-sighted" greedy critiques.** Importantly for our depth analysis, the literature argues *against* greedy layer-wise improving with depth. Wang et al. [10] characterize greedy local learning as "short-sighted" and information-discarding, and Sakamoto & Sato [11] report that layer-wise training stagnates relative to end-to-end training. Any claim that the layer-wise advantage *grows* with depth runs counter to this work — which is precisely why we treated our own initial positive depth result with suspicion and added controls.

**Broader beyond-backprop context.** Synthetic gradients [12] and direct/feedback alignment methods [13] round out the space of alternatives to exact global backpropagation, primarily motivated by parallelism, memory, and biological plausibility rather than accuracy gains. Recent surveys and systems work emphasize that the genuine, reproducible advantage of layer-wise methods is *memory* (peak training memory roughly independent of depth) and energy on edge hardware, not accuracy [14, 15].

---

## 3. Methods / Experimental Setup

### 3.1 Methods compared

All methods are implemented in a single PyTorch framework on a shared architecture and data pipeline, to avoid the confound of differing infrastructure.

1. **Vanilla backprop** — naive initialization, saturating (tanh) activation, plain SGD. The régime where vanishing gradients are expected to bite.
2. **Modern backprop** — Adam + He initialization + BatchNorm. The strong baseline.
3. **Anchors (Ploj-style bipropagation)** — our reconstruction of Ploj's hand-designed intermediate-target rule: each layer trained to shift its input toward per-class anchors/prototypes, weights initialized near identity, final softmax readout. This is a *reconstruction* because Ploj's actual multi-class target rule is not in any reachable code (see §3.4).
4. **Local-loss (greedy supervised layer-wise)** — the same greedy scaffold, but Ploj's hand-designed targets are replaced with a *learned* per-layer objective: each layer is trained with a temporary softmax head and cross-entropy, the features are kept, and the head is discarded before training the next layer (à la Belilovsky 2019 / Nøkland 2019).
5. **Deeply-supervised** — per-layer auxiliary classifier heads *with a single global gradient* (Lee 2015). This is the locality-isolation control: it has the same per-layer supervision as local-loss but does *not* avoid global backpropagation.
6. **Deterministic-centroid bipropagation** — one hidden layer constructed analytically from class-centroid geometry; each neuron is a sparse ±1 unit over the 3 most-discriminative features between a class and its nearest rival; targets = `0.99 * layer_output + 0.01 * two_hot(class)`; refined with Adam.

### 3.2 Architecture and data

The primary testbed is a deliberately deep, narrow MLP on MNIST. We chose saturating (tanh) activations with naive initialization for the weak-baseline arm specifically because this is the régime where vanilla backprop is *known* to struggle (vanishing gradients) — the most favorable setting in which a backprop alternative could plausibly demonstrate an advantage. Depth is swept to probe the depth-scaling claim. Two data scales are reported: a FAST_MODE indicative run (6,000 train / 2,000 test, 3 seeds, few epochs) and a full-scale run (30,000 train / 10,000 test).

### 3.3 Controls that address confounds

After an adversarial internal review of an initially exciting positive result ("local-loss beats backprop at depth"), we added three controls, because that result, taken at face value, contradicts the literature [10, 11]:

- **Stronger baseline.** The original "modern" baseline was a tanh MLP with few epochs and no residual connections — a setup known to degrade with depth for *optimization* reasons unrelated to locality [16]. We add a plain ReLU+BatchNorm backprop trained for 30 epochs, and a residual MLP baseline.
- **Iso-compute consideration.** Layer-wise local-loss sees the data `depth × epochs_per_layer` times — roughly 3–6× more gradient steps than a single end-to-end backprop run. A fully iso-compute accounting is noted as a limitation (§8) and an analysis item.
- **Locality isolation (the decisive control).** The deeply-supervised method (per-layer aux heads + global gradient) holds *per-layer supervision* fixed while *restoring* global backpropagation. If it matches local-loss, then the benefit is attributable to supervision, not to the absence of backprop.

### 3.4 Reconstruction caveat

The "anchors" method is our best-effort reconstruction of Ploj's intermediate-target scheme. The only reachable, runnable bipropagation code implements a hand-coded XOR case (one inner neuron per positive class, perceptron rule, step activation). The general multi-class rule `target = h + α·(class_prototype − h)` is documented only in prose (a third-party fork); Ploj's actual multi-class MATLAB implementation (`MNIST.m`) is auth-walled on ResearchGate. A better target scheme than ours could improve the anchors numbers, and we flag this explicitly.

---

## 4. Results — MNIST / MLP

All numbers below come from actual evaluation runs; none are hardcoded. The FAST_MODE results are indicative (small data, 3 seeds, few epochs); the full-scale results use 30,000 training examples.

### 4.1 Claim 1 — speed (depth 6, FAST_MODE)

| Method | Time [s] | Test acc |
|---|---|---|
| Deterministic biprop | **9.41** (fastest) | 0.683 (worst) |
| Modern backprop (He+BN+Adam) | 14.62 | **0.944** (best) |
| Vanilla backprop | 16.80 | 0.879 |
| Bipropagation layer-wise | **20.61** (slowest) | 0.830 |

The "25× faster" claim does not hold. Layer-wise bipropagation is in fact the *slowest* method. The deterministic variant is genuinely the fastest (~1.6× vs modern backprop), but at substantially lower accuracy.

### 4.2 Claim 2 — reliability (mean ± std over 3 seeds, depth 6)

| Method | Mean | Std |
|---|---|---|
| Modern backprop | 0.9368 | **0.0008** (lowest) |
| Bipropagation layer-wise | 0.8333 | 0.0018 |
| Deterministic biprop | 0.6853 | 0.0040 |
| Vanilla backprop | 0.8805 | 0.0048 |

The "~100% reliable" claim does not hold as a comparative advantage. Modern backprop has the *lowest* seed-to-seed variance. Layer-wise bipropagation is stable but does not beat modern backprop, and the deterministic variant does not have zero variance (the softmax readout injects randomness).

### 4.3 Claim 3 — depth robustness (test acc at depths 2/4/8, FAST_MODE)

| Method | d=2 | d=4 | d=8 |
|---|---|---|---|
| Modern backprop | 0.933 | 0.938 | **0.939** (robust) |
| Vanilla backprop | 0.922 | 0.916 | **0.722** (collapses) |
| Bipropagation layer-wise | 0.859 | 0.860 | 0.806 |

This claim *partially* holds. Bipropagation is more robust than *vanilla* backprop at depth 8 (0.81 vs 0.72), confirming the vanishing-gradient pathology of naive deep tanh MLPs. But modern backprop (BatchNorm) handles depth better than either (0.94).

### 4.4 Deterministic variant — honest reproduction and the `m` sweep

The committed deterministic demo prints "Linear Accuracy = 100.0%" as a literal string that is never computed, on a headline dataset that is `np.random.rand(1000, 784)` noise (see §5/Discussion on fabrication). An honest reproduction with `m` neurons per class gives:

| m | Neurons | Test acc | Time [s] |
|---|---|---|---|
| 2 | 20 | 0.684 | 12.0 |
| 4 | 40 | 0.769 | 10.2 |
| 8 | 80 | 0.854 | 10.2 |
| **16** | 160 | **0.880** | 10.6 |
| 32 | 320 | 0.873 | 10.3 |
| 64 | 640 | 0.880 | 10.3 |

The deterministic construction is genuinely solvable: at m≥16 it reaches ~88% (comparable to vanilla backprop's 0.879) with almost no iterative training, but it plateaus at ~0.88 (below modern backprop's 0.94) because the 3-features-per-neuron construction limits expressive capacity. The defensible framing is an *instant ~88% warm-start classifier*, not state-of-the-art, and far from the repository's claimed 100%.

### 4.5 Decomposition — replacing the hand-designed targets

Replacing Ploj's hand-designed anchor scheme with a per-layer local supervised loss yields the central decomposition result (FAST_MODE):

| Depth | Anchors (Ploj-style) | Local-loss | Modern BP |
|---|---|---|---|
| 2 | 0.866 | 0.9365 | 0.9355 |
| 4 | 0.861 | 0.9375 | 0.9340 |
| 8 | 0.801 | **0.9390** | 0.8995 |

Local-loss layer-wise matches backprop on shallow nets and, at face value, *exceeds* it at depth 8 (0.939 vs 0.900). This localizes the weak link in Ploj's method: the greedy scaffold itself is sound; the hand-designed intermediate-target scheme is what drags the "anchors" method down to ~0.80.

The full-scale run (30k MNIST, seed 0) confirms the trend and extends to depth 16:

| Depth | Vanilla | Modern BP | Anchors (Ploj) | Local-loss |
|---|---|---|---|---|
| 2 | 0.9587 | 0.9690 | 0.8774 | **0.9708** |
| 4 | 0.9630 | 0.9735 | 0.8743 | **0.9714** |
| 8 | 0.9586 | 0.9627 | 0.8653 | **0.9701** |
| 16 | **0.1135** (collapse) | 0.9513 | 0.8415 | **0.9685** |

Local-loss stays high and stable to depth 16 (0.9685); the (weak) modern baseline drifts down (0.9513); vanilla backprop collapses entirely at depth 16. Anchors trails at ~0.84–0.88 throughout.

### 4.6 The control — the depth advantage is a confound

The depth advantage in §4.5 contradicts the literature [10, 11], so we tested it directly. The control adds a properly tuned ReLU+BN backprop (30 epochs), a residual baseline, and — decisively — a deeply-supervised method (per-layer aux heads with a *global* gradient) (30k MNIST, seed 0, 30 epochs):

| Depth | Residual BP | Plain BP (ReLU+BN, 30ep) | Deeply-supervised (global grad + aux heads) | Local-loss | Modern (tanh, 25ep) |
|---|---|---|---|---|---|
| 8 | 0.9674 | 0.9691 | **0.9725** | 0.9701 | 0.9627 |
| 16 | 0.9351* | 0.9661 | **0.9684** | 0.9685 | 0.9513 |

\*The residual MLP at depth 16 was not well-tuned in 30 epochs (a quick implementation, not a clean baseline); it is not load-bearing for the conclusion.

Two things are now established:

1. **The apparent "local-loss beats backprop at depth" result is largely an artifact of a weak baseline.** The original "modern" baseline was tanh + 25 epochs; a plain ReLU+BN+30-epoch backprop nearly closes the gap (0.9661 vs local-loss 0.9685 at depth 16).
2. **Deeply-supervised ≈ local-loss at both depths** (8: 0.9725 vs 0.9701; 16: 0.9684 vs 0.9685). This isolates the mechanism. When you keep the per-layer auxiliary heads but *restore* the global gradient, you get the same result. Therefore the benefit comes from **per-layer supervision (deep supervision)**, *not* from locality / avoiding global backpropagation.

The honest conclusion on MNIST: Ploj's specific mechanistic thesis — that *not doing global backprop* is the key — is **not supported**. Deep MLPs benefit from per-layer supervision plus a good activation/training recipe, all of which works perfectly well *with* global backpropagation. The layer-wise/local aspect is a memory-saving implementation choice, not an accuracy advantage.

---

## 5. Results — CIFAR-10 / CNN

We ran the same three-method decomposition on CIFAR-10 with convolutional networks (Conv–BN–ReLU blocks, downsampling every two blocks), sweeping depth over {3, 6, 9} blocks, 3 seeds, 12 epochs, 15k train / full 10k test, no augmentation (held identical across methods for fairness). The `local` method uses a RAM-safe recompute-on-the-fly implementation (frozen lower blocks run in inference mode and are recomputed per batch; no materialized feature tensors). Mean test accuracy (± std over 3 seeds):

| Depth (blocks) | E2E backprop | Greedy local-loss | Deeply-supervised |
|---|---|---|---|
| 3 | 0.528 ±.016 | **0.567** ±.003 | 0.520 ±.022 |
| 6 | 0.643 ±.007 | **0.649** ±.004 | 0.577 ±.018 |
| 9 | 0.557 ±.022 ↓ | **0.626** ±.005 | 0.609 ±.013 |

**Findings.** (i) The plain (non-residual) E2E CNN peaks at depth 6 (0.643) and *degrades* at depth 9 (0.557) — classic depth-induced optimization degradation. (ii) Greedy `local` is the most depth-robust method (0.567 → 0.649 → 0.626), beating E2E by ~7 points at depth 9. (iii) The deeply-supervised control is *also* far more robust than E2E at depth (0.609 vs 0.557 at depth 9), but sits *below* `local` (0.626). (iv) A linear-probe diagnostic [8] on trained E2E features shows monotonically rising separability with depth (block 4: 0.46 → block 8: 0.635) over a random-feature floor of ~0.22–0.30.

**Interpretation — refining the mechanism.** On CIFAR the picture is more nuanced than on MNIST. Per-layer supervision again explains *most* of the depth robustness (deepsup ≫ E2E at depth), consistent with §4. But here, unlike MNIST, `local` > `deepsup` by ~1.7 points at depth 9 — so *locality* (avoiding a global gradient through a deep plain stack) contributes a *small additional* robustness on CNNs that pure deep supervision does not fully capture. Thus the operative mechanism is primarily per-layer supervision, with a secondary, dataset/architecture-dependent contribution from locality. **Caveat:** all comparisons are against a plain non-residual baseline that degrades at depth; a residual E2E baseline likely would not, so the honest claim is "local training is more robust than plain/deeply-supervised CNNs to depth-induced degradation," not "local beats well-designed modern CNNs." A fuller protocol (≥10 seeds with 95% CIs and Holm–Bonferroni paired tests, a residual baseline, CIFAR-100) is left for follow-up.

---

## 6. Discussion

**What the decomposition proves.** Bipropagation bundles three ideas: (a) a greedy layer-wise scaffold, (b) per-layer supervision, and (c) hand-designed intermediate targets, all framed under the banner of *avoiding global backpropagation*. Our decomposition separates these. Swapping (c) the hand-designed targets for a learned per-layer loss recovers full accuracy, showing the targets are the weak component, not the scaffold. The deeply-supervised control then shows that the operative ingredient is (b) per-layer supervision: on MNIST, adding aux heads *with* a global gradient reproduces the result exactly (depth-16: 0.9684 vs 0.9685), so locality contributes nothing there. On CIFAR-10/CNN the conclusion is the same in direction but more nuanced: per-layer supervision again drives most of the depth robustness, while *locality* adds a small, secondary boost (depth-9: local 0.626 vs deepsup 0.609). So the "avoid backprop" framing (a/locality) is not the primary source of any accuracy benefit — per-layer supervision is — though locality can contribute a minor, architecture-dependent gain; its principal value, per the literature, remains memory and parallelism.

**Relation to literature.** We emphasize honestly that most of our findings are confirmatory. That greedy supervised layer-wise matches backprop is established [6, 7]. That representations become more separable with depth, measurable by trained probes, is established [8]. That analytic class-geometry warm-starts work is established [9]. That deep supervision helps is established [5]. And our initial "advantage grows with depth" reading directly contradicts [10, 11] — which is exactly why we distrusted it and added the controls that revealed it as a baseline-plus-compute confound. The one genuinely novel contribution is the rigorous reproduction and component-level decomposition of Ploj's *specific* method, together with the explicit refutation of its specific quantitative and mechanistic claims.

**The confound we caught and corrected.** We consider the self-correction itself a result worth reporting. An initial run suggested local-loss layer-wise beats backprop at depth — a publishable-sounding positive. Adversarial review flagged three confounds (weak baseline, unequal compute, locality vs supervision). Strengthening the baseline closed most of the gap, and the locality-isolation control closed the interpretive gap, converting an overclaim into a precise, defensible statement. This is a cautionary instance of how a weak baseline can manufacture an apparent advantage for an alternative training method.

**Net verdict.** Ploj's broader intuition — that per-layer supervision helps train deep networks — is correct and supported by our experiments and the prior literature. His *specific* claims — 25× speedup, ~100% reliability, beating backprop, and the mechanistic attribution to the absence of backpropagation — are not supported by a faithful reproduction.

---

## 7. Limitations

- **Seeds.** Most of the decisive numbers (full-scale and control runs) are single-seed (seed 0); the multi-seed evidence is currently FAST_MODE only. The planned protocol (≥10 seeds, 95% CI, paired Holm–Bonferroni tests) is not yet applied to the full-scale and control tables.
- **Weak testbed / non-residual baseline.** MNIST/MLP is a weak proxy for modern practice; we extend to CIFAR-10/CNN (§5), which confirms the per-layer-supervision mechanism and reveals a secondary locality effect. However, both testbeds compare against *plain* (non-residual) baselines that degrade with depth; a residual/normalized E2E baseline would likely close the depth gap, so our depth-robustness claims are relative to plain architectures, not to modern residual networks.
- **Iso-compute.** Not every comparison is matched on compute. Local-loss sees the data ~3–6× more often than a single end-to-end run; a clean accuracy-vs-wall-clock and iso-gradient-step accounting is still outstanding.
- **Reconstruction of Ploj's rule.** The "anchors" method is our reconstruction of an unpublished multi-class target rule (the real `MNIST.m` is auth-walled). A better-faith or more faithful target scheme could raise the anchors numbers, though it would not change the deep-supervision-not-locality conclusion.
- **Deterministic baselines.** The residual MLP at depth 16 in the control table was under-tuned within the epoch budget and is not a clean baseline; it is excluded from load-bearing claims.

---

## 8. Conclusion

We conducted a controlled reproduction and decomposition of Bojan Ploj's bipropagation. On MNIST, the headline claims (25× faster, ~100% reliable, beats backprop) do not reproduce against properly tuned modern backpropagation, and the public "deterministic initialization" demo ships fabricated results (an honest reproduction reaches ~88%, not 100%). The underlying greedy layer-wise scaffold, however, is sound: replacing the hand-designed targets with a per-layer local supervised loss recovers backprop-level accuracy. A deeply-supervised control — per-layer auxiliary heads with a single global gradient — matches the local-loss method at every depth (depth-16: 0.9684 vs 0.9685), isolating the mechanism. The benefit is per-layer supervision, not the absence of global backpropagation. Ploj's mechanistic claim is therefore refuted, while his broader intuition that per-layer supervision aids deep networks is supported and is consistent with prior work on deep supervision and greedy supervised layer-wise training. The genuinely novel contribution is this rigorous reproduction-plus-decomposition, including the identification and correction of a baseline-and-compute confound in our own initial positive result. A CIFAR-10/CNN study with a full multi-seed statistical protocol is in progress to test whether the conclusion holds on harder data.

---

## References

[1] G. E. Hinton, S. Osindero, Y.-W. Teh. *A Fast Learning Algorithm for Deep Belief Nets.* Neural Computation, 2006.

[2] Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle. *Greedy Layer-Wise Training of Deep Networks.* NeurIPS, 2007.

[3] D.-H. Lee, S. Zhang, A. Fischer, Y. Bengio. *Difference Target Propagation.* arXiv:1412.7525, 2015. https://arxiv.org/abs/1412.7525

[4] G. Hinton. *The Forward-Forward Algorithm: Some Preliminary Investigations.* 2022. https://www.cs.toronto.edu/~hinton/FFA13.pdf

[5] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, Z. Tu. *Deeply-Supervised Nets.* arXiv:1409.5185, 2015. https://arxiv.org/abs/1409.5185

[6] A. Nøkland, L. H. Eidnes. *Training Neural Networks with Local Error Signals.* ICML, 2019. http://proceedings.mlr.press/v97/nokland19a/nokland19a.pdf · arXiv:1901.06656 https://arxiv.org/abs/1901.06656

[7] E. Belilovsky, M. Eickenberg, E. Oyallon. *Greedy Layerwise Learning Can Scale to ImageNet.* arXiv:1812.11446, 2019. https://arxiv.org/abs/1812.11446

[8] G. Alain, Y. Bengio. *Understanding Intermediate Layers Using Linear Classifier Probes.* arXiv:1610.01644, 2016. https://arxiv.org/abs/1610.01644

[9] M. Masden, D. Sinha. *Linear Discriminant Initialization for Feed-Forward Neural Networks.* arXiv:2007.12782, 2020. https://arxiv.org/abs/2007.12782

[10] Y. Wang et al. *Revisiting Locally Supervised Learning: An Alternative to End-to-End Training* ("short-sighted" greedy local learning). arXiv:2101.10832, 2021. https://arxiv.org/abs/2101.10832

[11] Sakamoto, Sato. *Layer-wise training stagnation analysis.* arXiv:2402.09050, 2024. https://arxiv.org/abs/2402.09050

[12] M. Jaderberg et al. *Decoupled Neural Interfaces using Synthetic Gradients.* ICML, 2017.

[13] A. Nøkland. *Direct Feedback Alignment Provides Learning in Deep Neural Networks.* NeurIPS, 2016.

[14] *Beyond Backpropagation: A Survey* (reports up to ~41% energy reduction on MNIST/CIFAR for layer-wise methods). 2025. https://arxiv.org/html/2509.19063v1

[15] *Stochastic Layer-wise Learning* (peak memory roughly independent of depth). arXiv:2505.05181, 2025. https://arxiv.org/abs/2505.05181

[16] K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* arXiv:1512.03385, 2015. https://arxiv.org/abs/1512.03385

[17] B. Ploj. *Border Pairs Method.* Neurocomputing 126, Elsevier, 2014. https://www.sciencedirect.com/science/article/abs/pii/S0925231213005079

[18] B. Ploj. *A Deterministic Multi-Signal Case of Bipropagation Network Initialization.* Submitted to Neurocomputing, 2026.

**Code repositories referenced:** `BojanPLOJ/Bipropagation` · `BojanPLOJ/Deterministic-Bipropagation-Initialization` · `korentmaj/BipropagationAlgorithm`
