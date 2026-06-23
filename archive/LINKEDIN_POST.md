# LinkedIn Post — Bipropagation Study

## Main version (~250 words)

I spent the last few weeks independently reproducing and decomposing a less-known neural-network training method — and the most interesting result wasn't the one I expected. 🔬

The method is "bipropagation," a greedy, layer-wise *supervised* training scheme proposed by Dr. Bojan Ploj: instead of one global backpropagation pass, you train each layer on its own toward an intermediate target. It comes with bold claims (25× faster than backprop, ~100% reliable). I wanted to know what actually holds.

So I rebuilt it from scratch in PyTorch and ran controlled experiments — MNIST/MLP and CIFAR-10/CNN — comparing four things on the same architecture and data pipeline: end-to-end backprop, greedy local-loss layer-wise training, and a deeply-supervised control (per-layer heads, but a single global gradient).

The honest findings:

➡️ The headline numbers didn't reproduce against a properly tuned modern backprop baseline. But the *core idea* is sound.

➡️ The real mechanism behind depth-robustness is PER-LAYER SUPERVISION — not the absence of backpropagation. The deeply-supervised control matches greedy local training almost exactly (MNIST depth-16: 0.9684 vs 0.9685).

➡️ Layer-wise/local training is genuinely more robust to depth on plain CNNs (CIFAR-10 depth-9: local ~0.63 vs end-to-end ~0.56).

The lesson I keep coming back to: controls and a strong baseline turned an exciting-but-wrong result into a precise, defensible one. Nuanced/negative results are valuable.

Full credit to Dr. Ploj for the original intuition — per-layer supervision really does help deep nets.

Where I'd love collaborators: residual baselines, more seeds + confidence intervals, harder datasets, and other local-learning methods. Repo + write-up in the comments. 👇

#MachineLearning #DeepLearning #Reproducibility #NeuralNetworks #Research

---

## Short version (~80 words)

I independently reproduced and decomposed "bipropagation" — a greedy layer-wise supervised training method by Dr. Bojan Ploj. 🔬

The bold claims (25× faster, ~100% reliable) didn't hold against a strong modern backprop baseline — but the core idea is real. My clean takeaway: the benefit comes from PER-LAYER SUPERVISION, not from avoiding backprop. A deeply-supervised control matches greedy local training exactly.

Negative/nuanced results matter. Repo + write-up in comments — collaborators welcome. 👇

#MachineLearning #DeepLearning #Reproducibility #Research
