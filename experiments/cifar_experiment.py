# =============================================================================
# CIFAR-10 + CNN decomposition study of "bipropagation" (greedy layer-wise).
# Three methods on the SAME per-depth architecture:
#   e2e     : end-to-end backprop (strong baseline)
#   local   : greedy local-loss layer-wise (bipropagation scaffold)
#   deepsup : deeply-supervised control (per-block aux heads, GLOBAL gradient)
# Single self-contained TF2/Keras script. Runs in ONE Colab cell (T4).
# FAST_MODE (smoke) and FULL configs. Reports mean +/- std test acc per (method,depth).
# Fairness: augmentation OFF for all methods (clean controlled comparison).
# =============================================================================
import os, time, random
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers
import matplotlib.pyplot as plt

print("TensorFlow:", tf.__version__)

FAST_MODE = False  # <-- set False for the FULL run

if FAST_MODE:
    CFG = dict(n_train=8000, n_test=10000, seeds=[0, 1], epochs=5,
               depths=[3, 6], batch_size=128, do_probe=False)
else:
    CFG = dict(n_train=15000, n_test=10000, seeds=[0, 1, 2], epochs=12,
               depths=[3, 6, 9], batch_size=128, do_probe=True)

LR = 1e-3
NUM_CLASSES = 10
PROBE_HELDOUT = 4000

print("=" * 78)
print("CONFIG (%s):" % ("FAST_MODE" if FAST_MODE else "FULL"))
for k, v in CFG.items():
    print("  %-12s = %s" % (k, v))
print("  - Train uses a %d-sample SUBSET; full 10k test for all reported acc." % CFG["n_train"])
print("  - Augmentation OFF for ALL methods (fair controlled comparison).")
print("=" * 78)


def set_seed(*, seed):
    tf.keras.backend.clear_session()  # free graph memory between training runs (RAM safety)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def load_cifar(*, n_train, n_test, seed=12345):
    (x_tr_all, y_tr_all), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
    x_tr_all = x_tr_all.astype("float32") / 255.0
    x_te = x_te.astype("float32") / 255.0
    y_tr_all = y_tr_all.astype("int64").reshape(-1)
    y_te = y_te.astype("int64").reshape(-1)
    mean = x_tr_all.mean(axis=(0, 1, 2), keepdims=True)
    std = x_tr_all.std(axis=(0, 1, 2), keepdims=True) + 1e-7
    x_tr_all = (x_tr_all - mean) / std
    x_te = (x_te - mean) / std
    rng = np.random.default_rng(seed)
    n_train = int(n_train); n_test = int(n_test)
    assert 0 < n_train <= x_tr_all.shape[0]
    assert 0 < n_test <= x_te.shape[0]
    perm = rng.permutation(x_tr_all.shape[0])
    tr_idx = perm[:n_train]
    return (x_tr_all[tr_idx], y_tr_all[tr_idx]), (x_te[:n_test], y_te[:n_test])


def make_train_ds(*, x, y, batch_size, seed):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(min(len(x), 10000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_eval_ds(*, x, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def filter_schedule(*, depth):
    base = [64, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256]
    assert depth >= 1
    return [min(f, 256) for f in base[:depth]]


def make_conv_block(*, filters, stride, name):
    assert filters > 0
    return [
        layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False,
                      kernel_initializer=initializers.HeNormal(), name="%s_conv" % name),
        layers.BatchNormalization(name="%s_bn" % name),
        layers.Activation("relu", name="%s_relu" % name),
    ]


def block_strides(*, depth):
    strides, n_down = [], 0
    for i in range(depth):
        if (i % 2 == 1) and (n_down < 3):
            strides.append(2); n_down += 1
        else:
            strides.append(1)
    return strides


def build_blocks(*, depth):
    f = filter_schedule(depth=depth); s = block_strides(depth=depth)
    return [make_conv_block(filters=f[i], stride=s[i], name="b%d" % i) for i in range(depth)]


def apply_block(*, x, block, training=None):
    for lyr in block:
        x = lyr(x, training=training) if isinstance(lyr, layers.BatchNormalization) else lyr(x)
    return x


# --- METHOD 1: end-to-end backprop ---
def train_e2e(*, depth, seed, x_tr, y_tr, x_te, y_te, epochs, batch_size):
    set_seed(seed=seed)
    inp = layers.Input(shape=(32, 32, 3)); x = inp
    for block in build_blocks(depth=depth):
        x = apply_block(x=x, block=block)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax",
                       kernel_initializer=initializers.HeNormal())(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(learning_rate=LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(make_train_ds(x=x_tr, y=y_tr, batch_size=batch_size, seed=seed),
              epochs=epochs, verbose=0)
    _, acc = model.evaluate(make_eval_ds(x=x_te, y=y_te, batch_size=batch_size), verbose=0)
    return float(acc), model


# --- METHOD 2: greedy local-loss layer-wise ---
def _forward_inference(*, model, x, batch_size):
    outs = []
    for start in range(0, x.shape[0], batch_size):
        yb = model(x[start:start + batch_size], training=False)
        outs.append(np.asarray(yb))
    return np.concatenate(outs, axis=0)


def train_local(*, depth, seed, x_tr, y_tr, x_te, y_te, epochs, batch_size):
    # RAM-safe greedy layer-wise: lower blocks are FROZEN (trainable=False => BN inference
    # mode) and recomputed on-the-fly per batch via tf.data. No materialized feature arrays.
    set_seed(seed=seed)  # clears session once at the start
    blocks = build_blocks(depth=depth)
    for i in range(depth):
        inp = layers.Input(shape=(32, 32, 3))
        x = inp
        for j in range(i):
            x = apply_block(x=x, block=blocks[j])   # blocks[0..i-1] already trainable=False
        x = apply_block(x=x, block=blocks[i])        # block i trains
        g = layers.GlobalAveragePooling2D()(x)
        head = layers.Dense(NUM_CLASSES, activation="softmax",
                            kernel_initializer=initializers.HeNormal())(g)
        m = models.Model(inp, head)
        for j in range(i):                           # ensure lower blocks are frozen
            for lyr in blocks[j]:
                lyr.trainable = False
        m.compile(optimizer=optimizers.Adam(learning_rate=LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        m.fit(make_train_ds(x=x_tr, y=y_tr, batch_size=batch_size, seed=seed + i),
              epochs=epochs, verbose=0)
        for lyr in blocks[i]:                         # freeze block i for the next stage
            lyr.trainable = False
    # Final readout on the full frozen stack (all blocks trainable=False -> inference mode).
    inp = layers.Input(shape=(32, 32, 3))
    x = inp
    for j in range(depth):
        x = apply_block(x=x, block=blocks[j])
    g = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax",
                       kernel_initializer=initializers.HeNormal())(g)
    readout = models.Model(inp, out)
    readout.compile(optimizer=optimizers.Adam(learning_rate=LR),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    readout.fit(make_train_ds(x=x_tr, y=y_tr, batch_size=batch_size, seed=seed + 7),
                epochs=epochs, verbose=0)
    _, acc = readout.evaluate(make_eval_ds(x=x_te, y=y_te, batch_size=batch_size), verbose=0)
    return float(acc), readout


# --- METHOD 3: deeply-supervised control (global gradient, aux head per block) ---
def train_deepsup(*, depth, seed, x_tr, y_tr, x_te, y_te, epochs, batch_size):
    set_seed(seed=seed)
    inp = layers.Input(shape=(32, 32, 3)); x = inp
    outputs = []
    for j, block in enumerate(build_blocks(depth=depth)):
        x = apply_block(x=x, block=block)
        g = layers.GlobalAveragePooling2D(name="aux_gap_%d" % j)(x)
        outputs.append(layers.Dense(NUM_CLASSES, activation="softmax",
                                    kernel_initializer=initializers.HeNormal(),
                                    name="aux_head_%d" % j)(g))
    model = models.Model(inp, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=LR),
                  loss=["sparse_categorical_crossentropy"] * depth,
                  loss_weights=[1.0] * depth)

    def _replicate(img, lbl):
        return img, tuple([lbl] * depth)

    tr_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
             .shuffle(min(len(x_tr), 10000), seed=seed, reshuffle_each_iteration=True)
             .map(_replicate, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    model.fit(tr_ds, epochs=epochs, verbose=0)
    preds = model.predict(make_eval_ds(x=x_te, y=y_te, batch_size=batch_size), verbose=0)
    last_pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
    return float((np.argmax(last_pred, axis=1) == y_te).mean()), model


# --- OPTIONAL: linear-probe separability diagnostic ---
def gap_features_per_block(*, depth, x, batch_size, trained_blocks=None, seed=0):
    if trained_blocks is None:
        set_seed(seed=seed); blocks = build_blocks(depth=depth)
    else:
        blocks = trained_blocks
    inp = layers.Input(shape=(32, 32, 3)); h = inp; gap_outs = []
    for block in blocks:
        h = apply_block(x=h, block=block, training=False)
        gap_outs.append(layers.GlobalAveragePooling2D()(h))
    feat_model = models.Model(inp, gap_outs)
    per_block = [[] for _ in range(depth)]
    for start in range(0, x.shape[0], batch_size):
        outs = feat_model(x[start:start + batch_size], training=False)
        if depth == 1:
            outs = [outs]
        for bi in range(depth):
            per_block[bi].append(np.asarray(outs[bi]))
    return [np.concatenate(p, axis=0) for p in per_block]


def linear_probe(*, feats_tr, y_tr, feats_te, y_te, seed, epochs=30, batch_size=256):
    set_seed(seed=seed)
    inp = layers.Input(shape=(feats_tr.shape[1],))
    out = layers.Dense(NUM_CLASSES, activation="softmax",
                       kernel_initializer=initializers.GlorotUniform())(inp)
    probe = models.Model(inp, out)
    probe.compile(optimizer=optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    probe.fit(feats_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)
    _, acc = probe.evaluate(feats_te, y_te, verbose=0)
    return float(acc)


def _extract_blocks_from_model(*, model, depth):
    convs, bns, acts = [], [], []
    for lyr in model.layers:
        if isinstance(lyr, layers.Conv2D):
            convs.append(lyr)
        elif isinstance(lyr, layers.BatchNormalization):
            bns.append(lyr)
        elif isinstance(lyr, layers.Activation):
            acts.append(lyr)
    if not (len(convs) == len(bns) == len(acts) == depth):
        return None
    return [[convs[i], bns[i], acts[i]] for i in range(depth)]


def main():
    t0 = time.time()
    (x_tr_pool, y_tr_pool), (x_te, y_te) = load_cifar(n_train=CFG["n_train"], n_test=CFG["n_test"])
    if CFG["do_probe"] and x_tr_pool.shape[0] > PROBE_HELDOUT + 1000:
        x_probe, y_probe = x_tr_pool[:PROBE_HELDOUT], y_tr_pool[:PROBE_HELDOUT]
        x_tr, y_tr = x_tr_pool[PROBE_HELDOUT:], y_tr_pool[PROBE_HELDOUT:]
    else:
        x_probe = y_probe = None
        x_tr, y_tr = x_tr_pool, y_tr_pool
    print("Train: %d | Test: %d | Probe heldout: %s" %
          (x_tr.shape[0], x_te.shape[0], "none" if x_probe is None else x_probe.shape[0]))

    methods = {"e2e": train_e2e, "local": train_local, "deepsup": train_deepsup}
    res = {m: {d: [] for d in CFG["depths"]} for m in methods}

    for depth in CFG["depths"]:
        f = filter_schedule(depth=depth); s = block_strides(depth=depth)
        print("\n" + "-" * 78)
        print("DEPTH=%d | filters=%s | strides=%s" % (depth, f, s))
        for mname, mfn in methods.items():
            for seed in CFG["seeds"]:
                ts = time.time()
                acc, _ = mfn(depth=depth, seed=seed, x_tr=x_tr, y_tr=y_tr,
                             x_te=x_te, y_te=y_te, epochs=CFG["epochs"],
                             batch_size=CFG["batch_size"])
                res[mname][depth].append(acc)
                print("  [depth=%d %-7s seed=%d] test_acc=%.4f  (%.1fs)" %
                      (depth, mname, seed, acc, time.time() - ts))
            arr = np.array(res[mname][depth])
            print("RESULT depth=%d method=%s mean=%.4f std=%.4f n=%d" %
                  (depth, mname, arr.mean(), arr.std(), len(arr)))

    try:
        plt.figure(figsize=(7, 5))
        for mname in methods:
            means = [np.mean(res[mname][d]) for d in CFG["depths"]]
            stds = [np.std(res[mname][d]) for d in CFG["depths"]]
            plt.errorbar(CFG["depths"], means, yerr=stds, marker="o", capsize=4, label=mname)
        plt.xlabel("CNN depth (# conv blocks)"); plt.ylabel("CIFAR-10 test accuracy")
        plt.title("Test accuracy vs depth (%s)" % ("FAST" if FAST_MODE else "FULL"))
        plt.xticks(CFG["depths"]); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plot skipped:", repr(e))

    if CFG["do_probe"] and x_probe is not None:
        try:
            deepest = max(CFG["depths"]); seed0 = CFG["seeds"][0]
            print("\n" + "=" * 78)
            print("LINEAR-PROBE DIAGNOSTIC (depth=%d, seed=%d)" % (deepest, seed0))
            cut = x_probe.shape[0] // 2
            xp_tr, yp_tr = x_probe[:cut], y_probe[:cut]
            xp_te, yp_te = x_probe[cut:], y_probe[cut:]
            rtr = gap_features_per_block(depth=deepest, x=xp_tr, batch_size=CFG["batch_size"], seed=999)
            rte = gap_features_per_block(depth=deepest, x=xp_te, batch_size=CFG["batch_size"], seed=999)
            print("Random-feature floor (untrained blocks):")
            for bi in range(deepest):
                a = linear_probe(feats_tr=rtr[bi], y_tr=yp_tr, feats_te=rte[bi], y_te=yp_te, seed=seed0)
                print("  PROBE random block=%d acc=%.4f" % (bi, a))
            _, e2e_model = train_e2e(depth=deepest, seed=seed0, x_tr=x_tr, y_tr=y_tr,
                                     x_te=x_te, y_te=y_te, epochs=CFG["epochs"],
                                     batch_size=CFG["batch_size"])
            blk = _extract_blocks_from_model(model=e2e_model, depth=deepest)
            if blk is not None:
                etr = gap_features_per_block(depth=deepest, x=xp_tr, batch_size=CFG["batch_size"], trained_blocks=blk)
                ete = gap_features_per_block(depth=deepest, x=xp_te, batch_size=CFG["batch_size"], trained_blocks=blk)
                print("Trained e2e features:")
                for bi in range(deepest):
                    a = linear_probe(feats_tr=etr[bi], y_tr=yp_tr, feats_te=ete[bi], y_te=yp_te, seed=seed0)
                    print("  PROBE e2e block=%d acc=%.4f" % (bi, a))
        except Exception as e:
            print("Probe skipped:", repr(e))

    print("\n" + "=" * 78)
    print("TOTAL WALL TIME: %.1f min" % ((time.time() - t0) / 60.0))
    printable = {m: {d: [round(a, 4) for a in res[m][d]] for d in res[m]} for m in res}
    print("FINAL res[method][depth]:", printable)
    print("CIFAR_DONE")
    return res


_ = main()
