# POLNI-SCALE RUN: skaliranje z globino na 30k MNIST, vse 4 metode (vkljucno z local-loss).
import numpy as np, time
(_fx, _fy), (_fxt, _fyt) = tf.keras.datasets.mnist.load_data()
_rng = np.random.default_rng(0)
_itr = _rng.permutation(len(_fx))[:30000]
X_TR = (_fx.reshape(-1, 784)[_itr] / 255.).astype('float32'); Y_TR = _fy[_itr]
X_TE = (_fxt.reshape(-1, 784) / 255.).astype('float32'); Y_TE = _fyt
CFG['bp_epochs'] = 25
CFG['biprop_epochs_per_layer'] = 12
print('FULL data:', X_TR.shape, X_TE.shape)

DEPTHS = [2, 4, 8, 16]
res = {'vanilla': [], 'modern': [], 'anchors': [], 'localloss': []}
for d in DEPTHS:
    res['vanilla'].append(train_backprop(d, 'vanilla', 0)['test_acc'])
    res['modern'].append(train_backprop(d, 'modern', 0)['test_acc'])
    res['anchors'].append(train_bipropagation_layerwise(d, seed=0)['test_acc'])
    res['localloss'].append(train_biprop_localloss(d, seed=0)['test_acc'])
    print('depth', d, {k: round(v[-1], 4) for k, v in res.items()})

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))
for k, mk in [('vanilla', 'o'), ('modern', 's'), ('anchors', '^'), ('localloss', 'D')]:
    plt.plot(DEPTHS, res[k], marker=mk, label=k)
plt.xlabel('st. skritih plasti'); plt.ylabel('test accuracy (30k MNIST)')
plt.title('Polni-scale: skaliranje z globino'); plt.legend(); plt.grid(alpha=.3); plt.show()
print('FULLRUN_DONE', res)
