# Generator for the Bipropagation comparison Colab notebook.
# Assembles cells into a valid .ipynb. Run: python _build_notebook.py
import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})

# ----------------------------------------------------------------------
md(r"""# Bipropagation vs Backpropagation, pošten test treh trditev

**Avtor metode:** dr. Bojan Ploj. **Okvir:** TensorFlow 2 / Keras. **Cilj:** Google Colab (GPU).

Bipropagacija (greedy, plast-po-plast, nadzorovano učenje z eksplicitnimi vmesnimi cilji in
identiteti-blizu / deterministično inicializacijo) ima tri ločljive trditve. Vsako testiramo
**izolirano in pošteno**, vsaka številka pride iz dejanske evalvacije (originalni demo jih ni računal):

| # | Trditev | Strukturni razlog za (ne)zmago | Test |
|---|---------|-------------------------------|------|
| 1 | **"25× hitreje"** | Det./plastna init je že na epohi 0 blizu rešitve | accuracy-vs-čas + epochs-to-target |
| 2 | **"~100% zanesljivo"** | Deterministična init → varianca čez seede ≈ 0 | box-plot čez N seedov |
| 3 | **Globina** | Plastno učenje obide izginjajoče gradiente | metrika vs globina mreže |
| ★ | **(bonus) malo podatkov** | Centroid-init gradi klasifikator iz povprečij razredov | accuracy vs vzorcev/razred |

> **Pomembno / poštenost:** Plojev multi-class pravilo za vmesne cilje ni v javni kodi (MNIST.m je auth-walled),
> zato je `bipropagation_layerwise` spodaj **moja zvesta rekonstrukcija** dokumentiranega pravila
> (uteži blizu identitete + per-razred premik cilja). `deterministic_bipropagation` pa je **zvesta
> reprodukcija dejanske kode** iz repozitorija `Deterministic-Bipropagation-Initialization`,
> le da accuracy **dejansko izračunamo** (njihov demo je vračal hardcoded "100%").
""")

# ----------------------------------------------------------------------
md(r"""## 0. Setup
`FAST_MODE = True` požene hiter smoke-test (podmnožica MNIST, malo epoh/seedov), za preverbo da vse teče.
Za pravi benchmark na GPU nastavi `FAST_MODE = False`.""")

code(r"""
import time, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, initializers, optimizers

print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices('GPU'))

FAST_MODE = True   # <-- nastavi False za polni benchmark na GPU

# Globalni konfig (FAST vs FULL)
CFG = dict(
    n_train   = 6000  if FAST_MODE else 60000,
    n_test    = 2000  if FAST_MODE else 10000,
    seeds     = [0,1,2] if FAST_MODE else list(range(10)),
    bp_epochs = 15    if FAST_MODE else 60,
    biprop_epochs_per_layer = 8 if FAST_MODE else 30,
    det_refine_epochs = 200 if FAST_MODE else 1000,
)

def set_seed(s):
    np.random.seed(s); tf.random.set_seed(s)
""")

# ----------------------------------------------------------------------
md(r"""## 1. Podatki, MNIST (in 2D igrača za intuicijo)""")

code(r"""
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
x_tr = (x_tr.reshape(-1,784).astype('float32'))/255.0
x_te = (x_te.reshape(-1,784).astype('float32'))/255.0
# fiksna podmnožica (stratifikacija ni nujna za demo, shuffle je dovolj)
rng = np.random.default_rng(0)
itr = rng.permutation(len(x_tr))[:CFG['n_train']]
ite = rng.permutation(len(x_te))[:CFG['n_test']]
X_TR, Y_TR = x_tr[itr], y_tr[itr]
X_TE, Y_TE = x_te[ite], y_te[ite]
K = 10; M = 784
print("MNIST:", X_TR.shape, X_TE.shape)
""")

# ----------------------------------------------------------------------
md(r"""## 2. Metode

Štiri metode pod enotnim API-jem. Vse vrnejo dict z `test_acc`, `train_time`, in po možnosti
`acc_curve` (seznam (cas, test_acc)) za "race" graf.

### 2a. Backprop baselines (vanilla = naivna init, brez BN; modern = He init + BatchNorm + Adam)
Uporabljamo **tanh** aktivacijo (saturira → stresira vanilla backprop pri globini).""")

code(r"""
class TimeAcc(tf.keras.callbacks.Callback):
    # zabelezi (kumulativni cas, val_accuracy) po vsaki epohi
    def __init__(self, Xte, Yte): self.Xte, self.Yte = Xte, Yte; self.curve=[]; self.t0=None
    def on_train_begin(self, logs=None): self.t0=time.time()
    def on_epoch_end(self, epoch, logs=None):
        acc = self.model.evaluate(self.Xte, self.Yte, verbose=0)[1]
        self.curve.append((time.time()-self.t0, acc))

def build_mlp(depth, width=128, kind='vanilla'):
    init = initializers.HeNormal() if kind=='modern' else initializers.RandomNormal(stddev=0.05)
    m = models.Sequential([layers.Input((M,))])
    for _ in range(depth):
        m.add(layers.Dense(width, kernel_initializer=init, use_bias=True))
        if kind=='modern': m.add(layers.BatchNormalization())
        m.add(layers.Activation('tanh'))
    m.add(layers.Dense(K, activation='softmax',
                       kernel_initializer=init if kind=='modern' else initializers.RandomNormal(stddev=0.05)))
    opt = optimizers.Adam(1e-3) if kind=='modern' else optimizers.SGD(0.05)
    m.compile(opt, 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def train_backprop(depth, kind, seed, epochs=None):
    set_seed(seed)
    epochs = epochs or CFG['bp_epochs']
    m = build_mlp(depth, kind=kind)
    cb = TimeAcc(X_TE, Y_TE)
    t0=time.time()
    m.fit(X_TR, Y_TR, epochs=epochs, batch_size=128, verbose=0, callbacks=[cb])
    tt=time.time()-t0
    acc = m.evaluate(X_TE, Y_TE, verbose=0)[1]
    return dict(test_acc=acc, train_time=tt, acc_curve=cb.curve, model=m)
""")

# ----------------------------------------------------------------------
md(r"""### 2b. Bipropagation, plastna (deep), zvesta rekonstrukcija
Vsako plast naučimo posebej (prejšnje zamrznjene), da preslika svoj vhod proti **per-razred sidru**
v skritem prostoru (one-hot v prvih K dimenzijah). Skrite plasti so kvadratne, init = **identiteta + šum**,
tako da vsaka naredi le majhen residualni premik. Na koncu plitek softmax readout. To **obide globalni
backprop** in s tem izginjajoče gradiente.""")

code(r"""
def class_anchors(width, seed=0):
    # sidra: e_y v prvih K dimenzijah skritega prostora (preprosto, interpretabilno)
    A = np.zeros((K, width), 'float32')
    for c in range(K): A[c, c % width] = 1.0
    return A

def train_bipropagation_layerwise(depth, width=128, seed=0, epochs_per_layer=None, alpha=0.5):
    set_seed(seed)
    epochs_per_layer = epochs_per_layer or CFG['biprop_epochs_per_layer']
    A = class_anchors(width, seed)
    anchors_tr = A[Y_TR]                      # cilj v skritem prostoru
    t0 = time.time(); curve=[]
    frozen = []                               # seznam ze naucenih plasti (W,b)
    # ----- plast 1: vhod M -> skriti width, cilj = sidro razreda -----
    H = X_TR.copy()
    in_dim = M
    for L in range(depth):
        inp = layers.Input((in_dim,))
        if L==0:
            dense = layers.Dense(width, kernel_initializer=initializers.RandomNormal(stddev=0.05))
        else:
            # identiteta + sum  (in_dim==width)
            Wid = np.eye(width, dtype='float32') + np.random.normal(0,0.01,(width,width)).astype('float32')
            dense = layers.Dense(width, kernel_initializer=initializers.Constant(Wid),
                                 bias_initializer='zeros')
        out = layers.Activation('tanh')(dense(inp))
        layer_model = models.Model(inp, out)
        # cilj te plasti: premik trenutne reprezentacije proti sidru
        if L==0:
            target = anchors_tr
        else:
            target = (1-alpha)*H + alpha*anchors_tr
        layer_model.compile(optimizers.Adam(1e-2), 'mse')
        layer_model.fit(H, target, epochs=epochs_per_layer, batch_size=128, verbose=0)
        H = layer_model.predict(H, batch_size=256, verbose=0)     # nova reprezentacija
        frozen.append(layer_model)
        in_dim = width
        # groba tocka krivulje: oceni z nearest-anchor na testu
        curve.append((time.time()-t0, _eval_layerwise(frozen, None, A)))
    # ----- readout: softmax na koncni reprezentaciji -----
    readout = models.Sequential([layers.Input((width,)),
                                 layers.Dense(K, activation='softmax')])
    readout.compile(optimizers.Adam(1e-2),'sparse_categorical_crossentropy',metrics=['accuracy'])
    readout.fit(H, Y_TR, epochs=max(10,epochs_per_layer), batch_size=128, verbose=0)
    tt=time.time()-t0
    acc=_eval_layerwise(frozen, readout, A)
    curve.append((tt, acc))
    return dict(test_acc=acc, train_time=tt, acc_curve=curve)

def _eval_layerwise(frozen, readout, A):
    H = X_TE
    for lm in frozen: H = lm.predict(H, batch_size=512, verbose=0)
    if readout is None:
        # nearest-anchor klasifikacija (groba ocena med gradnjo)
        d = ((H[:,None,:]-A[None,:,:])**2).sum(-1)
        pred = d.argmin(1)
    else:
        pred = readout.predict(H, batch_size=512, verbose=0).argmax(1)
    return float((pred==Y_TE).mean())
""")

# ----------------------------------------------------------------------
md(r"""### 2c. Deterministic Bipropagation, zvesta reprodukcija novega repozitorija
Analitična konstrukcija ene skrite plasti iz geometrije centroidov (±1 uteži nad 3 najbolj
diskriminativnimi značilkami na nevron), cilji = `0.99*izhod + 0.01*two_hot`, refine z Adam.
**Accuracy izračunamo zares** (njihov demo je vračal hardcoded 100%).""")

code(r"""
def _construct_layer(Xn, y, m=2):
    Kk=len(np.unique(y)); Mm=Xn.shape[1]; num=m*Kk
    W=np.zeros((num,Mm),'float32'); b=np.zeros(num,'float32')
    gmu=Xn.mean(0)
    cmu=np.stack([Xn[y==c].mean(0) for c in range(Kk)])
    idx=0
    for c in range(Kk):
        d=np.linalg.norm(cmu-cmu[c],axis=1); d[c]=np.inf; riv=d.argmin()
        order=np.argsort(-np.abs(cmu[c]-cmu[riv]))
        for nf in range(m):
            a =order[(nf*3)   % Mm]; bb=order[(nf*3+1)% Mm]; ct=order[(nf*3+2)% Mm]
            x = 1.0 if cmu[c,bb]>cmu[riv,bb] else -1.0
            z = 1.0 if cmu[c,ct]>cmu[riv,ct] else -1.0
            if cmu[c,a]<gmu[a]: W[idx,a]=-1.0; b[idx]=1.0
            else:               W[idx,a]= 1.0
            W[idx,bb]=x; W[idx,ct]=z; idx+=1
    return W,b,num

def train_deterministic_bipropagation(seed=0, m=2, refine_epochs=None):
    set_seed(seed)
    refine_epochs = refine_epochs or CFG['det_refine_epochs']
    # min-max norm na [0,1] (kot v repu)
    xmin=X_TR.min(0); xmax=X_TR.max(0)
    Xn=(X_TR-xmin)/(xmax-xmin+1e-8); Xte_n=(X_TE-xmin)/(xmax-xmin+1e-8)
    t0=time.time()
    W,b,num=_construct_layer(Xn,Y_TR,m)
    smart=Xn@W.T+b
    two=np.zeros((len(Y_TR),num),'float32')
    for i,c in enumerate(Y_TR): two[i,2*c]=1.0; two[i,2*c+1]=1.0
    targets=0.99*smart+0.01*two
    Wp=tf.Variable(W.T); Bp=tf.Variable(b); opt=optimizers.Adam(1e-2)
    Xn_t=tf.constant(Xn); T_t=tf.constant(targets.astype('float32'))
    for _ in range(refine_epochs):
        with tf.GradientTape() as g:
            out=tf.matmul(Xn_t,Wp)+Bp; loss=tf.reduce_mean((out-T_t)**2)
        gr=g.gradient(loss,[Wp,Bp]); opt.apply_gradients(zip(gr,[Wp,Bp]))
    # PRAVA accuracy: plitek softmax readout na konstruiranih znacilkah
    feat_tr=(Xn@Wp.numpy()+Bp.numpy()).astype('float32')
    feat_te=(Xte_n@Wp.numpy()+Bp.numpy()).astype('float32')
    ro=models.Sequential([layers.Input((num,)),layers.Dense(K,activation='softmax')])
    ro.compile(optimizers.Adam(1e-2),'sparse_categorical_crossentropy',metrics=['accuracy'])
    ro.fit(feat_tr,Y_TR,epochs=30,batch_size=128,verbose=0)
    tt=time.time()-t0
    acc=ro.evaluate(feat_te,Y_TE,verbose=0)[1]
    return dict(test_acc=acc, train_time=tt, num_neurons=num)
""")

# ----------------------------------------------------------------------
md(r"""## TEST, Trditev 1: hitrost konvergence (accuracy vs čas)
Globina = 6. Gledamo, kdo prej doseže visoko natančnost. Pričakovano: deterministična/plastna
metoda starta visoko, vanilla backprop pleza počasi.""")

code(r"""
DEPTH=6
r_van = train_backprop(DEPTH,'vanilla',seed=0)
r_mod = train_backprop(DEPTH,'modern', seed=0)
r_bip = train_bipropagation_layerwise(DEPTH, seed=0)
r_det = train_deterministic_bipropagation(seed=0)

plt.figure(figsize=(8,5))
for r,lab in [(r_van,'Backprop (vanilla)'),(r_mod,'Backprop (modern: He+BN+Adam)'),
              (r_bip,'Bipropagation (layer-wise)')]:
    c=np.array(r['acc_curve']); plt.plot(c[:,0],c[:,1],marker='o',label=lab)
plt.axhline(r_det['test_acc'],ls='--',color='gray',
            label=f"Deterministic biprop (1 plast): {r_det['test_acc']:.3f}")
plt.xlabel('cas treniranja [s]'); plt.ylabel('test accuracy'); plt.title(f'Konvergenca (globina={DEPTH})')
plt.legend(); plt.grid(alpha=.3); plt.show()
print({k:round(v['test_acc'],4) for k,v in
       dict(vanilla=r_van,modern=r_mod,biprop=r_bip,deterministic=r_det).items()})
print({k:round(v['train_time'],2) for k,v in
       dict(vanilla=r_van,modern=r_mod,biprop=r_bip,deterministic=r_det).items()})
""")

# ----------------------------------------------------------------------
md(r"""## TEST, Trditev 2: zanesljivost (varianca čez seede)
Deterministična init bi morala imeti skoraj nično varianco. Backprop niha.""")

code(r"""
acc_by={'vanilla':[],'modern':[],'biprop':[],'deterministic':[]}
for s in CFG['seeds']:
    acc_by['vanilla'].append(train_backprop(DEPTH,'vanilla',s)['test_acc'])
    acc_by['modern'].append(train_backprop(DEPTH,'modern', s)['test_acc'])
    acc_by['biprop'].append(train_bipropagation_layerwise(DEPTH,seed=s)['test_acc'])
    acc_by['deterministic'].append(train_deterministic_bipropagation(seed=s)['test_acc'])

plt.figure(figsize=(8,5))
plt.boxplot(acc_by.values(), labels=acc_by.keys(), showmeans=True)
plt.ylabel('test accuracy'); plt.title(f'Zanesljivost cez {len(CFG["seeds"])} seedov'); plt.grid(alpha=.3); plt.show()
for k,v in acc_by.items(): print(f"{k:14s} mean={np.mean(v):.4f}  std={np.std(v):.4f}")
""")

# ----------------------------------------------------------------------
md(r"""## TEST, Trditev 3: skaliranje z globino
Vanilla backprop s tanh + naivno init naj bi z globino propadal; plastna bipropagacija ne.""")

code(r"""
DEPTHS=[2,4,8,16] if not FAST_MODE else [2,4,8]
res={'vanilla':[],'modern':[],'biprop':[]}
for d in DEPTHS:
    res['vanilla'].append(train_backprop(d,'vanilla',0)['test_acc'])
    res['modern' ].append(train_backprop(d,'modern', 0)['test_acc'])
    res['biprop' ].append(train_bipropagation_layerwise(d,seed=0)['test_acc'])

plt.figure(figsize=(8,5))
for k,mk in [('vanilla','o'),('modern','s'),('biprop','^')]:
    plt.plot(DEPTHS,res[k],marker=mk,label=k)
plt.xlabel('st. skritih plasti'); plt.ylabel('test accuracy')
plt.title('Skaliranje z globino'); plt.legend(); plt.grid(alpha=.3); plt.show()
print(res)
""")

# ----------------------------------------------------------------------
md(r"""## TEST ★ Bonus: učinkovitost s podatki (low-data režim)
Centroid-init gradi iz povprečij razredov → naj bi zdržal z malo vzorci. Sweep vzorcev/razred.""")

code(r"""
def subset_per_class(n_per):
    idx=[]
    for c in range(K):
        ci=np.where(Y_TR==c)[0][:n_per]; idx+=list(ci)
    idx=np.array(idx); return idx

NPC=[5,20,100] if FAST_MODE else [5,20,100,500]
det_acc=[]; bp_acc=[]
_XTR,_YTR=X_TR.copy(),Y_TR.copy()
for n in NPC:
    idx=subset_per_class(n)
    globals()['X_TR'],globals()['Y_TR']=_XTR[idx],_YTR[idx]
    det_acc.append(train_deterministic_bipropagation(seed=0,refine_epochs=200)['test_acc'])
    bp_acc.append(train_backprop(4,'modern',0,epochs=40)['test_acc'])
globals()['X_TR'],globals()['Y_TR']=_XTR,_YTR
plt.figure(figsize=(8,5))
plt.plot(NPC,det_acc,'o-',label='Deterministic biprop')
plt.plot(NPC,bp_acc,'s-',label='Backprop (modern, 4 plasti)')
plt.xscale('log'); plt.xlabel('vzorcev na razred'); plt.ylabel('test accuracy')
plt.title('Ucinkovitost s podatki'); plt.legend(); plt.grid(alpha=.3); plt.show()
print('det',det_acc); print('bp',bp_acc)
""")

# ----------------------------------------------------------------------
md(r"""## Povzetek / verdikt
Ko poženeš, izpolni tabelo s PRAVIMI številkami iz zgornjih testov:

| Trditev | Rezultat (FAST) | Drži? |
|---------|-----------------|-------|
| 1. hitrost | _epochs/čas do cilja_ | … |
| 2. zanesljivost (std) | _std det vs backprop_ | … |
| 3. globina | _acc@depth16 biprop vs vanilla_ | … |
| ★ malo podatkov | _acc@5/razred det vs bp_ | … |

**Naslednji koraki za pravi benchmark:** `FAST_MODE=False`, dodaj CIFAR-10 (zamenjaj loader),
ablacija (centroid-init vs random; ročni cilj vs naučen), in wall-clock na GPU.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

with open("Bipropagation_Comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote Bipropagation_Comparison.ipynb with", len(cells), "cells")
