# BOLJSA SHEMA: lokalna nadzorovana izguba na plast (greedy supervised layer-wise).
# Vsako plast naucimo z zacasno softmax glavo (cross-entropy), obdrzimo znacilke, glavo zavrzemo.
def train_biprop_localloss(depth, width=128, seed=0, epochs_per_layer=None):
    set_seed(seed)
    epl = epochs_per_layer or CFG['biprop_epochs_per_layer']
    H = X_TR.astype('float32'); Hte = X_TE.astype('float32')
    t0 = time.time(); in_dim = H.shape[1]
    for L in range(depth):
        inp = layers.Input((in_dim,))
        feat = layers.Dense(width, activation='tanh', kernel_initializer=initializers.HeNormal())(inp)
        out = layers.Dense(K, activation='softmax')(feat)
        m = models.Model(inp, out)
        m.compile(optimizers.Adam(1e-3), 'sparse_categorical_crossentropy', metrics=['accuracy'])
        m.fit(H, Y_TR, epochs=epl, batch_size=128, verbose=0)
        feat_model = models.Model(inp, feat)
        H = feat_model.predict(H, batch_size=512, verbose=0)
        Hte = feat_model.predict(Hte, batch_size=512, verbose=0)
        in_dim = width
    ro = models.Sequential([layers.Input((width,)), layers.Dense(K, activation='softmax')])
    ro.compile(optimizers.Adam(1e-3), 'sparse_categorical_crossentropy', metrics=['accuracy'])
    ro.fit(H, Y_TR, epochs=max(10, epl), batch_size=128, verbose=0)
    tt = time.time() - t0
    acc = ro.evaluate(Hte, Y_TE, verbose=0)[1]
    return dict(test_acc=acc, train_time=tt)

# primerjava po globini: stara shema (sidra) vs nova (local-loss) vs modern backprop
print('depth | biprop_anchors | biprop_localloss | modern_BP')
for d in [2, 4, 8]:
    a_anchor = train_bipropagation_layerwise(d, seed=0)['test_acc']
    a_local  = train_biprop_localloss(d, seed=0)['test_acc']
    a_bp     = train_backprop(d, 'modern', 0)['test_acc']
    print(f'  {d:2d}  |     {a_anchor:.4f}     |      {a_local:.4f}      |  {a_bp:.4f}')
print('LOCALLOSS_DONE')
