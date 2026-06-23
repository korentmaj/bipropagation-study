# KONTROLE (naslavljajo adversarni pregled): mocni baseline + locality-isolation.
# Podatki so ze nalozeni kot 30k (iz polnega runa). res = rezultati polnega runa.
DEPTHS_FULL = [2, 4, 8, 16]
for i, d in enumerate(DEPTHS_FULL):
    print('FULL d=', d, {k: round(res[k][i], 4) for k in res})

# 1) MOCNI baseline: residual ReLU + BN MLP (manjkajoci posteni nasprotnik)
def build_resmlp(depth, width=128, residual=True):
    inp = layers.Input((M,))
    h = layers.Activation('relu')(layers.BatchNormalization()(
        layers.Dense(width, kernel_initializer=initializers.HeNormal())(inp)))
    for _ in range(depth - 1):
        y = layers.Activation('relu')(layers.BatchNormalization()(
            layers.Dense(width, kernel_initializer=initializers.HeNormal())(h)))
        h = layers.add([h, y]) if residual else y
    out = layers.Dense(K, activation='softmax')(h)
    m = models.Model(inp, out)
    m.compile(optimizers.Adam(1e-3), 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

def train_resmlp(depth, residual, seed, epochs):
    set_seed(seed); m = build_resmlp(depth, residual=residual)
    m.fit(X_TR, Y_TR, epochs=epochs, batch_size=128, verbose=0)
    return m.evaluate(X_TE, Y_TE, verbose=0)[1]

# 2) LOCALITY-ISOLATION: deeply-supervised (aux glava na vsaki plasti, a GLOBALNI gradient).
#    Ce to ~ local-loss, potem delo opravijo glave, ne lokalnost.
def train_deeply_supervised(depth, width=128, seed=0, epochs=30):
    set_seed(seed)
    inp = layers.Input((M,)); h = inp; outs = []
    for L in range(depth):
        h = layers.Dense(width, activation='tanh', kernel_initializer=initializers.HeNormal())(h)
        outs.append(layers.Dense(K, activation='softmax', name='aux%d' % L)(h))
    m = models.Model(inp, outs)
    m.compile(optimizers.Adam(1e-3), ['sparse_categorical_crossentropy'] * depth)
    m.fit(X_TR, [Y_TR] * depth, epochs=epochs, batch_size=128, verbose=0)
    preds = m.predict(X_TE, batch_size=512, verbose=0)
    return float((preds[-1].argmax(1) == Y_TE).mean())

print('--- KONTROLE (30 epoch, 30k MNIST, seed 0) ---')
for d in [8, 16]:
    idx = DEPTHS_FULL.index(d)
    a_res = train_resmlp(d, True, 0, 30)
    a_resno = train_resmlp(d, False, 0, 30)
    a_deep = train_deeply_supervised(d, 0, 30)
    print('depth', d,
          '| residual_BP=%.4f' % a_res,
          '| plain_BP(30ep)=%.4f' % a_resno,
          '| deeplySup=%.4f' % a_deep,
          '|| localloss=%.4f' % res['localloss'][idx],
          'modern=%.4f' % res['modern'][idx])
print('CONTROL_DONE')
