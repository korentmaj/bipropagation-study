# Popravljen klic (funkcije ze definirane v kernelu): seed kot keyword.
print('--- KONTROLE (30 epoch, 30k MNIST, seed 0) ---')
for d in [8, 16]:
    idx = DEPTHS_FULL.index(d)
    a_res = train_resmlp(d, True, 0, 30)
    a_resno = train_resmlp(d, False, 0, 30)
    a_deep = train_deeply_supervised(d, seed=0, epochs=30)
    print('depth', d,
          '| residual_BP=%.4f' % a_res,
          '| plain_BP30=%.4f' % a_resno,
          '| deeplySup=%.4f' % a_deep,
          '|| localloss=%.4f' % res['localloss'][idx],
          'modern=%.4f' % res['modern'][idx])
print('CONTROL_DONE')
