# SWEEP m: ali vec nevronov/razred dvigne natancnost deterministicne plasti?
sweep = []
for mm in [2, 4, 8, 16, 32, 64]:
    r = train_deterministic_bipropagation(seed=0, m=mm, refine_epochs=300)
    sweep.append((mm, r['num_neurons'], round(r['test_acc'], 4), round(r['train_time'], 2)))
    print('m=', mm, '| neuroni=', r['num_neurons'], '| acc=', round(r['test_acc'], 4), '| cas=', round(r['train_time'], 2), 's')
print('SWEEP_DONE', sweep)
