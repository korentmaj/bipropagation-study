R = _   # res dict returned by main()
import numpy as _np
for m in R:
    for d in sorted(R[m]):
        a = _np.array(R[m][d])
        print('SUM %-8s d=%d mean=%.4f std=%.4f n=%d' % (m, d, a.mean(), a.std(), len(a)))
print('SUMMARY_DONE')
