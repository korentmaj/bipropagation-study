import numpy as _np
print('C1_acc', {k:round(v['test_acc'],4) for k,v in dict(van=r_van,mod=r_mod,bip=r_bip,det=r_det).items()})
print('C1_time_s', {k:round(v['train_time'],2) for k,v in dict(van=r_van,mod=r_mod,bip=r_bip,det=r_det).items()})
print('C2_mean_std', {k:(round(_np.mean(v),4), round(_np.std(v),4)) for k,v in acc_by.items()})
