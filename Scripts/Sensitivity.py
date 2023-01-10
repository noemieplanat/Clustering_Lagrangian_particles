import os
import time
import multiprocessing.dummy as mp
import glob
from joblib import Parallel, delayed 
from Function_adv_pararelle import run_adv_journalier_sensitivity




#--------------------
n_coeurs = 5 # number of processors
#--------------------
Liste_set_up = []
Liste_h = [20,30, 40, 50]
Liste_z = [40, 60, 80, 100]
Liste_dt = [60, 30, 10,1]
Liste_dr = [None]
Liste_yr = [1993]

Liste_set_up = []
for h in Liste_h:
    for z in Liste_z:
        for t in Liste_dt:
            for r in Liste_dr:
                for y in Liste_yr:
                    Liste_set_up.append((h,z,t,r,y))
Liste_set_up = [(40, 20, 1, None, 1993), (50, 20, 1, None, 1993), (20, 50, 30, None, 1993), (20, 50, 10, None, 1993), (20, 50, 1, None, 1993)]                    
print('Starting')
tic = time.time()
res = Parallel(n_jobs=n_coeurs)(delayed(run_adv_journalier_sensitivity)(x) for x in Liste_set_up[:])
print(time.time()-tic)

