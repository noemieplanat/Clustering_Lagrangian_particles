import os
import time
import multiprocessing.dummy as mp
import glob
from joblib import Parallel, delayed 
from Function_adv_pararelle_no_dr import run_adv_journalier_sensitivity




#--------------------
n_coeurs = 20 # number of processors
#--------------------
#hb = 20, zb = 20, tb = 10
Liste_set_up = [(15, 20, 10, None, 1993), (20, 20, 10, None, 1993), (25, 20, 10, None, 1993),(30, 20, 10, None, 1993), (35, 20, 10, None, 1993), (40, 20, 10, None, 1993), (20, 10, 10, None, 1993), (20, 15, 10, None, 1993), (20, 20, 10, None, 1993), (20, 25, 10, None, 1993), (20, 30, 10, None, 1993), (20, 10, 1, None, 1993),(20, 10, 10, None, 1993),(20, 10, 20, None, 1993),(20, 10, 30, None, 1993),(20, 10, 45, None, 1993),(20, 10, 60, None, 1993),(20, 10, 120, None, 1993),(20, 10, 360, None, 1993)]
#Liste_set_up = [(10, 20, 10, None, 1993)]
print('Starting')
tic = time.time()
res = Parallel(n_jobs=n_coeurs)(delayed(run_adv_journalier_sensitivity)(x) for x in Liste_set_up[:])
print(time.time()-tic)

