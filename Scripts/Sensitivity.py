import os
import time
import multiprocessing.dummy as mp
import glob
from joblib import Parallel, delayed 
from Function_adv_pararelle_no_dr import run_adv_journalier_sensitivity




#--------------------
n_coeurs = 16 # number of processors
init = 159 # set up to start with
#--------------------
#hb = 20, zb = 20, tb = 10
#Liste_set_up = [(15, 20, 10, None, 1993), (20, 20, 10, None, 1993), (25, 20, 10, None, 1993),(30, 20, 10, None, 1993), (35, 20, 10, None, 1993), (40, 20, 10, None, 1993), (20, 10, 10, None, 1993), (20, 15, 10, None, 1993), (20, 20, 10, None, 1993), (20, 25, 10, None, 1993), (20, 30, 10, None, 1993), (20, 10, 1, None, 1993),(20, 10, 10, None, 1993),(20, 10, 20, None, 1993),(20, 10, 30, None, 1993),(20, 10, 45, None, 1993),(20, 10, 60, None, 1993),(20, 10, 120, None, 1993),(20, 10, 360, None, 1993)]
#Liste_set_up = [(10, 20, 10, None, 1993)]
#Liste_set_up = [(30, 20, 1, None, 1994), (30, 20, 5, None, 1994), (30, 20, 10, None, 1994),(30, 20, 15, None, 1994),(30, 20, 3, None, 1994),(30, 20, 7, None, 1994)]#(25, 20, 30, None, 1994), (30, 20, 30, None, 1994), (35, 20, 30, None, 1994), (40, 20, 30, None, 1994), (30, 15, 30, None, 1994), (30, 25, 30, None, 1994), (30, 30, 30, None, 1994), (30, 20, 20, None, 1994), (30, 20, 25, None, 1994), (30, 20, 35, None, 1994), (30, 20, 40, None, 1994), (30, 20, 45, None, 1994)]

Liste_set_up = []
for h in range(5, 45, 5):
    for z in range(10, 90, 10):
        for t in [1, 5, 10, 15, 20, 30, 60, 120, 180, 360]:
            Liste_set_up.append((h,z,t,None, 1993))
                
print('Starting')
tic = time.time()
res = Parallel(n_jobs=n_coeurs)(delayed(run_adv_journalier_sensitivity)(x) for x in Liste_set_up[init:])
print(time.time()-tic)

