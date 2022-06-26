# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:56:17 2022

@author: noemi
"""

from All_functions_ML import Part_1_pre_processing,  Part_2_PCA,Part_3_kmeans_clustering,\
    apply_to_all_data, Part_0_make_subset

from Config import norm, t, resamp, length_days, wtype, kernel, degreek,   n_components, copy, whiten, svd_solver,\
    Dataset_path, perctest, perctrain, Run_PCA_Local,  name_PCA_config, path_save_PCA, path_save_clustering, n_clusters,\
    ntest, init, max_iter, algorithm, n_split, verbose, tol, sample_weight, nmb_initialisations, files, init_year,\
    final_year, delta_year, N_particles, start_lon
import os


print('0/4-Generate sub-set for clustering------------------------------------')
if not os.path.exists(Dataset_path):
    Part_0_make_subset(Dataset_path, init_year,final_year, delta_year, files, length_days, N_particles)
else:
    print('This subset already exists')

print('1/4-Pre_processing------------------------------------')
X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w_train, w_valid, w_test =  Part_1_pre_processing(Dataset_path, length_days, perctest, perctrain, norm, t, resamp,wtype. start_lon)
    
print('2/4-PCA------------------------------------')
if Run_PCA_Local: 
    if not os.path.exists(path_save_PCA+'/'):
        os.makedirs(path_save_PCA+'/')
        Part_2_PCA(path_save_PCA, name_PCA_config, X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w_train, n_components, copy, whiten, svd_solver, kernel, degreek)
    else:
        print('This PCA already exists')
        
if ~ Run_PCA_Local:
    print('Do the commented lines remotly')
    print('To be implemented')


print('3/4-Clustering------------------------------------')
if not os.path.exists(path_save_clustering+'centroids'+ntest+'.npy'):
    Part_3_kmeans_clustering(name_PCA_config, path_save_PCA, n_split, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight, ntest, path_save_clustering)
    
else:
    print('This CLUSTERING already exists')


print('4/4-Apply to all data------------------------------------')
os.makedirs(path_save_clustering+'ClassifiedData/')
apply_to_all_data(path_save_clustering, ntest, init_year,final_year, length_days, files, start_lon,  norm, t, resamp,wtype, name_PCA_config, path_save_PCA)