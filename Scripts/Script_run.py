# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:16:30 2022

@author: Noemie Planat and Mathilde Jutras
contact: mathilde.jutras@mail.mcgill.ca
license: GNU General Public License v3.0

Please acknowledge using the following DOI:

This is the script used to perform the unsupervised clustering analysis
on geospatial Lagrangian trajectories detailed in Jutras, Planat & Dufour (2022).

#############################
### First script to run.
#############################

This portion comprises the pre-processing of the data, as well as
the first steps of the clustering algorithm.
After having run this portion, you will need to run
the script titled Script_supercomputer.py
"""

from All_functions_ML import Part_1_pre_processing, extract, split_sets, Save_PCA, Part_2_PCA, Part_2_Load_PCA, Part_3_kmeans_clustering,\
    Part_3_save_clustering, apply_to_all_data, get_traj_sub, random_sample

from Config import norm, t, resamp, length_days, wtype, kernel, degreek,   n_components, copy, whiten, svd_solver,\
    Dataset_path, perctest, perctrain, Run_PCA_Local, path, name_PCA_config, path_save_PCA, path_save_clustering, n_clusters, name_Clustering_config,\
    ntest, init, max_iter, algorithm, n_split, verbose, tol, sample_weight, nmb_initialisations, files, init_year,\
    final_year, delta_year, N_particles
from All_functions_plotting import load_data
import os

if not os.path.exists(path +'/'+name_Clustering_config):
    os.makedirs(path+'/'+name_Clustering_config)

print('1/4-Pre_processing------------------------------------')
lats_all,lons_all, temps_all, sals_all = extract(Dataset_path, length_days)
lats_test, lons_test, lats_train, lons_train, lats_valid, lons_valid, temps_train, temps_test, temps_valid, sals_train, sals_test, sals_valid = split_sets(lats_all, lons_all,temps_all, sals_all, perctest, perctrain)


X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w_train, w_valid, w_test = Part_1_pre_processing(norm, t, resamp, length_days, wtype, kernel, degreek, lats_valid, lons_valid, lats_train, lons_train, lats_test, lons_test,  n_components, copy, whiten, svd_solver)
    
print('2/4-PCA------------------------------------')
if Run_PCA_Local: 
    if not os.path.exists(path+'/'+name_PCA_config+'/'):
        os.makedirs(path+'/'+name_PCA_config+'/')
        X_reduced_train, X_reduced_valid, X_reduced_test, a, l, dc = Part_2_PCA(X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w_train, w_valid, w_test,  n_components, copy, whiten, svd_solver, kernel, degreek)
        Save_PCA(path, name_PCA_config,X_reduced_train, X_reduced_valid, X_reduced_test, a, l, dc )
    else:
        print('This PCA already exists')
        
if Run_PCA_Local != True:
    print('Do the commented lines remotly')
    print('Copy the saved files into', path+name_PCA_config+'/')

X_reduced_train, X_reduced_valid, X_reduced_test =  Part_2_Load_PCA(ntest, norm, t, resamp, length_days, wtype,n_clusters, kernel, lats_valid, lons_valid, lats_train, lons_train, path_save_clustering, name_PCA_config, lats_test, lons_test, path_save_PCA)
  


print('3/4-Clustering------------------------------------')
if not os.path.exists(path_save_clustering+'sals_train'+ntest+'.npy'):
    labels_valid, labels_test, X_reduced_valid, X_reduced_test, centroids, a_temp,X_centers_train =Part_3_kmeans_clustering(X_reduced_train, X_reduced_valid, X_reduced_test, n_split, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight)
    
    Part_3_save_clustering(lats_test, lons_test, lats_valid, lons_valid, temps_valid, sals_valid, sals_test, temps_test, lats_train, lons_train, temps_train, sals_train, ntest, path_save_clustering, labels_valid, labels_test, X_reduced_valid, X_reduced_test, centroids, a_temp,X_centers_train)
    
else:
    print('This CLUSTERING already exists')
    centroids, labels_test, labels_valid, lats_test, lats_valid, lons_test, lons_valid, lons_train, lats_train, \
        X_reduced_test, X_Reduced_valid, temps_test, temps_train, temps_valid, sals_test, sals_train, sals_valid, X_centers_train = load_data(path_save_clustering, ntest)


