# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:36:17 2022

@author: noemie
"""

import xarray as xr
import numpy as np
#import pandas as pd
import os
import random
#import pyproj as proj
from sklearn.decomposition import PCA
import copy as cp
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn import cluster
import wpca


def remove_short(ds): 
    l = [len(ds.sel(traj=i).dropna(dim='obs').obs) for i in ds.traj]
    keep = np.array([i for i in range(len(l)) if l[i] > 90]) # more than 3 months
    ds2 = ds.sel(traj=xr.DataArray(keep)).rename_dims({'dim_0':'traj'}) 
    return ds2

def get_traj_sub(year_init, year_final, delta_year, files, length_days):
    dslist = []
    for yr in range(year_init,year_final+1,delta_year):
        print(yr)
        filesS = files[np.where(np.array([int(files[i][-7:-3]) for i in range(len(files))]) == yr)[0][0]]
        dsl = xr.open_dataset(filesS)
        dsl = remove_short(dsl) # remove short trajectories -- to avoid memory errors
        dsl = dsl.sel(obs = np.arange(length_days+366))
        dslist.append(dsl)

    ds = xr.concat(dslist, dim='traj')
    return ds

def random_sample(N_particles, ds):
    random.seed(4)
    sample = random.sample(list(ds.traj.values), k=N_particles) 
    ds_samp = ds.sel(traj=xr.DataArray(sample)).rename_dims({'dim_0':'traj'})  # select the random particles
    return ds_samp

def Part_0_make_subset(Dataset_path, init_year,final_year, delta_year, files, length_days, N_particles):
    ds = get_traj_sub(init_year, final_year, delta_year, files, length_days)
    ds_samp = random_sample(N_particles, ds)
    ds_samp.to_netcdf(Dataset_path)
    return None    

def lons_in_interval(start_lon, lons):
    #this should shift all longitudes so it lies in start_lon + 360. 
    if start_lon<0 : 
        print('start_lons should be in [0-360[')
        return None
    else:
        lons_shifted = lons
        while np.any(lons_shifted<0):
            lons_shifted[lons_shifted<0] = lons_shifted[lons_shifted<0]+360
        lons_2 = cp.copy(lons_shifted)
        lons_2[lons_shifted>=start_lon] = lons_shifted[lons_shifted>=start_lon] - start_lon
        lons_2[lons_shifted<start_lon] = lons_shifted[lons_shifted<start_lon] - start_lon +360
        return cp.copy(lons_2)

def extract(path, length_days, start_lon) :
    ds = xr.open_dataset(path)
    ds = remove_short(ds)
    lats = ds.lat[:,0:length_days].values
    lons_i = ds.lon[:,0:length_days].values
    lons = lons_in_interval(start_lon, lons_i)
    temps = ds.temperature[:,0:length_days].values
    sal = ds.salinity[:,0:length_days].values
    date = ds.time[:,0].values
    return lats, lons, temps, sal, date


def extract_all_per_year(yr, length_days, all_filles, start_lon) :
    file = all_filles[np.where(np.array([int(all_filles[i][-7:-3]) for i in range(len(all_filles))]) == yr)[0][0]]
    lats, lons, temps, sal, date = extract(file, length_days, start_lon)    
    return lats, lons, temps, sal, date





def translate(lat, lon):
    return lat-np.repeat(lat[:,0][:, np.newaxis], lat.shape[1], axis=1), lon-np.repeat(lon[:,0][:, np.newaxis], lon.shape[1], axis=1)

def resample(lats, lons, length_days):
    
        
    lats_resamp = np.zeros(lats.shape)
    lons_resamp = np.zeros(lats.shape)
    for i in range(lats.shape[0]):
        latsl = lats[i,:]#lats_proj ? 
        latsl = latsl[~np.isnan(latsl)]
        xp = np.linspace(0,len(latsl),length_days)
        lats_resamp[i,:] = np.interp(xp, range(len(latsl)), latsl)
        lonsl = lons[i,:] #lons_proj ? 
        lonsl = lonsl[~np.isnan(lonsl)]
        lons_resamp[i,:] = np.interp(xp, range(len(lonsl)), lonsl)    
            
       
    return lats_resamp, lons_resamp

def weightPCA(lats, wtype):
    
    if wtype == 'cos' :
        w = np.cos( np.radians(lats) )
    elif wtype == 'sqrt-cos' :
        w = np.sqrt(abs(np.cos( np.radians(lats) )))
    elif wtype == False:
        w = np.ones(lats.shape)
    return w

def PCA_Kernelized(Xlat_lon, n_components, copy, whiten, svd_solver, w, kernel, degreek):
    
    # weight PCA

    s = int(Xlat_lon.shape[1]/2)
    Xlat_lon[:,s:] = Xlat_lon[:,s:]*w
    
    pca = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver = svd_solver)
    X_reduced = pca.fit_transform(Xlat_lon)
    N_features_PCA = pca.n_features_
    N_samples_PCA = pca.n_samples_
    N_components_PCA = pca.n_components_
    Explained_variance_ratio_PCA = pca.explained_variance_ratio_
    print('The number of initial features was: ', N_features_PCA)
    print('The number of selected features is: ', N_components_PCA)
    print('The number of samples is: ', N_samples_PCA)
    print('The explained variance desired is:', n_components, '%, the obtained variance explained is: ', np.sum(Explained_variance_ratio_PCA), '%')
    
    
    pca = KernelPCA(n_components=N_components_PCA, copy_X=copy, eigen_solver=svd_solver, kernel=kernel, degree=degreek, fit_inverse_transform=True)
    X_reduced = pca.fit_transform(Xlat_lon)
    return X_reduced, pca

def Part_1_pre_processing_one_sort(t, la, lo, resamp, length_days, wtype, norm):
    if t == True :
        lats_train_pp, lons_train_pp = translate(la, lo)
    if resamp == True:
        lats_train_pp, lons_train_pp = resample(lats_train_pp, lons_train_pp, length_days)
    w = weightPCA(lats_train_pp, wtype)
    
    # Reshape
    X_lats_lons_train = np.concatenate((lats_train_pp, lons_train_pp), axis = 1) #concatenation of features to make it in the correct shape
    if resamp == False:
        X_lats_lons_train[np.isnan(X_lats_lons_train)] = 0 
        w[np.isnan(w)]=0
    
   # Normalize
    if norm == True:
        X_lats_lons_train = normalize(X_lats_lons_train, copy=True)
    return w, X_lats_lons_train




def Part_1_pre_processing(Dataset_path, length_days, perctest, perctrain, norm, t, resamp,wtype, start_lon):
    lats_all,lons_all, temps_all, sals_all, date_all = extract(Dataset_path, length_days, start_lon)
    lats_test, lons_test, lats_train, lons_train, lats_valid, lons_valid, temps_train, temps_test, temps_valid, sals_train, sals_test, sals_valid = split_sets(lats_all, lons_all,temps_all, sals_all, perctest, perctrain)

    print('Pre_processing')
    # PRE-PROCESSING
    w_train, X_lats_lons_train = Part_1_pre_processing_one_sort(t, lats_train, lons_train, resamp, length_days, wtype, norm)
    w_test, X_lats_lons_test = Part_1_pre_processing_one_sort(t, lats_test, lons_test, resamp, length_days, wtype, norm)
    w_valid, X_lats_lons_valid = Part_1_pre_processing_one_sort(t, lats_valid, lons_valid, resamp, length_days, wtype, norm)
    
    # weights, for train it's applied in PCA - kernel
    s = int(X_lats_lons_valid.shape[1]/2)
    X_lats_lons_valid[:,s:] = X_lats_lons_valid[:,s:]*w_valid
    s = int(X_lats_lons_test.shape[1]/2)
    X_lats_lons_test[:,s:] = X_lats_lons_test[:,s:]*w_test

    return X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w_train, w_valid, w_test

def Part_2_PCA(path_save_PCA, name_PCA_config, X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w_train, n_components, copy, whiten, svd_solver, kernel, degreek):
    X_reduced_train, pca= PCA_Kernelized(X_lats_lons_train, n_components, copy, whiten, svd_solver, w_train, kernel, degreek)
    X_reduced_valid = pca.transform(X_lats_lons_valid)
    X_reduced_test  = pca.transform(X_lats_lons_test)
    a = pca.alphas_
    l = pca.lambdas_
    dc = pca.dual_coef_
    Save_PCA(path_save_PCA, name_PCA_config,X_reduced_train, X_reduced_valid, X_reduced_test, a, l, dc )
    return None


def k_means_X_cv(n_splits, X_reduced, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight):

    kf = KFold(n_splits=n_splits, shuffle=True)
    avg_silouhette = 0
    i = 0
    for train_index, val_index in kf.split(X_reduced):
        print('fold nmb ', i)
        i+=1
        X_train = X_reduced[train_index,:]
        k_means = cluster.KMeans(n_clusters=n_clusters, init=init, n_init = nmb_initialisations, max_iter = max_iter, tol = tol, algorithm = algorithm, verbose = verbose)
        k_means.fit(X_train, sample_weight = sample_weight)
        X_centers = k_means.cluster_centers_
        a_temp = k_means.inertia_
        #a_temp = silhouette_score(X_val, labs_val)
        if a_temp>avg_silouhette:
            X_centered_memory = X_centers
            k_means_memory = k_means
            avg_silouhette = a_temp
    
    return X_centered_memory, k_means_memory, a_temp


def get_number_centroids(X_centers_train, X_reduced_train):
    N_clusters = X_centers_train.shape[0]
    Liste_number= np.zeros(N_clusters)
    for n in range(N_clusters):
        RV = X_centers_train[n]-X_reduced_train[:,:]            
        N_RV = np.linalg.norm(RV, axis = 1)
        Liste_number[n] = np.argmin(N_RV)
    return Liste_number

def split_sets(lats_all, lons_all, temps_all, sals_all, perctest, perctrain):
    
    s0, s1 = lats_all.shape
    data = np.concatenate((lats_all,lons_all), axis=1)
    
    random.seed(4)
    random.shuffle(data)
    random.seed(4)
    random.shuffle(temps_all)
    random.seed(4)
    random.shuffle(sals_all)

    lats_test = data[:int(perctest*s0), 0:s1]
    lons_test = data[:int(perctest*s0), s1:]

    lats_train = data[int(perctest*s0):int(perctest*s0)+int(perctrain*s0), 0:s1]
    lons_train = data[int(perctest*s0):int(perctest*s0)+int(perctrain*s0), s1:]

    lats_valid = data[int(perctest*s0)+int(perctrain*s0):, 0:s1]
    lons_valid = data[int(perctest*s0)+int(perctrain*s0):, s1:]

    temps_test = temps_all[:int(perctest*s0),:]
    sals_test = sals_all[:int(perctest*s0),:]
    
    temps_train =temps_all[int(perctest*s0):int(perctest*s0)+int(perctrain*s0),:]
    sals_train = sals_all[int(perctest*s0):int(perctest*s0)+int(perctrain*s0),:]
    
    temps_valid = temps_all[int(perctest*s0)+int(perctrain*s0):,:]
    sals_valid = sals_all[int(perctest*s0)+int(perctrain*s0):,:]
    
    return lats_test, lons_test, lats_train, lons_train, lats_valid, lons_valid, temps_train, temps_test, temps_valid, sals_train, sals_test, sals_valid


def predict(X_reduced_val, X_centers):
    
    X_reduced_val2 = np.repeat(X_reduced_val[:,:,np.newaxis], X_centers.shape[0], axis = 2)
    X_centers2 = np.repeat(np.transpose(X_centers)[np.newaxis,:,:], X_reduced_val.shape[0], axis = 0)
    
    return np.argmin(np.linalg.norm(X_reduced_val2-X_centers2, axis=1), axis = 1)

def Save_PCA(path_save_PCA,name_PCA_config, X_reduced_train, X_reduced_valid, X_reduced_test, a, l, dc ):
    np.savetxt(path_save_PCA+'/X_train_kernelpca_%s.csv'%name_PCA_config, X_reduced_train, delimiter=',')
    np.savetxt(path_save_PCA+'/X_valid_kernelpca_%s.csv'%name_PCA_config, X_reduced_valid, delimiter=',')
    np.savetxt(path_save_PCA+'/X_test_kernelpca_%s.csv'%name_PCA_config, X_reduced_test, delimiter=',')
    np.savetxt(path_save_PCA+'/alphas_kernelpca_%s.csv'%name_PCA_config, a, delimiter=',')
    np.savetxt(path_save_PCA+'/lambdas_kernelpca_%s.csv'%name_PCA_config, l, delimiter=',')
    np.savetxt(path_save_PCA+'/dualcoef_kernelpca_%s.csv'%name_PCA_config, dc, delimiter=',')

def Part_2_Load_PCA(name_kernel_pca, path_kernelpca) :   
    X_reduced_train = np.genfromtxt(path_kernelpca+'X_train_kernelpca_'+name_kernel_pca+'.csv', delimiter=',')
    X_reduced_valid = np.genfromtxt(path_kernelpca+'X_valid_kernelpca_'+name_kernel_pca+'.csv', delimiter=',')        
    X_reduced_test = np.genfromtxt(path_kernelpca+'X_test_kernelpca_'+name_kernel_pca+'.csv', delimiter=',')        
    return X_reduced_train, X_reduced_valid, X_reduced_test

def Part_3_kmeans_clustering(name_PCA_config, path_save_PCA, n_split, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight, ntest, path_save_clustering):    
    X_reduced_train, X_reduced_valid, X_reduced_test = Part_2_Load_PCA(name_PCA_config, path_save_PCA)
    # K-MEANS CLUSTERING
    print('Kmeans')
    X_centers_train, k_means_model, a_temp = k_means_X_cv(n_split, X_reduced_train, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight)
    labels_valid = predict(X_reduced_valid, X_centers_train)
    labels_test = predict(X_reduced_test, X_centers_train)
    print('Get Centroids')
    centroids = get_number_centroids(X_centers_train, X_reduced_train)   
    Part_3_save_clustering(ntest, path_save_clustering, labels_valid, labels_test, X_reduced_valid, X_reduced_test, centroids, a_temp,X_centers_train)
    return None

def Part_3_save_clustering(ntest, path_save_clustering, labels_valid, labels_test, X_reduced_valid, X_reduced_test, centroids, a_temp,X_centers_train):
    os.makedirs(path_save_clustering)
    np.save(path_save_clustering+'centroids'+ntest, centroids)
    np.save(path_save_clustering+'X_centers_train'+ntest, X_centers_train)
    np.save(path_save_clustering+'labels_test'+ntest, labels_test)
    np.save(path_save_clustering+'X_reduced_test'+ntest, X_reduced_test)
    np.save(path_save_clustering+'labels_valid'+ntest, labels_valid)
    np.save(path_save_clustering+'X_reduced_valid'+ntest, X_reduced_valid)
    return None


def Part_4_prediction(X_lats_lons, X_lats_lons_train, Alphas,lambdas, X_centers_train, kernel) :   
    if kernel == 'cosine':
        Kernel_matrix = cosine_similarity(X_lats_lons, X_lats_lons_train)
    elif kernel == 'linear':
        Kernel_matrix = linear_kernel(X_lats_lons, X_lats_lons_train)
    else:
        print('Houston, issue here ! ')
    Alphas_scaled = Alphas/np.sqrt(lambdas)
    X_reduced = np.matmul(Kernel_matrix, Alphas_scaled)
    labels = predict(X_reduced, X_centers_train)
    return labels, X_reduced


def run_script_prediction(name, norm, t, resamp, length_days, wtype, lats, lons, lats_train, lons_train, path_save, name_kernel_pca, X_centers_train, path_kernelpca, kernel) :   
    w, X_lats_lons = Part_1_pre_processing_one_sort(t, lats, lons, resamp, length_days, wtype, norm)
    w_train, X_lats_lons_train = Part_1_pre_processing_one_sort(t, lats_train, lons_train, resamp, length_days, wtype, norm)


    s = int(X_lats_lons.shape[1]/2)
    X_lats_lons[:,s:] = X_lats_lons[:,s:]*w
    s = int(X_lats_lons_train.shape[1]/2)
    X_lats_lons_train[:,s:] = X_lats_lons_train[:,s:]*w_train
    
    Alphas = np.genfromtxt(path_kernelpca+'alphas_kernelpca_'+name_kernel_pca+'.csv', delimiter=',')   
    Lambdas = np.genfromtxt(path_kernelpca+'lambda_kernelpca_'+name_kernel_pca+'.csv', delimiter=',')   
    return Part_4_prediction(X_lats_lons, X_lats_lons_train, Alphas, Lambdas, X_centers_train, kernel)

def apply_to_all_data(path_save_clustering, ntest, init_year,final_year, length_days, files, start_lon,  norm, t, resamp,wtype, name_PCA_config, path_save_PCA, Dataset_path, perctest, perctrain, kernel):
    centroids, labels_test, labels_valid, lats_test, lats_valid, lons_test, lons_valid, lons_train, lats_train, \
        X_reduced_test, X_Reduced_valid, temps_test, temps_train, temps_valid, sals_test, sals_train, sals_valid, X_centers_train = load_data_test_train_valid(path_save_clustering, ntest, Dataset_path, length_days, perctest, perctrain, start_lon)

    for yr in range(init_year,final_year):
        print('Year', yr)
        lats_all,lons_all, temps_all, sals_all, date_all = extract_all_per_year(yr, length_days, files, start_lon)
        labels_all = []
        for sep in range(0,lats_all.shape[0],500) :
            print(sep,'/',lats_all.shape[0])
            lats_all_l = lats_all[sep:sep+500,:] ; lons_all_l = lons_all[sep:sep+500,:]
            labels, X_reduced =  run_script_prediction(ntest, norm, t, resamp, length_days, wtype, lats_all_l, lons_all_l, lats_train, lons_train, path_save_clustering,
                                            name_PCA_config, X_centers_train, path_save_PCA, kernel)
            labels_all.extend(labels)
        np.save(path_save_clustering+'ClassifiedData/'+'labels_data_%i'%yr, labels_all)
    
    return None


def load_data_test_train_valid(path_save_clustering, ntest, Dataset_path, length_days, perctest, perctrain, start_lon):
    lats_all,lons_all, temps_all, sals_all, date_all = extract(Dataset_path, length_days, start_lon)
    lats_test, lons_test, lats_train, lons_train, lats_valid, lons_valid, temps_train, temps_test, temps_valid, sals_train, sals_test, sals_valid = split_sets(lats_all, lons_all,temps_all, sals_all, perctest, perctrain)

    CENTROID = np.load(path_save_clustering+'centroids'+ntest+'.npy', allow_pickle = True)
    LABELS_TEST = np.load(path_save_clustering+'labels_test'+ntest+'.npy', allow_pickle = True)
    X_REDUCED_TEST = np.load(path_save_clustering+'X_reduced_test'+ntest+'.npy', allow_pickle = True)
    LABELS_VALID = np.load(path_save_clustering+'labels_valid'+ntest+'.npy', allow_pickle = True)
    X_REDUCED_VALID = np.load(path_save_clustering+'X_reduced_valid'+ntest+'.npy', allow_pickle = True)
    X_centers_train = np.load(path_save_clustering+'X_centers_train'+ntest+'.npy', allow_pickle = True)
    
    return CENTROID, LABELS_TEST, LABELS_VALID, lats_test, lats_valid, lons_test, lons_valid, lons_train, lats_train, \
        X_REDUCED_TEST, X_REDUCED_VALID, temps_test, temps_train, temps_valid, sals_test, sals_train, sals_valid, X_centers_train

def load_data_year(path_save_clustering, yr, length_days, files, start_lon): 
    labels = np.load(path_save_clustering+'ClassifiedData/'+'labels_data_%i'%yr +'.npy', allow_pickle = True)
    lats_all,lons_all, temps_all, sals_all, date_all = extract_all_per_year(yr, length_days, files, start_lon)
    return lats_all, lons_all, temps_all, sals_all, date_all, labels



def get_labels_time_series(path_save_clustering, init_year,final_year, n_clusters,  length_days, files, start_lon):
    Dates_all = []; Perc_labels_all = [];
    for yr in range(init_year,final_year):
        print('Year', yr)
        lats_all, lons_all, temps_all, sals_all, time_data, labels_data = load_data_year(path_save_clustering, yr, length_days, files, start_lon)
        Date_data = np.unique(time_data)
        Dates_all.extend(Date_data)
        Perc_labels = np.zeros((len(Date_data), n_clusters))
        for i in range(len(Date_data)):
            d0 = np.where(Date_data[i] ==time_data)[0]
            for di in d0:
                Perc_labels[i,labels_data[di]] +=1
            Perc_labels[i,:] =100*Perc_labels[i,:]/len(d0)
        Perc_labels_all.extend(Perc_labels)
    Perc_labels_all = np.array(Perc_labels_all)
    return Dates_all,  Perc_labels_all
