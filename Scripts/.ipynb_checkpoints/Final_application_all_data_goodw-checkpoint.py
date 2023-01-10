#!/usr/bin/env python
# coding: utf-8

# In[33]:


import xarray as xr
import numpy as np
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://github.com/jakevdp/wpca
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
import cartopy.crs as ccrs
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity 


# # All functions

# Apply the model to the complete dataset.

# ## Data extraction

# In[34]:


# Remove short trajectories
def remove_short(ds):
    
    l = [len(ds.sel(traj=i).dropna(dim='obs').obs) for i in ds.traj]
    keep = [i for i in range(len(l)) if l[i] > 90] # more than 3 months
    ds2 = ds.sel(traj=xr.DataArray(keep)).rename_dims({'dim_0':'traj'})
    
    return ds2

def extract(Dataset_path, yr, length_days) :
    
    file = Dataset_path+'run_continuous_25years_track_%04d.nc'%yr
    
    ds = xr.open_dataset(file)
    
    # remove short trajectories
    print('Remove short trajectories')
    ds = remove_short(ds) 
    
    lats = ds.lat[:,0:length_days].values # keep only one year of data
    lons = ds.lon[:,0:length_days].values
    temps = ds.temperature[:,0:length_days].values
    sal = ds.salinity[:,0:length_days].values
    date = ds.time[:,1].values
    
    return lats, lons, temps, sal, date


# ##  Pre-processing

# ### Convert the lat/lon

# #### Translation

# In[35]:


def translate(lat, lon):
    return lat-np.repeat(lat[:,0][:, np.newaxis], lat.shape[1], axis=1), lon-np.repeat(lon[:,0][:, np.newaxis], lon.shape[1], axis=1)


# #### Resampling

# In[36]:


def resample(lats, lons, length_days):
    
    if length_days != None:
        
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
            
    else:
        lats_resamp = lats
        lons_resamp = lons
        
    return lats_resamp, lons_resamp



# ## Run TOTAL script

# In[38]:


def weightPCA(lons, lats, wtype):
    
    if wtype == 'cos' :
        w = np.cos( np.radians(lats) )
    elif wtype == 'sqrt-cos' :
        w = np.sqrt(abs(np.cos( np.radians(lats) )))
    elif wtype == None:
        w = []
        
    return w


# In[39]:


def predict(X_reduced_val, X_centers):
    
    X_reduced_val2 = np.repeat(X_reduced_val[:,:,np.newaxis], X_centers.shape[0], axis = 2)
    X_centers2 = np.repeat(np.transpose(X_centers)[np.newaxis,:,:], X_reduced_val.shape[0], axis = 0)
    
    return np.argmin(np.linalg.norm(X_reduced_val2-X_centers2, axis=1), axis = 1)


# In[40]:


def run_script(prefixe_kernel,name, norm, t, resamp, length_days, wtype, n_clusters, kernel, lats, lons, lats_train, lons_train, path_save, name_kernel_pca, X_centers_train) :   
        
    # PRE-PROCESSING
    print('pre-processing')

    # Translate
    if t == True :
        lats_pp, lons_pp = translate(lats, lons)
        lats_train_pp, lons_train_pp = translate(lats_train, lons_train)
    # Resample
    if resamp == True:
        lats_pp, lons_pp = resample(lats_pp, lons_pp, length_days)
        lats_train_pp, lons_train_pp = resample(lats_train_pp, lons_train_pp, length_days)
    # Weight
    w = weightPCA(lats_pp, lons_pp, wtype)
    w_train = weightPCA(lats_train_pp, lons_train_pp, wtype)
    
    # Reshape
    X_lats_lons = np.concatenate((lats_pp, lons_pp), axis = 1) #concatenation of features to make it in the correct shape
    X_lats_lons_train = np.concatenate((lats_train_pp, lons_train_pp), axis = 1)
    
    # Normalize
    if norm == True:
        X_lats_lons = normalize(X_lats_lons, copy=True)
        X_lats_lons_train = normalize(X_lats_lons_train, copy=True)
    if len(w)> 0:
        s = int(X_lats_lons.shape[1]/2)
        X_lats_lons[:,s:] = X_lats_lons[:,s:]*w
        s = int(X_lats_lons_train.shape[1]/2)
        X_lats_lons_train[:,s:] = X_lats_lons_train[:,s:]*w_train

    if resamp == False:
        X_lats_lons[np.isnan(X_lats_lons)] = 0 
        X_lats_lons_train[np.isnan(X_lats_lons_train)] = 0 

    #PCA
    print('load kernel')        
    path_kernelpca = '/storage/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Clustering/testnoresamp/'
    #vérifier que c'est OK de loader les matrices comme ca. Alphas doit etre de taille (N_train, N_components)
    Alphas = np.genfromtxt(path_kernelpca+prefixe_kernel+'alphas_kernelpca'+name_kernel_pca+'.csv', delimiter=',')   
    print('Apply model')
    Kernel_matrix = cosine_similarity(X_lats_lons, X_lats_lons_train)
    X_reduced = np.matmul(Kernel_matrix, Alphas)

    # K-MEANS CLUSTERING
    print('load results Kmeans')
    print('predict')
    labels = predict(X_reduced, X_centers_train)

    
    return labels, X_reduced


# ## Script for analysis

# In[42]:


# All possible set-ups
path_save = '/storage/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Clustering/Last_Version/config_k-cosine-w-cos_30noresampnoresamp_goodw/ClassifiedData/'
path_load = '/storage/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Clustering/Last_Version/config_k-cosine-w-cos_30noresampnoresamp_goodw/'
length_days = 550
Dataset_path = '/storage/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Retroflection/'
#Dataset_path = '/storage/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Clustering/training_25years_no_bias_corr_100000.nc'


#### Vérfiier si oui ou non on regait un clustering ? Normalement il suffit de charger lezs centroids ? 

#***************************************************************************************
#             NO COR - Kernel = cos, Weigths = cos, N =30
#***************************************************************************************

projs=None # type of projection of lat/lon
norm = False # if normalize
t=True # translation
resamp = False # resampling
wtype='cos' # Type of weight. None if no weight. Otherwise cos or sqrt-cos
kernel='cosine'
ntest='k-cosine-w-cos_30noresamp'
name_kernels_pca='_noresamp_goodw'
prefixe_kernel=''
n_clusters=30

# ## Execution

# In[43]:


# Load training set

lats_train = np.load(path_load+'lats_train'+ntest+'.npy', allow_pickle = True)
lons_train = np.load(path_load+'lons_train'+ntest+'.npy', allow_pickle = True)
X_centers_train = np.load(path_load+'X_centers_train'+ntest+'.npy', allow_pickle = True)


# In[45]:


# Loop over all years

for yr in range(1993,2017):
    print('Year', yr)

    lats_all,lons_all, temps_all, sals_all, date_all = extract(Dataset_path, yr, length_days)

    # split into segments to avoid memory error
    labels_all = []
    for sep in range(0,lats_all.shape[0],500) :
        print(sep,'/',lats_all.shape[0])
        lats_all_l = lats_all[sep:sep+501,:] ; lons_all_l = lons_all[sep:sep+501,:]
    
        labels, X_reduced =  run_script(prefixe_kernel, ntest, norm, t, resamp, length_days, wtype, n_clusters, kernel, lats_all_l, lons_all_l, lats_train, lons_train, path_save, name_kernels_pca, X_centers_train)


        # Get retroflection index
        labels_all.extend(labels)
        ###time, retro = retro_idx(labels, date_all)
        
    np.save(path_save+'labels_data_%i'%yr, labels_all)
    np.save(path_save+'time_data_%i'%yr, date_all)






