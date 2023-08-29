#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import pyproj as proj
import random
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib as mpl
import time
import cartopy.crs as ccrs
import cartopy as cr
import os
import copy as cp
import matplotlib.colors as colors
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker


# # Propagation analysis

# Analyse du spread de temps de propagation dans les clusters

# In[2]:


# Load

NTEST = 'k-cosine-w-cos_30noresamp'
path_save = '/mnt/shackleton/storage1/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Clustering/Last_Version/config_k-cosine-w-cos_30noresampnoresamp_goodw/'
Dataset_path = '/mnt/shackleton/storage1/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Clustering/training_25years_100000.nc'

path_clusters = '/mnt/shackleton/storage1/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Clustering/Last_Version/config_k-cosine-w-cos_30noresampnoresamp_goodw/ClassifiedData/'
path_figs = '../Figures_Clustering/Figures/Labrador/' 

lats_all = [] ; lons_all = [] ; depths_all = [] ; labels_all = []
for yr in range(1993,2014):#2017
    print('Year', yr)
    labels_all.extend( np.load(path_clusters+'labels_data_'+str(yr)+'.npy', allow_pickle = True) )
    lats_all.extend( np.load(path_clusters+'lats_'+str(yr)+'.npy', allow_pickle = True) )
    lons_all.extend( np.load(path_clusters+'lons_'+str(yr)+'.npy', allow_pickle = True) )
    
lats_all = np.array(lats_all)
lons_all = np.array(lons_all)

# Load bathymetry
#bathy = xr.open_dataset('/storage3/shared/Glorys12/GLO-MFC_001_030_mask_bathy.nc')
bathy = xr.open_dataset('/mnt/shackleton/storage1/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/GLO-MFC_001_030_mask_bathy.nc')
latb = bathy.latitude ; lonb = bathy.longitude
bathy = bathy.deptho


# In[3]:


# plot density plots for each group

Col = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple', 'tab:pink', 'tab:cyan']

L1 = [1,3,6,7,8,9,10,18,19,23,27,29] # Retro
L2 = [21] # Slope Sea avant retro
L3 = [2,17,21,22,25] # Slope Sea
L4 = [26] # Labrador Sea
L5 = [0,4,5,12,13,14,15,16,20] # Labrador Shelf
L6 = [11,28] # South
L7 = [24] # Belle Isle
Ls = [L1,L2,L3,L4,L5,L6,L7]
labels = ['Retroflected', 'Westward, then retroflected', 'Westward-flowing', 'Labrador Sea', 'Labrador Shelf', 'Southward-flowing', 'Belle Isle']

# Retroflected: BB-s line, West: SESPB but longer, South: 37N line, Labrador Sea: du bout de SI au bout de BB
loc_propag = {'Retroflected':[-53.9, -46, 48, 49.5], 'Westward-flowing':[-56., -56., 40., 49.], 'Westward, then retroflected':[-54.2, -54.2, 40, 47.4], 'Southward-flowing':[-55, -40, 37, 37], 'Labrador Sea':[-47.4, -51.8, 50.4, 55.1], 'Labrador Shelf':[0,0,0,0], 'Belle Isle':[-56.3, -56.8, 51.3, 51.8]}
loc_name = {'Retroflected':'BB', 'Westward-flowing':'SESPB', 'Westward, then retroflected':'SESPB', 'Southward-flowing':'37N', 'Labrador Sea':'Lab Sea', 'Labrador Shelf':'-', 'Belle Isle':'BI strait'}


# In[ ]:



tol = 0.1

# for each cluster
days = [[] for i in range(30)] ; secs = []
for clust in range(30):
    print('Doing cluster', clust, '/ 30')
    
    # get the cluster location we want
    c = [idx for idx in range(len(Ls)) if clust in Ls[idx]][0]
    label = labels[c]
    print(label)
    secs.append(loc_name[label])

    for n in range(len(labels_all)):
        
        if labels_all[n]==clust :
            
            # tilted line
            if label == 'Retroflected' or label == 'Labrador Sea' or label == 'Belle Isle' :
                latsec = np.arange(loc_propag[label][2], loc_propag[label][3], 0.1)
                lonsec = np.linspace(loc_propag[label][0], loc_propag[label][1], len(latsec))
            
                idxlat = [] ; idxlon = []
                for i in range(len(latsec)-1):
                    sublat = [idx for idx in range(len(lats_all[n,:])) if lats_all[n,idx]<latsec[i]+tol and lats_all[n,idx]>latsec[i]-tol ] 
                    sublon = [idx for idx in range(len(lons_all[n,:])) if lons_all[n,idx]<lonsec[i]+tol and lons_all[n,idx]>lonsec[i]-tol ] 
                    if len(sublat)>0 and len(sublon)>0:
                        idxlat.extend(sublat) ; idxlon.extend(sublon)
                if len(idxlat) > 0 and len(idxlon) > 0:
                    dum = list(set(idxlat).intersection(idxlon))
                    if len(dum) > 0:
                        days[clust].append(dum[0]) # lag
                        
            # straight line
            elif label == 'Westward-flowing' or label == 'Westward, then retroflected' or label == 'Southward-flowing' :
                idxlat = [idx for idx in range(len(lats_all[n,:])) if lats_all[n,idx]<loc_propag[label][3]+tol and lats_all[n,idx]>loc_propag[label][2]-tol ] 
                idxlon = [idx for idx in range(len(lons_all[n,:])) if lons_all[n,idx]<loc_propag[label][0]+tol and lons_all[n,idx]>loc_propag[label][1]-tol ] 
                if len(idxlat)>0 and len(idxlon)>0:
                    dum = list(set(idxlat).intersection(idxlon))
                    if len(dum) > 0:
                        days[clust].append(dum[0])
                        


# In[ ]:


# plot
#f = plt.figure(figsize=(20,15))
#for n in range(30):
#    
#    c = [idx for idx in range(len(Ls)) if n in Ls[idx]][0]
#    label = labels[c]
#    
#    ax = f.add_subplot(5,6,n+1)
#    ax.hist(days[n], bins=range(0,500,25), color=Col[c])
#    ax.set_xlim([0,500])
#    if n>24:
#        ax.set_xlabel('Days')
#    ax.set_title(secs[n])
#    
#plt.tight_layout()
#plt.savefig(path_figs+'lags.png',dpi=300)
#plt.show()


# plot per group
lw = 8
pt = [0.75,0.85]

f = plt.figure(figsize=(15,13))
c=1
# Retroflected
ax = f.add_subplot(7,6,c, projection = ccrs.Robinson(central_longitude=-50))
ax.coastlines(color='silver')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5)
gl.top_labels = True ; gl.left_labels = True ; gl.bottom_labels=False ; gl.right_labels = False
gl.ylocator = mticker.FixedLocator([40,50,60])
ax.set_extent([-70, -35, 35, 60],  crs=ccrs.PlateCarree())
ax.add_feature(cr.feature.LAND, linewidth=0.5, edgecolor='white')
loc = loc_propag['Retroflected']
ax.plot([loc[0], loc[1]], [loc[2], loc[3]], lw=lw, c='k', transform = ccrs.PlateCarree())
ax.plot([-56.7,-52],[53,54.3], c='k', lw=4, transform=ccrs.PlateCarree())
plt.contour(lonb, latb, bathy, [350], colors='dimgrey', transform = ccrs.PlateCarree(), zorder=2)
ax.text(-0.45, 0.5, labels[0], va='center', fontsize=12, fontweight='bold', rotation=90, transform=ax.transAxes)
for n in L1: 
    c+=1
    if c==7 or c==13:
        c+=1
    ax = f.add_subplot(7,6,c)
    ax.hist(days[n], bins=range(0,500,25), color=Col[0])
    ax.text(pt[0],pt[1],'# %i'%(n+1), fontweight='bold', transform=ax.transAxes)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
c+=4

# West
ax = f.add_subplot(7,6,c, projection = ccrs.Robinson(central_longitude=-50))
ax.coastlines(color='silver')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5)
gl.top_labels = False ; gl.left_labels = False ; gl.right_labels=False ; gl.bottom_labels=False
gl.ylocator = mticker.FixedLocator([40,50,60])
ax.set_extent([-70, -35, 35, 60],  crs=ccrs.PlateCarree())
ax.add_feature(cr.feature.LAND, linewidth=0.5, edgecolor='white')
loc = loc_propag['Westward-flowing']
ax.plot([loc[0], loc[1]], [loc[2], loc[3]], lw=lw, c='k', transform = ccrs.PlateCarree())
ax.plot([-56.7,-52],[53,54.3], c='k', lw=4, transform=ccrs.PlateCarree())
plt.contour(lonb, latb, bathy, [350], colors='dimgrey', transform = ccrs.PlateCarree(), zorder=2)
ax.text(-0.45, 0.5, labels[2], va='center', fontsize=12, fontweight='bold', rotation=90, transform=ax.transAxes)

for n in L3:
    c+=1
    ax = f.add_subplot(7,6,c)
    ax.hist(days[n], bins=range(0,500,25), color=Col[2])
    ax.text(pt[0],pt[1],'# %i'%(n+1), fontweight='bold',transform=ax.transAxes)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
c+=1


# Labrador Sea
ax = f.add_subplot(7,6,c, projection = ccrs.Robinson(central_longitude=-50))
ax.coastlines(color='silver')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5)
gl.top_labels = False ; gl.left_labels = False ; gl.right_labels=False ; gl.bottom_labels=False
gl.ylocator = mticker.FixedLocator([40,50,60])
ax.set_extent([-70, -35, 35, 60],  crs=ccrs.PlateCarree())
ax.add_feature(cr.feature.LAND, linewidth=0.5, edgecolor='white')
loc = loc_propag['Labrador Sea']
ax.plot([loc[0], loc[1]], [loc[2], loc[3]], lw=lw, c='k', transform = ccrs.PlateCarree())
ax.plot([-56.7,-52],[53,54.3], c='k', lw=4, transform=ccrs.PlateCarree())
plt.contour(lonb, latb, bathy, [350], colors='dimgrey', transform = ccrs.PlateCarree(), zorder=2)
ax.text(-0.45, 0.5, labels[3], va='center', fontsize=12, fontweight='bold', rotation=90, transform=ax.transAxes)

for n in L4:
    c+=1
    ax = f.add_subplot(7,6,c)
    ax.hist(days[n], bins=range(0,500,25), color=Col[3])
    ax.text(pt[0],pt[1],'# %i'%(n+1), fontweight='bold',transform=ax.transAxes)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    if c == 20 or c == 21:
        ax.set_xlabel('Days')
c+=5


# West then retro
ax = f.add_subplot(7,6,c, projection = ccrs.Robinson(central_longitude=-50))
ax.coastlines(color='silver')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5)
gl.top_labels = False ; gl.left_labels = False ; gl.right_labels=False ; gl.bottom_labels=False
gl.ylocator = mticker.FixedLocator([40,50,60])
ax.set_extent([-70, -35, 35, 60],  crs=ccrs.PlateCarree())
ax.add_feature(cr.feature.LAND, linewidth=0.5, edgecolor='white')
loc = loc_propag['Westward, then retroflected']
ax.plot([loc[0], loc[1]], [loc[2], loc[3]], lw=lw, c='k', transform = ccrs.PlateCarree())
ax.plot([-56.7,-52],[53,54.3], c='k', lw=4, transform=ccrs.PlateCarree())
plt.contour(lonb, latb, bathy, [350], colors='dimgrey', transform = ccrs.PlateCarree(), zorder=2)
ax.text(-0.45, 0.5, "Westward,\nthen retro", va='center', fontsize=12, fontweight='bold', rotation=90, transform=ax.transAxes)

for n in L2:
    c+=1
    ax = f.add_subplot(7,6,c)
    ax.hist(days[n], bins=range(0,500,25), color=Col[1])
    ax.text(pt[0],pt[1],'# %i'%(n+1), fontweight='bold',transform=ax.transAxes)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
c+=5


# Belle Isle
ax = f.add_subplot(7,6,c, projection = ccrs.Robinson(central_longitude=-50))
ax.coastlines(color='silver')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5)
gl.top_labels = False ; gl.left_labels = False ; gl.right_labels=False ; gl.bottom_labels = False
gl.ylocator = mticker.FixedLocator([40,50,60])
ax.set_extent([-70, -35, 35, 60],  crs=ccrs.PlateCarree())
ax.add_feature(cr.feature.LAND, linewidth=0.5, edgecolor='white')
loc = loc_propag['Belle Isle']
ax.plot([loc[0], loc[1]], [loc[2], loc[3]], lw=lw, c='k', transform = ccrs.PlateCarree())
ax.plot([-56.7,-52],[53,54.3], c='k', lw=4, transform=ccrs.PlateCarree())
plt.contour(lonb, latb, bathy, [350], colors='dimgrey', transform = ccrs.PlateCarree(), zorder=2)
ax.text(-0.45, 0.5, labels[6], va='center', fontsize=12, fontweight='bold', rotation=90, transform=ax.transAxes)

for n in L7:
    c+=1
    ax = f.add_subplot(7,6,c)
    ax.hist(days[n], bins=range(0,500,25), color=Col[6])
    ax.text(pt[0],pt[1],'# %i'%(n+1), fontweight='bold',transform=ax.transAxes)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlabel('Days')

plt.subplots_adjust(left=0.1,bottom=None,right=None,top=None,wspace=0.25,hspace=0.05)
#plt.tight_layout()
plt.savefig(path_figs+'lags_groups_all.png',dpi=300)
#plt.show()
plt.close()

# In[ ]:




