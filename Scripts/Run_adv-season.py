# -*- coding: utf-8 -*-


import glob
from parcels import Field,FieldSet, ParticleSet, Variable, JITParticle,ScipyParticle, AdvectionRK4_3D, AdvectionRK4
#from parcels.tools.error import recovery_kernel_out_of_bounds
import xarray as xr
from datetime import timedelta as delta
import os
#import cartopy.crs as ccrs
from operator import attrgetter
import numpy as np
import time


class ocean_particle(JITParticle):
    #add some variables
    age = Variable('age',dtype=np.float32, initial=0.)
    stuck = Variable('stuck', dtype=np.int32, initial=0.)
    prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon'))  # the previous longitude
    prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat'))  # the previous latitude.
    temperature = Variable('temperature', dtype=np.float32)
    uvel = Variable('uvel', dtype=np.float64)
    vvel = Variable('vvel', dtype=np.float64)
    wvel = Variable('wvel', dtype=np.float32)
    salinity = Variable('salinity', dtype=np.float32)
    #maskU = Variable('maskU', dtype=np.float32)
    #maskV = Variable('maskV', dtype=np.float32)
    #maskF = Variable('maskF', dtype=np.float32)
    #maskT = Variable('maskT', dtype=np.float32)
    
    #area = Variable('area', dtype=np.float32, to_write='once', initial=0.)
    # It would be much better if we could store it only at the first time step, but at the first time step u=0 so volume=0...
    #volume = Variable('volume', dtype=np.float32, initial=0.)
    #volumeperp = Variable('volumeperp', dtype=np.float32, initial=0.)


# Kernel to check if the particles are stuck
def stuckParticle(particle, fieldset, time):
    if (particle.prev_lon == particle.lon) and (particle.prev_lat == particle.lat):
        particle.stuck += 1
    # Set the stored values for next iteration.
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat    
# Kernel to track water properties such as temperature and salinity
def SampleVars(particle, fieldset, time):
    particle.temperature = fieldset.T[time, particle.depth, particle.lat, particle.lon]
    particle.salinity = fieldset.S[time, particle.depth, particle.lat, particle.lon]    
    particle.uvel = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    particle.vvel = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    particle.wvel = fieldset.W[time, particle.depth, particle.lat, particle.lon]
    #particle.maskU = fieldset.umask[time, particle.depth, particle.lat, particle.lon]
    #particle.maskV = fieldset.vmask[time, particle.depth, particle.lat, particle.lon]
    #particle.maskF = fieldset.fmask[time, particle.depth, particle.lat, particle.lon]
    #particle.maskT = fieldset.tmask[time, particle.depth, particle.lat, particle.lon]
# Kernel to track the age of particles
def ageing(particle, fieldset, time):
    particle.age += particle.dt/86400. #en jour, doit correspondre a dt(secondes)/86400

def killold(particle, fieldset, time):
    if particle.age > 365.*4. : #delete after X days
        particle.delete()
        
def Modified_AdvectionRK4_3D(particle, fieldset, time):
    # make sure we don't exceed the surface and the boundaries 

    depmin = .494025 #premier point de deptht
       
    if particle.depth <depmin:
        particle.delete()

    (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
   
    lon1 = particle.lon + u1*.5*particle.dt
    lat1 = particle.lat + v1*.5*particle.dt
    dep1 = particle.depth + w1*.5*particle.dt
    if dep1 <depmin :
        particle.delete() #delete if surface

    # RK4 (k2, k3, k4)    
    (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]

    lon2 = particle.lon + u2*.5*particle.dt
    lat2 = particle.lat + v2*.5*particle.dt
    dep2 = particle.depth + w2*.5*particle.dt
    if dep2 < depmin:
        particle.delete()
    
    (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]

    lon3 = particle.lon + u3*particle.dt
    lat3 = particle.lat + v3*particle.dt
    dep3 = particle.depth + w3*particle.dt
    if dep3 < depmin:
        particle.delete()
    (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]

    particle.lon += (u1 + 2.*u2 + 2.*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2.*v2 + 2.*v3 + v4) / 6. * particle.dt
    depth_old = particle.depth
    particle.depth += (w1 + 2.*w2 + 2.*w3 + w4) / 6. * particle.dt
    if particle.lat<65.5:
        particle.delete()
    #if particle.depth == depth_old: #remove particles if they keep the same depth (on land ...)
    #    particle.delete()
    if particle.depth <depmin:
        particle.delete()
        
def check_velocities(particle, fieldset, time):
    if particle.uvel==0 or particle.vvel==0. or particle.wvel==0.:
        particle.delete()
        
    


def sort_file_liste(files):
    dtype = [('file', 'S100'), ('date', 'datetime64[ns]')]
    values = [(file, xr.open_dataset(file).time_counter.values[0]) for file in files]
    Correct_indices = np.where([type(values[i][1])==np.datetime64 for i in range(len(values))])[0].astype(int)
    Incorrect_indices = np.where([type(values[i][1])!=np.datetime64 for i in range(len(values))])[0].astype(int)
    Val_sel = [values[x] for x in Correct_indices]
    Array = np.array(Val_sel, dtype=dtype)
    np.sort(Array, order='date')
    Times = [x[1] for x in Array]
    unique_elements, indices = np.unique(Times, return_index = True)
    return [x[0] for x in Array[indices]], Incorrect_indices


def get_liste_time(filesU):
    Liste = []
    for file in filesU:
        ds = xr.open_dataset(file)
        timeU = ds.time_counter.values
        Liste.append(timeU)
    ListeT = np.array(Liste)
    return ListeT


################# Paths
folder_files = '/storage/nplanat/Glorys12_ORCA_seasons/'
mesh_file_h = '/storage/nplanat/Glorys12_masks/Mask_hgr_12.nc'
mesh_file_z = '/storage/nplanat/Glorys12_masks/Mask_zgr_12.nc'
mask = '/storage/nplanat/Glorys12_masks/Mask_G12.nc'



# ----  Getting paths
# ----  Getting paths
filesU = [name for name in sorted(glob.glob(folder_files + 'NEW_G12_gridU*')) if name[-3:]!='tmp']
filesV = [name for name in sorted(glob.glob(folder_files + 'NEW_G12_gridV*')) if name[-3:]!='tmp']
filesW = [name for name in sorted(glob.glob(folder_files + 'NEW_G12_gridW*')) if name[-3:]!='tmp']
filesT = [name for name in sorted(glob.glob(folder_files + 'NEW_G12_gridT*')) if name[-3:]!='tmp']
filesS = [name for name in sorted(glob.glob(folder_files + 'NEW_G12_gridS*')) if name[-3:]!='tmp']

filesM = [mask for k in range(len(filesU))]

print('files done')

#output file 
Liste_time_journalier= get_liste_time(filesU)
Liste_Years = [Liste_time_journalier[i][0].astype('datetime64[Y]').astype(int) + 1970 for i in range(len(Liste_time_journalier))]
print('liste_time - done')

i_file = 200
fname = '/storage/nplanat/Glorys12_OP_journalier/ADV_s_%i_' %i_file 
while os.path.exists(fname):
    i_file +=1
    fname = '/storage/nplanat/Glorys12_OP_journalier/ADV_s_%i_' %i_file
print(fname)


# Loop every year
yrs = range(2005,2019)
for i in range(len(yrs)-4):
    tic = time.time()
    print('Doing loop y:', yrs[i])

    # --- Get the data --- #
    # Adjust files to the length of the run
    # ----  Getting paths
    filesU1 = [filesU[j] for j in range(len(filesU)) if Liste_Years[j]<=yrs[i+4] and Liste_Years[j]>=yrs[i]]
    filesV1 = [filesV[j] for j in range(len(filesU)) if Liste_Years[j]<=yrs[i+4] and Liste_Years[j]>=yrs[i]]
    filesW1 = [filesW[j] for j in range(len(filesU)) if Liste_Years[j]<=yrs[i+4] and Liste_Years[j]>=yrs[i]]
    filesT1 = [filesT[j] for j in range(len(filesU)) if Liste_Years[j]<=yrs[i+4] and Liste_Years[j]>=yrs[i]]
    filesS1 = [filesS[j] for j in range(len(filesU)) if Liste_Years[j]<=yrs[i+4] and Liste_Years[j]>=yrs[i]]
    filesM1 = [mask for k in range(len(filesU1))]

    Liste_times = get_liste_time(filesU1)
    
    # dictionaries
    filenames = {'U': {'lon': mesh_file_h, 'lat': mesh_file_h,'depth':filesW[0], 'data': filesU1},
                 'V': {'lon': mesh_file_h, 'lat': mesh_file_h,'depth':filesW[0], 'data': filesV1},
                 'W': {'lon': mesh_file_h, 'lat': mesh_file_h,'depth':filesW[0], 'data': filesW1}, 
                 'T': {'lon': mesh_file_h, 'lat': mesh_file_h,'depth':filesW[0], 'data': filesT1},
                 'S': {'lon': mesh_file_h, 'lat': mesh_file_h,'depth':filesW[0], 'data': filesS1}}
 
    variables = {'U': 'vozocrtx',
                 'V': 'vomecrty', 
                 'W': 'vovecrtz',
                 'T': 'votemper', 
                 'S': 'vosaline'}

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time':'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time':'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time':'time_counter'},
                  'T': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time':'time_counter'},
                  'S': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time':'time_counter'}}

    fset = FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation =True, transpose=False)#field_chunksize=False)
    print('Field set - Done')

    #filenamesM = {'lon': mesh_file_h, 'lat': mesh_file_h,'depth':filesW[0], 'data':filesM1}
    #Mu = Field.from_netcdf(filenamesM, 'umask', {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time':'t'}, timestamps = Liste_times)
    #Mv = Field.from_netcdf(filenamesM, 'vmask', {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time':'t'}, timestamps = Liste_times)
    #$Mf = Field.from_netcdf(filenamesM, 'fmask', {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time':'t'}, timestamps = Liste_times)
    #Mt = Field.from_netcdf(filenamesM, 'tmask', {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time':'t'}, timestamps = Liste_times)
    
    #fset.add_field(Mu)
    #fset.add_field(Mv)
    #fset.add_field(Mf)
    #fset.add_field(Mt)



    # Particle set
    nmb_h = 24
    Levs_vert = np.array([4.9402538e-01, 1.5413754e+00, 2.6456685e+00, 3.8194947e+00,\
       5.0782237e+00, 6.4406142e+00, 7.9295602e+00, 9.5729971e+00,\
       1.1405003e+01, 1.3467138e+01, 1.5810073e+01, 1.8495560e+01,\
       2.1598816e+01, 2.5211409e+01, 2.9444729e+01, 3.4434155e+01,\
       4.0344051e+01, 4.7373688e+01, 5.5764290e+01, 6.5807274e+01])
    nmb_z = len(Levs_vert)
    Lats = np.repeat(np.linspace(65.59, 66.04, nmb_h), nmb_z)
    Lons = np.repeat(np.linspace(-168.09, -169.62, nmb_h), nmb_z)
    Depths = np.array([Levs_vert for i in range(nmb_h)]).reshape((nmb_h*nmb_z))

    repeatdt = delta(days=7) # release a new set of particles every week


    pset = ParticleSet(fieldset=fset,   # the fields on which the particles are advected
                       pclass=ocean_particle,  # the type of particles (JITParticle or ScipyParticle)
                       lon=Lons, 
                       depth = Depths,# release longitudes 
                       lat=Lats, 
                       repeatdt=repeatdt, 
                       lonlatdepth_dtype=np.float64)             # release latitudes

    #print('Particle Set - Done')
    kernels = pset.Kernel(Modified_AdvectionRK4_3D)+ pset.Kernel(SampleVars) +   pset.Kernel(ageing) +pset.Kernel(killold)


    #Want to run simulation for 1 years releasing particles every week, and let it run for 3 years.
    output_file = pset.ParticleFile(fname+"%i"%yrs[i]+'.nc', outputdt=delta(days=1))

    #Start run for one year
    print('First part of run')
    pset.execute(kernels, runtime=delta(days=365), dt=delta(minutes =10), output_file=output_file, verbose_progress=True)

    #Now we want to stop releasing new particles
    pset.repeatdt = None

    #Now we do the next 3 years with no new particles being released
    print('Second part of run')
    pset.execute(kernels, runtime=delta(days=4*365), dt=delta(minutes = 10), output_file=output_file, verbose_progress=True)


    # Save output
    print('Save')
    output_file.export()
    print('time is ', time.time()-tic)

output_file.close()

