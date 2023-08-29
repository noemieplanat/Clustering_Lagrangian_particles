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






def run_adv_journalier_sensitivity(tup):
    (nmb_h, nmb_z, dt, dr, year) = tup
    #---------------------------------------- Fixed PARAMETERS
    # ---- Paths
    folder_files = '/storage/nplanat/Glorys12_ORCA_journalier/'
    mesh_file_h = '/storage/nplanat/Glorys12_masks/Mask_hgr_12.nc'
    mesh_file_z = '/storage/nplanat/Glorys12_masks/Mask_zgr_12.nc'
    mask = '/storage/nplanat/Glorys12_masks/Mask_G12.nc'

    # ---- Getting paths
    filesU = [name for name in sorted(glob.glob(folder_files + 'Glorys12_GridU*')) if name[-3:]!='tmp']
    filesV = [name for name in sorted(glob.glob(folder_files + 'Glorys12_GridV*')) if name[-3:]!='tmp']
    filesW = [name for name in sorted(glob.glob(folder_files + 'Glorys12_GridW*')) if name[-3:]!='tmp']
    filesT = [name for name in sorted(glob.glob(folder_files + 'Glorys12_GridT*')) if name[-3:]!='tmp']
    filesS = [name for name in sorted(glob.glob(folder_files + 'Glorys12_GridS*')) if name[-3:]!='tmp']
    filesM = [mask for k in range(len(filesU))]


    exp = str(nmb_h)+'_'+str(nmb_z)+'_'+str(dt)+'_'+str(dr)+'_'+str(year)+'_'
    i_file = 0
    fname ='/mnt/shackleton/storage3/nplanat/OP/ADV_'+exp+'%i_' %i_file #'/storage/nplanat/Glorys12_OP_journalier/ADV_'+exp+'%i_' %i_file 
    if os.path.exists(fname):
        i_file +=1
        print('ERROR : this experiment already exist !!!! ')
        fname ='/mnt/shackleton/storage3/nplanat/OP/ADV_'+exp+'%i_' %i_file
    print(fname)    

    #--------------------------------------------------
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
    # Kernel to track the age of particles
    def ageing(particle, fieldset, time):
        particle.age += particle.dt/86400. #en jour, doit correspondre a dt(secondes)/86400

    def killold(particle, fieldset, time):
        if particle.age > 365.*4. : #delete after X days
            particle.delete()

    def Modified_AdvectionRK4_3D(particle, fieldset, time):
        # make sure we don't exceed the surface and the boundaries 

        depmin = .10 #premier point de deptht
        maxlat = 89.95
        if particle.depth <depmin:
            particle.delete()
        if particle.lon >maxlat:
            particle.delete()
        (u1, v1, w1) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]

        lon1 = particle.lon + u1*.5*particle.dt
        lat1 = particle.lat + v1*.5*particle.dt
        dep1 = particle.depth + w1*.5*particle.dt
        if dep1 <depmin :
            particle.delete() #delete if surface
        if lat1>maxlat:
            particle.delete()
        # RK4 (k2, k3, k4)    
        (u2, v2, w2) = fieldset.UVW[time + .5 * particle.dt, dep1, lat1, lon1]

        lon2 = particle.lon + u2*.5*particle.dt
        lat2 = particle.lat + v2*.5*particle.dt
        dep2 = particle.depth + w2*.5*particle.dt
        if dep2 < depmin:
            particle.delete()
        if lat2>maxlat:
            particle.delete()
        (u3, v3, w3) = fieldset.UVW[time + .5 * particle.dt, dep2, lat2, lon2]

        lon3 = particle.lon + u3*particle.dt
        lat3 = particle.lat + v3*particle.dt
        dep3 = particle.depth + w3*particle.dt
        if dep3 < depmin:
            particle.delete()
        if lat3>maxlat:
            particle.delete()
        (u4, v4, w4) = fieldset.UVW[time + particle.dt, dep3, lat3, lon3]

        particle.lon += (u1 + 2.*u2 + 2.*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2.*v2 + 2.*v3 + v4) / 6. * particle.dt
        depth_old = particle.depth
        particle.depth += (w1 + 2.*w2 + 2.*w3 + w4) / 6. * particle.dt
        if particle.lat<65.5:
            particle.delete()

        if particle.depth <depmin:
            particle.delete()
        if particle.lat>maxlat:
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



    # Times
    Liste_time_journalier= get_liste_time(filesU)
    Liste_Years = [Liste_time_journalier[i][0].astype('datetime64[Y]').astype(int) + 1970 for i in range(len(Liste_time_journalier))]
    print('liste_time - done')


    # Loop every year
    yrs = range(year,year+5)
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



        Lats = np.repeat(np.linspace(65.59, 66.04, nmb_h), nmb_z)
        Lons = np.repeat(np.linspace(-168.09, -169.62, nmb_h), nmb_z)
        Depths = np.array([np.linspace(0, 90, nmb_z) for i in range(nmb_h)]).reshape((nmb_h*nmb_z))

        repeatdt = delta(days=1) # release a new set of particles every day


        pset = ParticleSet(fieldset=fset,   # the fields on which the particles are advected
                           pclass=ocean_particle,  # the type of particles (JITParticle or ScipyParticle)
                           lon=Lons, 
                           depth = Depths,# release longitudes 
                           lat=Lats, 
                           repeatdt=repeatdt, 
                           lonlatdepth_dtype=np.float64)             # release latitudes
        
        kernels = pset.Kernel(Modified_AdvectionRK4_3D)+ pset.Kernel(SampleVars) +   pset.Kernel(ageing) +pset.Kernel(killold)
        
        #Want to run simulation for 1 years releasing particles every week, and let it run for 3 years.
        output_file = pset.ParticleFile(fname+"%i"%yrs[i]+'.nc', outputdt=delta(days=1))

        pset.execute(kernels, runtime=delta(days=30), dt=delta(minutes = dt), output_file=output_file, verbose_progress=True)
        pset.repeatdt = None
        pset.execute(kernels, runtime=delta(days=3*365), dt=delta(minutes = dt), output_file=output_file, verbose_progress=True)
        #pset.repeatdt = delta(days=1) 
        #pset.execute(kernels, runtime=delta(days=30), dt=delta(minutes = dt), output_file=output_file, verbose_progress=True)
        #pset.repeatdt = None
        #pset.execute(kernels, runtime=delta(days=30*5), dt=delta(minutes = dt), output_file=output_file, verbose_progress=True)


        # Save output
        print('Save')
        output_file.export()
        print('time is ', time.time()-tic)

    output_file.close()
    return None

