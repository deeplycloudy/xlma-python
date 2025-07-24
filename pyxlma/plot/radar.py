import os
import pyart
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from math import radians, cos, sin, asin, sqrt, degrees, atan
import cartopy.feature as cfeature
import gc

try:
    from metpy.plots import USCOUNTIES
    county_scales = ['20m', '5m', '500k']
    COUNTIES = USCOUNTIES.with_scale(county_scales[0])
except ImportError:
    COUNTIES = None

def negativeaxis(position, radar):
    
    if position < 0:
        if position < radar[0].radar_longitude['data'][0]:
            return True
    else:
        if position < radar[0].radar_latitude['data'][0]:
            return True
    return False

def haversine(lat1, lon1, lat2, lon2):

      R = 6378.137e3 # this is in meters.  For Earth radius in kilometers use 6372.8 km

      dLat = radians(lat2 - lat1)
      dLon = radians(lon2 - lon1)
      lat1 = radians(lat1)
      lat2 = radians(lat2)

      a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
      c = 2*asin(sqrt(a))

      return R * c

def ReadRadar(radar_file):
    radar_pickle = radar_file[0]+str(len(radar_file))+'.pickle'
    if os.path.exists(radar_pickle):
        with open(radar_pickle, 'rb') as pickle_file:
            grided_radar = pickle.load(pickle_file) 
            start_times = pickle.load(pickle_file)
    else:   
            radar_list = []
            start_times = []
            radar_file.sort()
            for files in radar_file:
                    print('Reading: ' + files)
                    radar = pyart.io.read(files)
                    radar_list.append(radar)
                    radar_name = radar.metadata['instrument_name']
                    try:
                        start = dt.datetime.strptime(files.split('/')[-1], radar_name + "%Y%m%d_%H%M%S_V06")#.strftime('%d/%m/%Y %H%M.%S')
                    except ValueError:
                        start = dt.datetime.strptime(files.split('\\')[-1], radar_name + "%Y%m%d_%H%M%S_V06")
                    start_times.append(start)
                    if int(start.strftime('%M')) >= 50:
                            gc.collect()
                    #print(start_times)
            grided_radar = GridRadar(radar_list)
            with open(radar_pickle, 'wb') as pickle_file:
                    pickle.dump(grided_radar, pickle_file)
                    pickle.dump(start_times, pickle_file)
    start_times.sort()
    grided_radar.append(start_times)
                    
    return grided_radar, start_times

def GridRadar(radar_objs):
        grided = []
        for radars in radar_objs:
                print("Griding: " + str(radars))
                grided.append(pyart.map.grid_from_radars(radars, grid_shape=(20, 181, 181), grid_limits=((100, 15000), (-175000, 175000), (-155000, 155000)), grid_origin=(pyart.graph.RadarMapDisplay(radars).loc[0], pyart.graph.RadarMapDisplay(radars).loc[1]), fields=["reflectivity", 'spectrum_width', 'differential_reflectivity', 'cross_correlation_ratio', 'velocity', 'differential_phase']))
        return grided
    
def kdpData(trad_sweep):
        #run this function only once!!
        kdpdata = pyart.retrieve.kdp_vulpiani(trad_sweep,  phidp_field ='differential_phase',  band ='S' , windsize = 34)
        kdpdata = kdpdata[0]['data'] #Retreive the kdp 'data' from the the kdpdata that was calculated using the Vulpiani method 
        mask = np.logical_and(kdpdata > -0.01, kdpdata < 0.01) #mask the values from -0.01 to 0.01 so they do not plot
        kdpdata = np.where(mask, np.nan, kdpdata) #Apply the mask to kdpdata
        trad_sweep.add_field('specific_differential_phase_hv', {'data': kdpdata.data})  #Add a new field to the radar data dictionary with the kdp data
        return kdpdata

def xsecPoints(mlon, mlat, xlon, xlat, maxz, radar_data, spacing = 0.1):
        
        radarx = radar_data[0].radar_longitude['data'][0]
        radary = radar_data[0].radar_latitude['data'][0]
        
        mlon_rad, xlon_rad = haversine(mlat, mlon, mlat, radarx)/1e3, haversine(xlat, xlon, xlat, radarx)/1e3
        mlat_rad, xlat_rad = haversine(radary, mlon, mlat, mlon)/1e3, haversine(radary, xlon, xlat, xlon)/1e3

        if negativeaxis(mlon, radar_data): mlon_rad = mlon_rad*-1
        if negativeaxis(mlat, radar_data): mlat_rad = mlat_rad*-1
        if negativeaxis(xlon, radar_data): xlon_rad = xlon_rad*-1
        if negativeaxis(xlat, radar_data): xlat_rad = xlat_rad*-1
        
        nn = int(((xlon_rad-mlon_rad)**2+(xlat_rad-mlat_rad)**2)**0.5/spacing)
        des_x = np.linspace(mlon_rad,xlon_rad,nn)*1e3
        des_y = np.linspace(mlat_rad,xlat_rad,nn)*1e3
        des_z = np.arange(0,maxz+spacing,spacing)*1e3
        
        desx_grid,desz_grid = np.meshgrid(des_x,des_z)
        desy_grid,desz_grid = np.meshgrid(des_y,des_z)
        new_x = np.arange(0,np.shape(des_x)[0]*spacing ,spacing)*1e3
        _new_x = new_x
        return desx_grid, desy_grid, desz_grid, new_x

def keepSweep(radar_data):
        #Must be a list of radar objects
        radar_pol=[]
        radar_trad=[]
        for radar in radar_data:     
                keep_sweeps_pol=[]
                keep_sweeps_trad=[]
                for sweep in radar.sweep_number['data']:
                    if not (np.size(radar.extract_sweeps([sweep]).fields['differential_reflectivity']['data'].mask)-
                             np.sum(radar.extract_sweeps([sweep]).fields['differential_reflectivity']['data'].mask))==0:
                        #print ('Keeping polarimetric sweep number: ', sweep)
                        keep_sweeps_pol+=[sweep]
                    if not (np.size(radar.extract_sweeps([sweep]).fields['velocity']['data'].mask)-
                             np.sum(radar.extract_sweeps([sweep]).fields['velocity']['data'].mask))==0:
                        #print ('Keeping traditional sweep number: ', sweep)
                        keep_sweeps_trad+=[sweep]

                # Keep only the good sweeps for each set of radar variables
                radar_pol.append(radar.extract_sweeps(keep_sweeps_pol))
                radar_trad.append(radar.extract_sweeps(keep_sweeps_trad))
        return radar_pol, radar_trad

def plotRadar(radar_data, variable, mlon=0, mlat=0, xlon=0, xlat=0):
        fig = plt.figure(figsize = (15, 10))
        ax = plt.subplot(projection = ccrs.PlateCarree())
        #ax = fig.add_subplot()
        display = pyart.graph.RadarMapDisplay(radar_data)
        #variable must be a pyart defined radar variable
        #can be found by using radar.into(fields)
        display.plot_ppi_map(variable, sweep = 1, vmin = -20, vmax=60, cmap='pyart_HomeyerRainbow',
                             min_lat=radar_data.latitude['data'][0]-2, 
                             max_lat=radar_data.latitude['data'][0]+2, 
                             min_lon=radar_data.longitude['data'][0]-2, 
                             max_lon=radar_data.longitude['data'][0]+2, 
                             lat_lines = [40, 41, 42, 43, 44, 45], lon_lines = [-74, -75, -76, -77],
                             ax = ax)
                
        ax.add_feature(COUNTIES, facecolor='none', edgecolor='gray')
        ax.add_feature(cfeature.BORDERS)
        plt.plot([mlon, xlon], [mlat, xlat], color = '#6a8f9c', linewidth = 2)
        
        
def plotRadarxsec(radar_data, pol_sweep, trad_sweep, desx_grid, desy_grid, desz_grid, new_x, kdpdata, _maxz = 10):
        
        des_tree_pol = spatial.KDTree(np.array([pol_sweep.gate_x['data'].ravel(),pol_sweep.gate_y['data'].ravel(), pol_sweep.gate_z['data'].ravel()]).T)

        des_tree_trad = spatial.KDTree(np.array([trad_sweep.gate_x['data'].ravel(),trad_sweep.gate_y['data'].ravel(), trad_sweep.gate_z['data'].ravel()]).T)
        # Find the radar gate closest to the cross-section grid
        dists_pol, indext_pol = des_tree_pol.query(np.array([desx_grid.ravel(), desy_grid.ravel(), desz_grid.ravel()]).T)

        # Find the radar gate closest to the cross-section grid
        dists_trad, indext_trad = des_tree_trad.query(np.array([desx_grid.ravel(),  desy_grid.ravel(),  desz_grid.ravel()]).T)

        # Get the x,y,z locations of the radar gates for later
        # Theoretically these should match, but let's not make any assumptions
        rx_pol, ry_pol, rz_pol  = pol_sweep.get_gate_x_y_z(0)
        rx_trad,ry_trad,rz_trad = trad_sweep.get_gate_x_y_z(0)
        
        fig = plt.figure(figsize=(28,16))
        #Reflectivity
        ax1 = fig.add_subplot(321)
        plt.contourf(new_x/1e3, desz_grid[:,0]/1e3, trad_sweep.fields['reflectivity']['data'].ravel()[indext_trad].reshape(np.shape(desz_grid)),levels = np.arange(-20,72,1),cmap='pyart_HomeyerRainbow')
        plt.colorbar(label='Reflectivity (dBZ)')
        plt.xlim(0, np.max(new_x)/1e3)
        plt.ylim(0, _maxz)
        #Velocity
        ax2 = fig.add_subplot(322)
        plt.contourf(new_x/1e3, desz_grid[:,0]/1e3, trad_sweep.fields['velocity']['data'].ravel()[indext_trad].reshape(np.shape(desz_grid)),levels = np.arange(-40,40,1),cmap='NWSVel')
        plt.colorbar(label='Velocity (m/s)')
        plt.xlim(0, np.max(new_x)/1e3)
        plt.ylim(0, _maxz)
        #Spectrum Width
        ax3 = fig.add_subplot(323)
        colors1 = plt.cm.binary_r(np.linspace(0.2,0.8,33))
        colors2 = plt.cm.gnuplot_r(np.linspace(0.,0.7,100))
        colors = np.vstack((colors1, colors2[10:121]))
        swcolours = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        plt.contourf(new_x/1e3, desz_grid[:,0]/1e3, trad_sweep.fields['spectrum_width']['data'].ravel()[indext_trad].reshape(np.shape(desz_grid)),levels = np.arange(0,14,0.1),cmap='NWS_SPW')
        plt.colorbar(label='Spectrum Width')
        plt.xlim(0, np.max(new_x)/1e3)
        #Differential Reflectivity
        ax4 = fig.add_subplot(324)
        plt.contourf(new_x/1e3, desz_grid[:,0]/1e3, pol_sweep.fields['differential_reflectivity']['data'].ravel()[indext_pol].reshape(np.shape(desz_grid)),levels = np.arange(-4,8,0.1),cmap='ChaseSpectral')
        plt.colorbar(label='Differential Reflectivity (ZDR)')
        plt.xlim(0, np.max(new_x)/1e3)
        plt.ylim(0, _maxz)
        #Correlation Coefficient
        ax5 = fig.add_subplot(325)
        plt.contourf(new_x/1e3, desz_grid[:,0]/1e3, pol_sweep.fields['cross_correlation_ratio']['data'].ravel()[indext_pol].reshape(np.shape(desz_grid)),levels = np.arange(0,1.1,0.01),cmap='SCook18')
        plt.colorbar(label='Correlation Coefficient')
        plt.xlim(0, np.max(new_x)/1e3)
        plt.ylim(0, _maxz)
        #Specific Differential Phase
        ax6 = fig.add_subplot(326)
        plt.contourf(new_x/1e3, desz_grid[:,0]/1e3, kdpdata.ravel()[indext_pol].reshape(np.shape(desz_grid)),np.arange(-1,4.1,0.1),cmap=swcolours)
        plt.colorbar(label='Specific Differential Phase (KDP)')
        plt.xlim(0, np.max(new_x)/1e3)
        plt.ylim(0, _maxz)
                 
                 
        plt.ylabel('Altitude (km AGL)')
        plt.xlabel('Distance (km)')
        plt.tight_layout()
        
        
        
        