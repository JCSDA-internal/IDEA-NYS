#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 13:16:40 2021

@author: whung
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from functools import reduce
import os, fnmatch
from datetime import datetime
from mpl_toolkits.basemap import Basemap

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)



'''Settings'''
path_SNPP   = '/home/idea/IDEA-I_aerosol/products/CONUS/Aerosol/SNPP'
path_JPSS   = '/home/idea/IDEA-I_aerosol/products/CONUS/Aerosol/JPSS'
path_high   = '/home/idea/IDEA-I_aerosolEntHR/products/NORTHEAST/AerosolEntHR/SNPP'
path_output = '/lulab/weiting/AQ_risk_map/products'
shp_USA = '/lulab/weiting/shp_files/gadm36_USA_shp/gadm36_USA_1'
shp_CAN = '/lulab/weiting/shp_files/gadm36_CAN_shp/gadm36_CAN_1'

## date
today    = '20200925'
startday = (pd.to_datetime(today)-pd.Timedelta('2 days')).strftime('%Y%m%d')
date     = pd.date_range(start=startday, end=today, freq='d')[::-1]
date_factor = [1, 0.9, 0.8]
DD = len(date)

## forecast time
time_window = [6, 12]    # forecast time, 6h time window
tt0 = time_window[0]
tt1 = time_window[1]

## height threshold, only trajectories lower than this threshold are considered
height_lim  = 850   # hPa

## NY domain
lon_lim = [-80, -72]
lat_lim = [40.5, 45.5]
grid_interval    = 0.1    # degree
xx_grid, yy_grid = np.meshgrid(np.arange(lon_lim[0], lon_lim[1]+grid_interval, grid_interval), \
                               np.arange(lat_lim[0], lat_lim[1]+grid_interval, grid_interval))
#print(xx_grid, yy_grid)
#print(xx_grid.shape, yy_grid.shape)

## initial/forecast time
initial_time  = (date[0]+pd.Timedelta('12 hours')).strftime('%Y%m%d %H00')
forecast_time = (date[0]+pd.Timedelta('12 hours')+pd.Timedelta('12 hours')).strftime('%Y%m%d %H00')
#print(initial_time, forecast_time)

## colormap
norm = cls.Normalize(vmin=0, vmax=20)
tick = np.arange(0, 20+1, 4)

## lifetime
## C = C0*exp(-kt), -kt=-1 when t=lifetime
## Colarco et al. (2010JGR): 8.82 days for BC and 6.90 days for OC
## Reid et al. (2005): smoke aersols are composed of 5-10% BC and 50-60% OC -> 1:10
lifetime     = 8.82*(1/11)+6.9*(10/11)    # 7 days
k_lifetime   = 1/lifetime
decay_factor = np.exp([0, -1*k_lifetime, -2*k_lifetime])


'''Functions'''
def file_finder(path, date, filename):
    grid_file = []
    traj_file = []
    for dd in date.strftime('20%y%m%d'):
        if os.path.isdir(path+'/'+dd) == False:
            grid_file = np.append(grid_file, 'no-data')
            traj_file = np.append(traj_file, 'no-data')
        else:
            for root, dirs, files in os.walk(path+'/'+dd):
                for name in fnmatch.filter(files, filename+'_grid_*_'+dd+'.nc'):
                    grid_file = np.append(grid_file, os.path.join(root, name))
                for name in fnmatch.filter(files, filename+'_traj_*_'+dd+'.nc'):
                    traj_file = np.append(traj_file, os.path.join(root, name))
            del [root, dirs, files, name]
    return grid_file, traj_file

def grid_finder(xgrid, ygrid, x, y):
    dis = (xgrid-x)**2+(ygrid-y)**2
    return np.squeeze(np.argwhere(dis==np.min(dis)))


    
'''Check Needed Files'''
grid_fileS, traj_fileS = file_finder(path_SNPP, date, 'VIIRSaerosolS')
grid_fileJ, traj_fileJ = file_finder(path_JPSS, date, 'VIIRSaerosolJ')
grid_fileH, traj_fileH = file_finder(path_high, date[:-1], 'VIIRSaerosolEntHRS')
member_norm = 1/(len(grid_fileS)+len(grid_fileJ)+len(grid_fileH))

if (len(np.argwhere(grid_fileS == 'no-data'))!=0) or (len(np.argwhere(grid_fileJ == 'no-data'))!=0) or (len(np.argwhere(grid_fileH == 'no-data'))!=0): 
    print('== NO AVAILABLE IDEA-NYS OR IDEA-NYS-HR OUTPUTS')
    exit()



print('==', today, str(tt0)+'-'+str(tt1)+'H RISK FORECAST')
print('== INITIAL:', initial_time, 'FORECAST:', forecast_time)
print('== START TIME',  datetime.now())
print('Member number: ', int(1/member_norm))
print(grid_fileS)
print(grid_fileJ)
print(grid_fileH)
#print(member_norm)
#exit()




'''LOW RESOL'''
## == Reading Data ==
for i in np.arange(DD):
    if i == 0:       # today
        ## mete grid
        readin      = Dataset(grid_fileS[i])
        longitude_L = readin['Longitude'][:]
        latitude_L  = readin['Latitude'][:]
        longitude_L = longitude_L[(longitude_L >= lon_lim[0]) & (longitude_L <= lon_lim[1])]
        latitude_L  = latitude_L[(latitude_L >= lat_lim[0]) & (latitude_L <= lat_lim[1])]
        del readin

        ## trajectory
        readin = Dataset(traj_fileS[i])
        index  = np.squeeze(np.argwhere((readin['time'][:]>tt0) & (readin['time'][:]<=tt1)))
        lon_S  = readin['xtraj'][index, :]
        lat_S  = readin['ytraj'][index, :]
        lev_S  = readin['ptraj'][index, :]    # pressure level
        time_S = readin['time'][index]        # forecast hour
        aod_S  = readin['aod_traj'][:]        # initial AOD
        del [readin, index]

        readin = Dataset(traj_fileJ[i])
        index  = np.squeeze(np.argwhere((readin['time'][:]>tt0) & (readin['time'][:]<=tt1)))
        lon_J  = readin['xtraj'][index, :]
        lat_J  = readin['ytraj'][index, :]
        lev_J  = readin['ptraj'][index, :]    # pressure level
        time_J = readin['time'][index]        # forecast hour
        aod_J  = readin['aod_traj'][:]        # initial AOD
        del [readin, index]


    elif i == 1:    # day-1
        ## trajectory
        readin = Dataset(traj_fileS[i])
        index  = np.squeeze(np.argwhere((readin['time'][:]>tt0+24) & (readin['time'][:]<=tt1+24)))
        lon_S  = readin['xtraj'][index, :]
        lat_S  = readin['ytraj'][index, :]
        lev_S  = readin['ptraj'][index, :]    # pressure level
        time_S = readin['time'][index]        # forecast hour
        aod_S  = readin['aod_traj'][:]        # initial AOD
        del [readin, index]

        readin = Dataset(traj_fileJ[i])
        index  = np.squeeze(np.argwhere((readin['time'][:]>tt0+24) & (readin['time'][:]<=tt1+24)))
        lon_J  = readin['xtraj'][index, :]
        lat_J  = readin['ytraj'][index, :]
        lev_J  = readin['ptraj'][index, :]    # pressure level
        time_J = readin['time'][index]        # forecast hour
        aod_J  = readin['aod_traj'][:]        # initial AOD
        del [readin, index]
        #print(time_S, time_J)


    elif i == 2:    # day-2
        ## trajectory
        readin = Dataset(traj_fileS[i])
        index  = np.squeeze(np.argwhere((readin['time'][:]>tt0+48) & (readin['time'][:]<=tt1+48)))
        lon_S  = readin['xtraj'][index, :]
        lat_S  = readin['ytraj'][index, :]
        lev_S  = readin['ptraj'][index, :]    # pressure level
        time_S = readin['time'][index]        # forecast hour
        aod_S  = readin['aod_traj'][:]        # initial AOD
        del [readin, index]

        readin = Dataset(traj_fileJ[i])
        index  = np.squeeze(np.argwhere((readin['time'][:]>tt0+48) & (readin['time'][:]<=tt1+48)))
        lon_J  = readin['xtraj'][index, :]
        lat_J  = readin['ytraj'][index, :]
        lev_J  = readin['ptraj'][index, :]    # pressure level
        time_J = readin['time'][index]        # forecast hour
        aod_J  = readin['aod_traj'][:]        # initial AOD
        del [readin, index]


    ## choose traj pass through NYS
    ## only consider traj lower than 850hPa
    valid_S = []
    for j in np.arange(len(aod_S)):
        index1 = np.argwhere((lon_S[:, j]>=lon_lim[0]) & (lon_S[:, j]<=lon_lim[1]))
        index2 = np.argwhere((lat_S[:, j]>=lat_lim[0]) & (lat_S[:, j]<=lat_lim[1]))
        index  = np.intersect1d(index1, index2)
        if (len(index) != 0) & (len(np.argwhere(lev_S[index, j] > height_lim)) != 0):
            valid_S = np.append(valid_S, j)
        del [index1, index2, index]

    valid_J = []
    for j in np.arange(len(aod_J)):
        index1 = np.argwhere((lon_J[:, j]>=lon_lim[0]) & (lon_J[:, j]<=lon_lim[1]))
        index2 = np.argwhere((lat_J[:, j]>=lat_lim[0]) & (lat_J[:, j]<=lat_lim[1]))
        index  = np.intersect1d(index1, index2)
        if (len(index) != 0) & (len(np.argwhere(lev_J[index, j] > height_lim)) != 0):
            valid_J = np.append(valid_J, j)
        del [index1, index2, index]

    print('DAY '+str(int(i*(-1)))+' VALID TRAJ')
    print('SNPP', len(valid_S), 'JPSS', len(valid_J))
    #print(valid_S, valid_J)


    ## == Risk Map ==
    NN_S = len(valid_S)
    NN_J = len(valid_J)
    risk_S = np.zeros(xx_grid.shape)
    risk_J = np.zeros(xx_grid.shape)

    ## SNPP
    if NN_S != 0:
        lon_S = lon_S[:, valid_S.astype(int)]
        lat_S = lat_S[:, valid_S.astype(int)]
        lev_S = lev_S[:, valid_S.astype(int)]
        aod_S = aod_S[valid_S.astype(int)]
        del valid_S

        for j in np.arange(NN_S):
            index1 = np.argwhere((lon_S[:, j]>=lon_lim[0]) & (lon_S[:, j]<=lon_lim[1]))
            index2 = np.argwhere((lat_S[:, j]>=lat_lim[0]) & (lat_S[:, j]<=lat_lim[1]))
            index3 = np.argwhere(lev_S[:, j] > height_lim)
            index  = reduce(np.intersect1d, (index1, index2, index3))

            for hh in index:    # forecast hour, 3x3 grid?
                x, y         = grid_finder(xx_grid, yy_grid, lon_S[hh, j], lat_S[hh, j])
                risk_S[x, y] = risk_S[x, y]+(aod_S[j]*decay_factor[i]*date_factor[i])
                #risk_S[x-1:x+2, y-1:y+2] = risk_S[x-1:x+2, y-1:y+2]+(aod_S[j]*date_factor[i])
                del [x, y]
            del [hh, index1, index2, index3, index]

    ## JPSS
    if NN_J != 0:
        lon_J = lon_J[:, valid_J.astype(int)]
        lat_J = lat_J[:, valid_J.astype(int)]
        lev_J = lev_J[:, valid_J.astype(int)]
        aod_J = aod_J[valid_J.astype(int)]
        del valid_J

        for j in np.arange(NN_J):
            index1 = np.argwhere((lon_J[:, j]>=lon_lim[0]) & (lon_J[:, j]<=lon_lim[1]))
            index2 = np.argwhere((lat_J[:, j]>=lat_lim[0]) & (lat_J[:, j]<=lat_lim[1]))
            index3 = np.argwhere(lev_J[:, j] > height_lim)
            index  = reduce(np.intersect1d, (index1, index2, index3))

            for hh in index:    # forecast hour, 3x3 grid?
                x, y         = grid_finder(xx_grid, yy_grid, lon_J[hh, j], lat_J[hh, j])
                risk_J[x, y] = risk_J[x, y]+(aod_J[j]*decay_factor[i]*date_factor[i])
                #risk_J[x-1:x+2, y-1:y+2] = risk_J[x-1:x+2, y-1:y+2]+(aod_J[j]*date_factor[i])
                del [x, y]
            del [hh, index1, index2, index3, index]

    #print(np.nanmin(risk_S), np.nanmax(risk_S))
    #print(np.nanmin(risk_J), np.nanmax(risk_J))

    if i == 0:
        total_num_S  = np.copy(NN_S)
        total_num_J  = np.copy(NN_J)
        total_risk_S = np.copy(risk_S)*member_norm
        total_risk_J = np.copy(risk_J)*member_norm
    else:
        total_num_S  = np.append(total_num_S, NN_S)
        total_num_J  = np.append(total_num_J, NN_J)
        total_risk_S = total_risk_S+(risk_S*member_norm)
        total_risk_J = total_risk_J+(risk_J*member_norm)
    del [NN_S, NN_J, risk_S, risk_J]

#print(total_num_S, total_num_J)
#print(np.nanmin(total_risk_S), np.nanmax(total_risk_S))
#print(np.nanmin(total_risk_J), np.nanmax(total_risk_J))

#plt.figure()
#plt.title('Low resol SNPP')
#plt.pcolor(xx_grid, yy_grid, total_risk_S, cmap='YlOrRd')
#plt.colorbar()
#plt.show()
#exit()

print('== LOW RESOL DONE')


'''HIGH RESOL'''
## == Reading Data ==
for i in np.arange(DD-1):
    if i == 0:       # today
        ## mete field
        readin   = Dataset(grid_fileH[i])
        longitude_H = readin['Longitude'][:]
        latitude_H  = readin['Latitude'][:]
        del readin
        #print(longitude_L.shape, latitude_L.shape)

        ## trajectory
        readin = Dataset(traj_fileH[i])
        index  = np.squeeze(np.argwhere((readin['time'][:]>tt0) & (readin['time'][:]<=tt1)))
        lon_H  = readin['xtraj'][index, :]
        lat_H  = readin['ytraj'][index, :]
        lev_H  = readin['ptraj'][index, :]    # pressure level
        time_H = readin['time'][index]        # forecast hour
        aod_H  = readin['aod_traj'][:]        # initial AOD
        del [readin, index]

    elif i == 1:    # day-1
        ## trajectory
        readin = Dataset(traj_fileH[i])
        index  = np.squeeze(np.argwhere((readin['time'][:]>tt0+24) & (readin['time'][:]<=tt1+24)))
        lon_H  = readin['xtraj'][index, :]
        lat_H  = readin['ytraj'][index, :]
        lev_H  = readin['ptraj'][index, :]    # pressure level
        time_H = readin['time'][index]        # forecast hour
        aod_H  = readin['aod_traj'][:]        # initial AOD
        del [readin, index]


    ## choose traj pass through NYS
    ## only consider traj lower than 850hPa
    valid_H = []
    for j in np.arange(len(aod_H)):
        index1 = np.argwhere((lon_H[:, j]>=lon_lim[0]) & (lon_H[:, j]<=lon_lim[1]))
        index2 = np.argwhere((lat_H[:, j]>=lat_lim[0]) & (lat_H[:, j]<=lat_lim[1]))
        index  = np.intersect1d(index1, index2)
        if (len(index) != 0) & (len(np.argwhere(lev_H[index, j] > height_lim)) != 0):
            valid_H = np.append(valid_H, j)
        del [index1, index2, index]

    print('DAY '+str(int(i*(-1)))+' VALID TRAJ')
    print('SNPP', len(valid_H))
    #print(valid_H)


    ## == Risk Map ==
    NN_H = len(valid_H)
    risk_H = np.zeros(xx_grid.shape)
    if NN_H != 0:
        lon_H = lon_H[:, valid_H.astype(int)]
        lat_H = lat_H[:, valid_H.astype(int)]
        lev_H = lev_H[:, valid_H.astype(int)]
        aod_H = aod_H[valid_H.astype(int)]
        del valid_H

        for j in np.arange(NN_H):
            index1 = np.argwhere((lon_H[:, j]>=lon_lim[0]) & (lon_H[:, j]<=lon_lim[1]))
            index2 = np.argwhere((lat_H[:, j]>=lat_lim[0]) & (lat_H[:, j]<=lat_lim[1]))
            index3 = np.argwhere(lev_H[:, j] > height_lim)
            index  = reduce(np.intersect1d, (index1, index2, index3))

            for hh in index:    # forecast hour, 3x3 grid?
                x, y         = grid_finder(xx_grid, yy_grid, lon_H[hh, j], lat_H[hh, j])
                risk_H[x, y] = risk_H[x, y]+(aod_H[j]*decay_factor[i]*date_factor[i])
                #risk_H[x-1:x+2, y-1:y+2] = risk_H[x-1:x+2, y-1:y+2]+(aod_H[j]*date_factor[i])
                del [x, y]
            del [hh, index1, index2, index3, index]

    #print(np.nanmin(risk_H), np.nanmax(risk_H))

    if i == 0:
        total_num_H  = np.copy(NN_H)
        total_risk_H = np.copy(risk_H)*member_norm
    else:
        total_num_H  = np.append(total_num_H, NN_H)
        total_risk_H = total_risk_H+(risk_H*member_norm)
    del [NN_H, risk_H]

#print(total_num_H)
#print(np.nanmin(total_risk_H), np.nanmax(total_risk_H))

#plt.figure()
#plt.title('High resol SNPP')
#plt.pcolor(xx_grid, yy_grid, total_risk_H, cmap='YlOrRd')
#plt.colorbar()
#plt.show()

print('== HIGH RESOL DONE')



'''ENSEMBLE RISK MAP'''
total_num  = np.append(np.append(total_num_S, total_num_J), total_num_H).astype(int)
total_risk = total_risk_S + total_risk_J + total_risk_H
total_risk[total_risk==0] = np.nan

#print(np.nanmin(total_risk), np.nanmax(total_risk))

#plt.figure()
#plt.title('Total risk')
#plt.pcolor(xx_grid, yy_grid, total_risk_H, cmap='YlOrRd')
#plt.colorbar()
#plt.show()
#exit()



if os.path.isdir(path_output+'/'+today) == False:
    os.mkdir(path_output+'/'+today)


'''Writing NetCDF'''
output = Dataset(path_output+'/'+today+'/VIIRSaerosol_RiskForecast_'+initial_time.replace(' ','')[:-2]+'_'+forecast_time.replace(' ','')[:-2]+'.nc', 'w')
output.createDimension('lon', xx_grid.shape[0])
output.createDimension('lat', xx_grid.shape[1])
output.createDimension('time', 1)    # forecast time
output.createDimension('member_num', int(1/member_norm))
var_xx   = output.createVariable('xgrid', 'float', ('lon', 'lat'))
var_yy   = output.createVariable('ygrid', 'float', ('lon', 'lat'))
var_risk = output.createVariable('risk', 'float', ('lon', 'lat'))
var_time = output.createVariable('forecast_time', 'i4', ('time', ))
var_num  = output.createVariable('valid_traj_num', 'i4', ('member_num', ))
var_num.long_name = 'Number of valid trajectories: Low-SNPP day 0~-2, Low-JPSS day 0~-2, High-SNPP day 0~-1'
var_xx[:]   = xx_grid
var_yy[:]   = yy_grid
var_risk[:] = total_risk
var_time[:] = int(forecast_time.replace(' ','')[:-2])
var_num[:]  = total_num
output.close()
print('== NetCDF DONE')

    
'''Plotting'''
fig, ax = plt.subplots(figsize=(12,9))    # unit=100pixel
h = ax.get_position()
ax.set_position([h.x0-0.04, h.y0+0.02, h.width+0.06, h.height+0.04])
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2.5)

plt.annotate('IDEA-NYS '+str(tt0)+'-'+str(tt1)+'h Risk Forecast', (h.x0-0.04, h.y0+0.02+h.height+0.03), fontsize=22, fontweight='bold', xycoords='figure fraction')
plt.annotate('Forecast member: '+str(int(1/member_norm))+'; Valid trajectory: '+str(sum(total_num)), (h.x0-0.04, h.y0+0.02+h.height), fontsize=18, xycoords='figure fraction')
plt.annotate('Initial at '+initial_time+' UTC\nValid  at '+forecast_time+' UTC', (h.x0-0.04+h.width-0.21, h.y0+0.02+h.height), fontsize=16, xycoords='figure fraction')

m = Basemap(llcrnrlon=lon_lim[0],urcrnrlon=lon_lim[-1],llcrnrlat=lat_lim[0],urcrnrlat=lat_lim[-1], projection='cyl', resolution='l')
#m.arcgisimage(service='World_Physical_Map', xpixels=2000, verbose=True)
m.readshapefile(shp_USA, 'USA', color='k')
m.readshapefile(shp_CAN, 'CAN', color='k')
m.drawmeridians(np.arange(lon_lim[0], lon_lim[-1]+1, 2), color='none', labels=[0,0,0,1], fontsize=22)
m.drawparallels(np.arange(round(lat_lim[0]), lat_lim[-1]+1, 2), color='none', labels=[1,0,0,0], fontsize=22)

x, y = m(xx_grid, yy_grid)
cs = m.pcolor(x, y, total_risk, cmap='YlOrRd', norm=norm, alpha=0.7)    # today

cbaxes = fig.add_axes([h.x0-0.04, h.y0, h.width+0.06, 0.02])        
cb     = plt.colorbar(cs, ticks=tick, extend='max', orientation='horizontal', cax=cbaxes)
cb.set_label('Risk value', fontsize=22, fontweight='bold')
cb.ax.tick_params(labelsize=22, length=12, width=2, direction='in')
cb.outline.set_linewidth(2)

#plt.show()
#exit()

plt.savefig(path_output+'/'+today+'/VIIRSaerosol_risk_map_'+initial_time.replace(' ','')[:-2]+'_'+forecast_time.replace(' ','')[:-2]+'.png')
plt.close()
del [fig, ax, h, m, x, y, cs, cb]

print('== END TIME', datetime.now())
