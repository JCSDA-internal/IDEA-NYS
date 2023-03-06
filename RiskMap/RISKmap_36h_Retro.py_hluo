# Retro 36 hr (30-36) risk forecast based on low resolution IDEA runs
# Modified by Huiying Luo based on Wei-Ting Hung's 30hr forecast
# define date in Line 97-100

import os
os.environ["PROJ_LIB"]='C:\\Users\\Huiying\\Anaconda3\\envs\\luopy\\Library\\share\\basemap'

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from functools import reduce
import os, fnmatch
from datetime import datetime
from mpl_toolkits.basemap import Basemap
from datetime import date
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

'''Settings'''

# path_SNPP   = '/home/idea/IDEA-I_aerosol/products/CONUS/Aerosol/SNPP'
# path_JPSS   = '/home/idea/IDEA-I_aerosol/products/CONUS/Aerosol/JPSS'
# path_high   = '/home/idea/IDEA-I_aerosolEntHR/products/NORTHEAST/AerosolEntHR/SNPP'
# path_output = '/lulab/hluo/AQ_risk_map/products'
# shp_USA = '/lulab/hluo/AQ_risk_map/gadm36_USA_shp/gadm36_USA_1'
# shp_CAN = '/lulab/hluo/AQ_risk_map/gadm36_CAN_shp/gadm36_CAN_1'


path_SNPP = 'G:\\viirs\\IDEA'
path_JPSS = 'G:\\viirs\\IDEA'
path_high = 'G:\\viirs\\IDEA'
path_output = 'C:\Projects\WRF_SUNY\RISK\products'
shp_USA = 'C:\Projects\WRF_SUNY\DataPro\shp_files\gadm36_USA_shp\gadm36_USA_1'
shp_CAN = 'C:\Projects\WRF_SUNY\DataPro\shp_files\gadm36_CAN_shp\gadm36_CAN_1'

##################################################
date_factor = [1, 1, 1]

##################################################
time_window = [30,36]  # forecast time, 6h time window
tt0 = time_window[0]
tt1 = time_window[1]

## height threshold, only trajectories lower than this threshold are considered
height_lim = 850  # hPa

## NY domain
lon_lim = [-80, -72]
lat_lim = [40, 45.5] # grid center; limits included

grid_interval = 0.1  # degree
xx_grid, yy_grid = np.meshgrid(np.arange(lon_lim[0], lon_lim[1] + grid_interval, grid_interval), \
                               np.arange(lat_lim[0], lat_lim[1] + grid_interval, grid_interval))
# print(xx_grid, yy_grid)
# print(xx_grid.shape, yy_grid.shape)

## colormap
norm = cls.Normalize(vmin=0, vmax=5)
tick = np.arange(0, 6, 1)

## lifetime
## C = C0*exp(-kt), -kt=-1 when t=lifetime
## Colarco et al. (2010JGR): 8.82 days for BC and 6.90 days for OC
## Reid et al. (2005): smoke aersols are composed of 5-10% BC and 50-60% OC -> 1:10
lifetime = 8.82 * (1 / 11) + 6.9 * (10 / 11)  # 7 days
k_lifetime = 1 / lifetime
decay_factor = np.exp([0, -1 * k_lifetime, -2 * k_lifetime])
print(decay_factor)


'''Functions'''

def file_finder(path, date, filename):
    grid_file = []
    traj_file = []
    for dd in date.strftime('20%y%m%d'):

        for root, dirs, files in os.walk(path):
            for name in fnmatch.filter(files, filename + '_grid_*_' + dd + '.nc'):
                grid_file = np.append(grid_file, os.path.join(root, name))
            for name in fnmatch.filter(files, filename + '_traj_*_' + dd + '.nc'):
                traj_file = np.append(traj_file, os.path.join(root, name))
        del [root, dirs, files]
    return grid_file, traj_file


def grid_finder(xgrid, ygrid, x, y):
    dis = (xgrid - x) ** 2 + (ygrid - y) ** 2
    return np.squeeze(np.argwhere(dis == np.min(dis)))

'''Date settings for retro runs'''

for yearn in ['2019','2020']:  # , ]:#
    # time
    date_start = yearn + '0701'
    date_end = yearn + '0930'
    datelist = pd.date_range(start=date_start, end=date_end, freq='D')  # datelist include date_end

    for d in datelist:
        tempd = pd.to_datetime(str(d))
        today = tempd.strftime('%Y%m%d') ## forecast base date

        # if NRT
        # dt = date.today()
        # today = dt.strftime("%Y%m%d")

        startday = (pd.to_datetime(today) - pd.Timedelta('1 days')).strftime('%Y%m%d')
        date = pd.date_range(start=startday, end=today, freq='d')[::-1]

        DD = len(date)

        ## initial/forecast time
        initial_time = (date[0] + pd.Timedelta('12 hours')).strftime('%Y%m%d %H00')
        ##########################################
        forecast_times = (date[0] + pd.Timedelta('12 hours') + pd.Timedelta('30 hours')).strftime('%Y%m%d %H00')
        forecast_time = (date[0] + pd.Timedelta('12 hours') + pd.Timedelta('36 hours')).strftime('%Y%m%d %H00')
        # print(initial_time, forecast_time)

        '''Check Needed Files'''
        grid_fileS, traj_fileS = file_finder(path_SNPP, date, 'VIIRSaerosolS')
        # print(path_SNPP)
        # print(date)
        # print(grid_fileS)
        # print(traj_fileS)

        grid_fileJ, traj_fileJ = file_finder(path_JPSS, date, 'VIIRSaerosolJ')

        if ((len(grid_fileS) + len(grid_fileJ) )== 0):#(len(np.argwhere(grid_fileS == 'no-data')) != 0) or (len(np.argwhere(grid_fileJ == 'no-data')) != 0) or (len(np.argwhere(grid_fileH == 'no-data')) != 0)
            print('== NO AVAILABLE IDEA-NYS OUTPUTS for '+today)
            print('')
        else:
            member_norm = 1 / (len(grid_fileS)+ len(grid_fileJ))####################

            print('==', today, str(tt0) + '-' + str(tt1) + 'H RISK FORECAST')
            print('== INITIAL:', initial_time, 'FORECAST:', forecast_time)
            print('== START TIME', datetime.now())
            print('Member number: ', int(1 / member_norm))
            print(grid_fileS)
            print(grid_fileJ)
            # print(grid_fileH)
            # print(member_norm)

            '''LOW RESOL'''
            ## == Reading Data ==
            ## choose traj pass through NYS
            ## only consider traj lower than 850hPa

            for i in np.arange(len(grid_fileS)):
                if today in grid_fileS[i]:  # today
                    ## mete grid
                    # readin = Dataset(grid_fileS[i])
                    # longitude_L = readin['Longitude'][:]
                    # latitude_L = readin['Latitude'][:]
                    # longitude_L = longitude_L[(longitude_L >= lon_lim[0]) & (longitude_L <= lon_lim[1])]
                    # latitude_L = latitude_L[(latitude_L >= lat_lim[0]) & (latitude_L <= lat_lim[1])]
                    # del readin

                    ## trajectory
                    readin = Dataset(traj_fileS[i])
                    index = np.squeeze(np.argwhere((readin['time'][:] > tt0) & (readin['time'][:] <= tt1)))
                    lon_S = readin['xtraj'][index, :]
                    lat_S = readin['ytraj'][index, :]
                    lev_S = readin['ptraj'][index, :]  # pressure level
                    time_S = readin['time'][index]  # forecast hour
                    aod_S = readin['aod_traj'][:]  # initial AOD
                    del [readin, index]
                    decay_factorf=decay_factor[0]

                else: #-1 day
                    ## trajectory
                    readin = Dataset(traj_fileS[i])
                    index = np.squeeze(np.argwhere((readin['time'][:] > tt0 + 24) & (readin['time'][:] <= tt1 + 24)))
                    lon_S = readin['xtraj'][index, :]
                    lat_S = readin['ytraj'][index, :]
                    lev_S = readin['ptraj'][index, :]  # pressure level
                    time_S = readin['time'][index]  # forecast hour
                    aod_S = readin['aod_traj'][:]  # initial AOD
                    del [readin, index]
                    decay_factorf = decay_factor[1]


                valid_S = []
                for j in np.arange(len(aod_S)):
                    index1 = np.argwhere((lon_S[:, j] >= lon_lim[0]) & (lon_S[:, j] <= lon_lim[1]))
                    index2 = np.argwhere((lat_S[:, j] >= lat_lim[0]) & (lat_S[:, j] <= lat_lim[1]))
                    index = np.intersect1d(index1, index2)
                    if (len(index) != 0) & (len(np.argwhere(lev_S[index, j] > height_lim)) != 0):
                        valid_S = np.append(valid_S, j)
                    del [index1, index2, index]

                NN_S = len(valid_S)
                # print(NN_S)
                risk_S = np.zeros(xx_grid.shape)

                ## SNPP
                if NN_S != 0:
                    lon_S = lon_S[:, valid_S.astype(int)]
                    lat_S = lat_S[:, valid_S.astype(int)]
                    lev_S = lev_S[:, valid_S.astype(int)]
                    aod_S = aod_S[valid_S.astype(int)]
                    del valid_S

                    for j in np.arange(NN_S):
                        index1 = np.argwhere((lon_S[:, j] >= lon_lim[0]) & (lon_S[:, j] <= lon_lim[1]))
                        index2 = np.argwhere((lat_S[:, j] >= lat_lim[0]) & (lat_S[:, j] <= lat_lim[1]))
                        index3 = np.argwhere(lev_S[:, j] > height_lim)
                        index = reduce(np.intersect1d, (index1, index2, index3))

                        for hh in index:  # forecast hour, 3x3 grid?
                            x, y = grid_finder(xx_grid, yy_grid, lon_S[hh, j], lat_S[hh, j])
                            risk_S[x, y] = risk_S[x, y] + (aod_S[j] * decay_factorf * date_factor[i])

                            # risk_S[x-1:x+2, y-1:y+2] = risk_S[x-1:x+2, y-1:y+2]+(aod_S[j]*date_factor[i])
                            del [x, y]
                        del [hh, index1, index2, index3, index]

                if i == 0:
                    total_num_S = np.copy(NN_S)
                    total_risk_S = np.copy(risk_S) * member_norm
                else:
                    total_num_S = np.append(total_num_S, NN_S)
                    total_risk_S = total_risk_S + (risk_S * member_norm)
                del [risk_S,NN_S]


            for i in np.arange(len(grid_fileJ)):
                if today in grid_fileJ[i]:  # today

                    readin = Dataset(traj_fileJ[i])
                    index = np.squeeze(np.argwhere((readin['time'][:] > tt0) & (readin['time'][:] <= tt1)))
                    lon_J = readin['xtraj'][index, :]
                    lat_J = readin['ytraj'][index, :]
                    lev_J = readin['ptraj'][index, :]  # pressure level
                    time_J = readin['time'][index]  # forecast hour
                    aod_J = readin['aod_traj'][:]  # initial AOD
                    del [readin, index]
                    decay_factorf = decay_factor[0]

                else:  # day-1
                    readin = Dataset(traj_fileJ[i])
                    index = np.squeeze(np.argwhere((readin['time'][:] > tt0 + 24) & (readin['time'][:] <= tt1 + 24)))
                    lon_J = readin['xtraj'][index, :]
                    lat_J = readin['ytraj'][index, :]
                    lev_J = readin['ptraj'][index, :]  # pressure level
                    time_J = readin['time'][index]  # forecast hour
                    aod_J = readin['aod_traj'][:]  # initial AOD
                    del [readin, index]
                    decay_factorf = decay_factor[1]
                    # print(time_S, time_J)

                valid_J = []
                for j in np.arange(len(aod_J)):
                    index1 = np.argwhere((lon_J[:, j] >= lon_lim[0]) & (lon_J[:, j] <= lon_lim[1]))
                    index2 = np.argwhere((lat_J[:, j] >= lat_lim[0]) & (lat_J[:, j] <= lat_lim[1]))
                    index = np.intersect1d(index1, index2)
                    if (len(index) != 0) & (len(np.argwhere(lev_J[index, j] > height_lim)) != 0):
                        valid_J = np.append(valid_J, j)
                    del [index1, index2, index]

                # print('DAY ' + str(int(i * (-1))) + ' VALID TRAJ')
                # print('JPSS', len(valid_J)) #'SNPP', len(valid_S),
                # print('SNPP', len(valid_S))
                # print(valid_S, valid_J)

                ## == Risk Map ==
                NN_J = len(valid_J)
                # print(NN_J)
                risk_J = np.zeros(xx_grid.shape)

                ## JPSS
                if NN_J != 0:
                    lon_J = lon_J[:, valid_J.astype(int)]
                    lat_J = lat_J[:, valid_J.astype(int)]
                    lev_J = lev_J[:, valid_J.astype(int)]
                    aod_J = aod_J[valid_J.astype(int)]
                    del valid_J

                    for j in np.arange(NN_J):
                        index1 = np.argwhere((lon_J[:, j] >= lon_lim[0]) & (lon_J[:, j] <= lon_lim[1]))
                        index2 = np.argwhere((lat_J[:, j] >= lat_lim[0]) & (lat_J[:, j] <= lat_lim[1]))
                        index3 = np.argwhere(lev_J[:, j] > height_lim)
                        index = reduce(np.intersect1d, (index1, index2, index3))

                        for hh in index:  # forecast hour, 3x3 grid?
                            x, y = grid_finder(xx_grid, yy_grid, lon_J[hh, j], lat_J[hh, j])
                            risk_J[x, y] = risk_J[x, y] + (aod_J[j] * decay_factorf * date_factor[i])
                            # risk_J[x-1:x+2, y-1:y+2] = risk_J[x-1:x+2, y-1:y+2]+(aod_J[j]*date_factor[i])
                            del [x, y]
                        del [hh, index1, index2, index3, index]

                # print(np.nanmin(risk_S), np.nanmax(risk_S))
                # print(np.nanmin(risk_J), np.nanmax(risk_J))

                if i == 0:
                    total_num_J = np.copy(NN_J)
                    total_risk_J = np.copy(risk_J) * member_norm
                else:
                    total_num_J = np.append(total_num_J, NN_J)
                    total_risk_J = total_risk_J + (risk_J * member_norm)
                del [ NN_J,  risk_J]

            # print(total_num_S, total_num_J)
            # print(np.nanmin(total_risk_S), np.nanmax(total_risk_S))
            # print(np.nanmin(total_risk_J), np.nanmax(total_risk_J))

            # plt.figure()
            # plt.title('Low resol SNPP')
            # plt.pcolor(xx_grid, yy_grid, total_risk_S, cmap='YlOrRd')
            # plt.colorbar()
            # plt.show()
            # exit()


            '''ENSEMBLE RISK MAP'''
            #############################
            if len(grid_fileS)==0:
                total_num = total_num_J.astype(int)
                total_risk = total_risk_J
                total_risk[total_risk == 0] = np.nan
            elif len(grid_fileJ)==0:
                total_num = total_num_S.astype(int)
                total_risk = total_risk_S
                total_risk[total_risk == 0] = np.nan
            else:
                total_num = np.append(total_num_S, total_num_J).astype(int)
                total_risk = total_risk_S + total_risk_J
                total_risk[total_risk == 0] = np.nan

            # print(np.nanmin(total_risk), np.nanmax(total_risk))

            # plt.figure()
            # plt.title('Total risk')
            # plt.pcolor(xx_grid, yy_grid, total_risk_H, cmap='YlOrRd')
            # plt.colorbar()
            # plt.show()
            # exit()

            '''Writing NetCDF'''
            output = Dataset(path_output + '/VIIRSaerosol_RiskForecast_' + initial_time.replace(' ', '')[
                                                                                         :-2] + '_' + forecast_time.replace(' ',
                                                                                                                            '')[
                                                                                                      :-2] + '.nc', 'w')
            output.createDimension('lon', xx_grid.shape[0])
            output.createDimension('lat', xx_grid.shape[1])
            output.createDimension('time', 1)  # forecast time
            output.createDimension('member_num', int(1 / member_norm))
            var_xx = output.createVariable('xgrid', 'float', ('lon', 'lat'))
            var_yy = output.createVariable('ygrid', 'float', ('lon', 'lat'))
            var_risk = output.createVariable('risk', 'float', ('lon', 'lat'))
            var_time = output.createVariable('forecast_time', 'i4', ('time',))
            var_num = output.createVariable('valid_traj_num', 'i4', ('member_num',))
            var_num.long_name = 'Number of valid trajectories: Low-SNPP day 0~-1, Low-JPSS day 0~-1'
            var_xx[:] = xx_grid
            var_yy[:] = yy_grid
            var_risk[:] = total_risk
            var_time[:] = int(forecast_time.replace(' ', '')[:-2])
            var_num[:] = total_num
            output.close()
            print('== NetCDF DONE')

            '''Plotting'''
            fig, ax = plt.subplots(figsize=(12, 9))  # unit=100pixel
            h = ax.get_position()
            #ax.set_position([h.x0 - 0.04, h.y0 + 0.02, h.width + 0.06, h.height + 0.04])
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.5)

            plt.annotate('IDEA-NYS ' + str(tt0) + '-' + str(tt1) + 'h Risk Forecast', (h.x0 - 0.04, h.y0 + 0.02 + h.height + 0.03),
                         fontsize=22, fontweight='bold', xycoords='figure fraction')
            plt.annotate('Forecast member: ' + str(int(1 / member_norm)) + '; Valid trajectory: ' + str(np.sum(total_num)),
                         (h.x0 - 0.04, h.y0 + 0.02 + h.height), fontsize=18, xycoords='figure fraction')
            plt.annotate('Initial at ' + initial_time + ' UTC\nValid  at ' + forecast_times+'-'+forecast_time + ' UTC',
                         (h.x0 - 0.12 + h.width - 0.21, h.y0 + 0.02 + h.height), fontsize=16, xycoords='figure fraction')

            m = Basemap(llcrnrlon=lon_lim[0], urcrnrlon=lon_lim[-1], llcrnrlat=lat_lim[0], urcrnrlat=lat_lim[-1], projection='cyl',
                        resolution='l')
            # m.arcgisimage(service='World_Physical_Map', xpixels=2000, verbose=True)
            m.readshapefile(shp_USA, 'USA', color='k')
            m.readshapefile(shp_CAN, 'CAN', color='k')
            m.drawmeridians(np.arange(lon_lim[0], lon_lim[-1] + 1, 2), color='none', labels=[0, 0, 0, 1], fontsize=22)
            m.drawparallels(np.arange(round(lat_lim[0]), lat_lim[-1] + 1, 2), color='none', labels=[1, 0, 0, 0], fontsize=22)

            x, y = m(xx_grid, yy_grid)
            cs = m.pcolor(x, y, total_risk, cmap='YlOrRd', norm=norm, alpha=0.7)  # today

            cbaxes = fig.add_axes([h.x0, h.y0-0.07, h.width, 0.02])
            cb = plt.colorbar(cs, ticks=tick, extend='max', orientation='horizontal', cax=cbaxes)#
            #cb.set_label('Risk value', fontsize=22, fontweight='bold')
            cb.ax.tick_params(labelsize=22, length=12, width=2, direction='in')
            cb.outline.set_linewidth(2)

            # plt.show()
            # exit()

            plt.savefig(path_output + '/VIIRSaerosol_risk_map_' + initial_time.replace(' ', '')[
                                                                                :-2] + '_' + forecast_time.replace(' ', '')[
                                                                                             :-2] + '.png')
            plt.close()
            del [fig, ax, h, m, x, y, cs, cb]

            print('== END TIME', datetime.now())
            print('')
