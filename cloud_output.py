import numpy as np
from numpy import mean
from numpy import loadtxt
import math
import matplotlib.pyplot as plt
import cftime as ct
from fileselector import file_selector
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
from datetime import datetime
import time
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature

import imageio

#import matplotlib.cm as cm
#import matplotlib.animation as animation


'''
some defination
'''
def hh1970(yy,mm,dd,hh):
  return 24.*360.*(yy-1970)+30.*24.*(mm-1)+24.*(dd-1)+hh

def ymd1970(hh):
  import numpy as np
  hd=hh/24.
  yy=np.trunc(hd/360.)
  mm=np.trunc((hd-yy*360.)/30.)+1
  dd=np.trunc(hd-yy*360.-(mm-1)*30)+1
  yy=yy+1970
  return yy,mm,dd#,hd

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


#model="CP4_regridded" # CP4, CP4_regridded or R25
model="CP4" # CP4, CP4_regridded or R25
#model="R25" # CP4, CP4_regridded or R25
fc="C" # C for current, F for future

#date range
daystart,monthstart,yearstart=1,6,2006
dayend,monthend,yearend=1,6,2006

d1 = str(yearstart)+'-'+str(monthstart)+'-'+str(daystart)
d2 = str(yearend)+'-'+str(monthend)+'-'+str(dayend)
days =  days_between(d1,d2)

#lon1,lat1
lon1,lat1 = -20+360.,0.
lon2,lat2 = 20+360.,30.

hour1970=hh1970(yearstart,monthstart,daystart,0)
hourend1970=hh1970(yearend,monthend,dayend+1,0)


stash='a01208' #short wave
#stash='a03332' #long wave
#stash='f30208' #to slow, 2 mins per day
#stash='a04203' #rain amount 



'''
output rain amount loop
'''

#field_af_1998 = np.empty(shape = (24,740,987),dtype = float)
start_time = time.time()  
field_af = np.empty(shape = (24,740,987),dtype = float)
while hour1970 < hourend1970:
    year,month,day=ymd1970(hour1970)
    print('acquiring date ',year,month,day,hour1970)


    some_cftime_object = ct.Datetime360Day(year,month,day,0,) # hour = 3 for f30208
    
    # get data
    nc_f=file_selector(some_cftime_object,stash,model,fc)
    print('date, file ',year,month,day,nc_f)
    nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                                   # and create an instance of♦ the ncCDF4 class
    # Extract data from NetCDF file

    lat = nc_fid.variables['latitude'][:]  # extract/copy the data
    lon = nc_fid.variables['longitude'][:]
    #plev = nc_fid.variables['pressure'][:]
    time_cloud = nc_fid.variables['time'][:]
    field = nc_fid.variables[stash][:] # rain is in kg m-2 s-1
                
                        # shape is time, lat, lon as shown above
    
    dlat=np.float(lat[3]-lat[2])
    dlon=np.float(lon[3]-lon[2])
    ilat1,ilon1=0,0
    ilat2,ilon2=0,0
    
    lati1=lat[ilat1] 
    loni1=lon[ilon1]  
    lati2=lat[ilat2]
    loni2=lon[ilon2]
    
    ilat1=ilat1+int((lat1-lati1)/dlat+0.5) # 1124.6
    ilon1=ilon1+int((lon1-loni1)/dlon+0.5) #111.6
    ilat2=ilat2+int((lat2-lati2)/dlat+0.5) # 1865.5
    ilon2=ilon2+int((lon2-loni2)/dlon+0.5) #1099.4
    
    #pha = np.where(plev == 850.)[0][0]
    
    field_af_1d = field[:,ilat1:ilat2,ilon1:ilon2] #the greater row/column index is, the greater lat/lon degree is.
    #print(ilat,lat[ilat],ilon,lon[ilon])
    
    field_af = field_af + field_af_1d

    
    hour1970=int(time_cloud[-1]+1.)
end_time = time.time()
print("total time taken this loop: ", end_time - start_time) #f30208 12.5s, a01208 2s, a04203 18s 


    
'''
save out rain amount 3D array     
'''
arr_reshaped = field_af.reshape(field_af.shape[0], -1)
# saving reshaped array to file.
#fout = open(r'C:/users/mm21zh/1997_1.txt','w')
np.savetxt('2006.txt', arr_reshaped, fmt="%.6f") #round to 6 digit

field_af_2007 = field_af

# retrieving data from file.
loaded_arr = np.loadtxt("C:/Users/mm21zh/OneDrive - University of Leeds/uol_vm/a03332longwave_data/2005.txt")  
# reshape
load_1997 = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])

load_1998 = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])

load_1999 = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])

load_2000 = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])

load_2001 = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])

load_2002 = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])

load_2003 = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])

load_2004 = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])

load_2005 = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])





field_af = load_1997+load_1998+load_1999+load_2000+load_2001+load_2002+load_2003+load_2004+load_2005+field_af_2007 #2007 actual is 2006

arr_reshaped = field_af.reshape(field_af.shape[0], -1)
# saving reshaped array to file.
np.savetxt('combine_year9706.txt', arr_reshaped, fmt="%.6f") #round to 6 digit


loaded_arr = np.loadtxt("C:/Users/mm21zh/OneDrive - University of Leeds/uol_vm/a01208shortwave_data/2006.txt")  
sw = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2]) #3am to 8pm there are data

loaded_arr = np.loadtxt("C:/Users/mm21zh/OneDrive - University of Leeds/uol_vm/a03332longwave_data/combine_year9706.txt")  
lw = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])


loaded_arr = np.loadtxt("C:/Users/mm21zh/OneDrive - University of Leeds/uol_vm/a04203rain_data/combine_year9706.txt")  
rain = loaded_arr.reshape(
    loaded_arr.shape[0], loaded_arr.shape[1] // field_af_2007.shape[2], field_af_2007.shape[2])

'''
calculate correlation between t0 and t15
'''



num_year = 10
days = 121
h0 = rain[0,:,:]/(days*num_year) 


h0_ilat1=int((10-lat[0])/dlat+0.5) # lat1
h0_ilat2=int((15-lat[0])/dlat+0.5) # lat2
h0_ilon1=int((-10+360.-lon[0])/dlon+0.5) # lon1+360
h0_ilon2=int((15+360.-lon[0])/dlon+0.5) #lon2+360
    
#select 5-15N,-10-15E from h0 array through using: (rain area index - west africa index), 
#both are in terms of whole field), the difference is rain area index in h0
h0_xr1,h0_xr2 = h0_ilat1-ilat1,h0_ilat2-ilat1 
h0_yr1,h0_yr2 = h0_ilon1-ilon1,h0_ilon2-ilon1


#import time

#make sure sub domain is within the whole west africa domain range
L1_x = 200 #lat_south
L2_x = 150 #lat_north
L1_y = 250 #lon_west
L2_y = 50 #lon_east

L1_x = L1_x if h0_xr1-L1_x > 0 else h0_xr1
L2_x = L2_x if h0_xr2+L2_x < len(rain[1]) else len(rain[1])-h0_xr2
L1_y = L1_y if h0_yr1-L1_y > 0 else h0_yr1
L2_y = L2_y if h0_yr2+L2_y < len(rain[2]) else len(rain[1][1])-h0_yr2

endhour = 21
for ig in range(0,endhour+3,3):
    corr_map = np.zeros(shape = (L1_x+L2_x,L1_y+L2_y),dtype = float)
    counter2 = 0
    for xr in range(h0_xr1,h0_xr2):#(0,ilat2-ilat1-40): 
        for yr in range(h0_yr1,h0_yr2):
            hr = h0[xr,yr]*3600
            if hr >= 1: 
                cl_ra = lw[ig,xr-L1_x:xr+L2_x,yr-L1_y:yr+L2_y]/(days*num_year)
                cl_ra = hr*cl_ra # mm/hour* W/m^2
                #rain = rain + h0[xr,yr]*h15[xr+i,yr+j]
                corr_map = corr_map + cl_ra
                corr_map[np.isnan(corr_map)] = 0
                counter2 += 1
    print("working on: h"+str(ig))
    corr_map = corr_map/counter2
    corr_map = np.flipud(corr_map) #reverse lat upside down
    plt.close()
    fig = plt.figure()
    plt.imshow(corr_map, cmap='jet')
    datestr = 'Correlation between hr00 and hr'+str(ig)
    plt.text(5.,10.,datestr,color='white')
    plt.clim(220,340) #max of field_af, for lw 5-15 it is (230,320), lw 8-22/10-15 is (220,340)
    plt.colorbar()
    plt.savefig('RAmount h'+str(ig)+'.png')





'''
create gif for plots
'''
filenames = ['RAmount h0.png','RAmount h3.png','RAmount h6.png','RAmount h9.png','RAmount h12.png',
             'RAmount h15.png','RAmount h18.png','RAmount h21.png']

filenames_sw = ['RAmount h6.png','RAmount h9.png','RAmount h12.png',
             'RAmount h15.png','RAmount h18.png']

with imageio.get_writer('mygif.gif', mode='I', duration= 1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)





#make 8 charts in one chart 
plt.close()
plt.figure(figsize=(25, 20))
plt.subplots_adjust(hspace=0.0001)

count=1
for filename in filenames:
    # add a new subplot iteratively
    image = imageio.imread(filename)
    ax = plt.subplot(3, 3, count)
    count += 1
    plt.xticks([])
    plt.yticks([])
    # filter df and plot ticker on the new subplot axis
    plt.imshow(image, 'gray')


plt.savefig('all 8 pics')