"""
Header text
Read CP4/R25 data files and extract timeseries or other subsets
Doug Parker, December 2020
University of Leeds
"""
import numpy as np
import matplotlib
matplotlib.use('agg') # This is needed to suppress the need for a DISPLAY
import matplotlib.pyplot as plt
import cftime as ct
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from fileselector import file_selector

from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/

#model="CP4_regridded" # CP4, CP4_regridded or R25
model="CP4" # CP4, CP4_regridded or R25
#model="R25" # CP4, CP4_regridded or R25
fc="C" # C for current, F for future


def hh1970(yy,mm,dd,hh):
  return 24.*360.*(yy-1970)+30.*24.*(mm-1)+24.*(dd-1)+hh

def ymd1970(hh):
  import numpy as np
  hd=hh/24.
  yy=np.trunc(hd/360.)
  mm=np.trunc((hd-yy*360.)/30.)+1
  dd=np.trunc(hd-yy*360.-(mm-1)*30)+1
  yy=yy+1970
  return yy,mm,dd

# all data
#daystart,monthstart,yearstart=1,6,1997
#dayend,monthend,yearend=30,8,1997
#dayend,monthend,yearend=30,6,1997
daystart,monthstart,yearstart=8,6,1997
dayend,monthend,yearend=8,6,1997
#dayend,monthend,yearend=30,2,1997
# august 2006
#daystart,monthstart,yearstart=1,8,2006
#dayend,monthend,yearend=1,9,2006

lon1,lat1 = -20+360.,0.
lon2,lat2 = 20+360.,30.

hour1970=hh1970(yearstart,monthstart,daystart,0)
hourend1970=hh1970(yearend,monthend,dayend+1,0)


if model == 'R25':
  stashprec='a05216'
elif model == 'CP4_regridded':
  stashprec='a04203'
elif model == 'CP4':
  stashprec='a04203'

stash=stashprec
fieldname='precip'


'''
output rain amount loop
'''

#field_af_2007 = np.empty(shape = (24,740,987),dtype = float)
while hour1970 < hourend1970:
    year,month,day=ymd1970(hour1970)
    print('acquiring date ',year,month,day,hour1970)


    some_cftime_object = ct.Datetime360Day(year,month,day,0,) # hour = 0
    
    # get data
    nc_f=file_selector(some_cftime_object,stash,model,fc)
    print('date, file ',year,month,day,nc_f)
    nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                                   # and create an instance of the ncCDF4 class
    # Extract data from NetCDF file

    lat = nc_fid.variables['latitude'][:]  # extract/copy the data
    lon = nc_fid.variables['longitude'][:]
    time = nc_fid.variables['time'][:]
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
    
    ilat1=ilat1+int((lat1-lati1)/dlat+0.5) # check this especially pos/neg jumps， get lat and lon index of west af 
    ilon1=ilon1+int((lon1-loni1)/dlon+0.5)
    ilat2=ilat2+int((lat2-lati2)/dlat+0.5) # check this especially pos/neg jumps
    ilon2=ilon2+int((lon2-loni2)/dlon+0.5)
    
    field_af_1d = field[:,ilat1:ilat2,ilon1:ilon2] #the greater row/column index is, the greater lat/lon degree is.
    #print(ilat,lat[ilat],ilon,lon[ilon])
    
    #field_af_2007 = field_af_2007 + field_af_1d
    
    
    hour1970=int(time[-1]+1.)

clevs=[.1,5.,10.,20.]
xlims=[-20.,20.] # west af
ylims=[0.,30.] # west af
#xlims=[35.,57.] # madagascar
#ylims=[-25.,-5.] # 
rcolors=['white','blue','yellow','red']

hour1970=hh1970(yearstart,monthstart,daystart,0)


endhour = 21
for ig in range(0,endhour+3,3):
  gfile='./frames/'+str(int(hour1970+ig))+'_'+fc+'.png'
  hh=(int(hour1970)+ig)%24
  #hourstr=str(np.where(hh<10,'0'+str(hh),str(hh)))
  year1,month1,day1=ymd1970(hour1970+ig)
  hourstr=str(hh).zfill(2)
  daystr=str(int(day1)).zfill(2)
  monthstr=str(int(month1)).zfill(2)
  datestr=hourstr+'00 '+daystr+' '+monthstr+' '+str(int(year1))
  print (day,daystr,month,monthstr)
  print (datestr)
  #print gfile

  f=plt.figure(1)
  ax = plt.axes(projection=ccrs.PlateCarree())
  plt.axis(np.concatenate((xlims,ylims),axis=0))
  #plt.contourf(lon-360.,lat,3600.*field[ig,:,:],clevs,colors=rcolors,extend='max')
  plt.contourf(lon-360.,lat,3600*field[ig,:,:],clevs,colors=rcolors,extend='max')
  #plt.contourf(lon-360.,lat,3600.*field[ig,:,:],clevs,cmap="jet")
  ax.add_feature(cfeature.COASTLINE,lw=0.5, edgecolor='gray')
  ax.add_feature(cfeature.BORDERS,  lw=0.5, edgecolor='gray')
  plt.text(xlims[0]+1.,ylims[1]-1.,datestr)
  ax.gridlines()
  plt.colorbar()
  plt.savefig(gfile)

  plt.clf()



import imageio
filenames = ['237048_C.png','237051_C.png','237054_C.png','237057_C.png','237060_C.png',
             '237063_C.png','237066_C.png','237069_C.png']


with imageio.get_writer('mygif.gif', mode='I', duration= 1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)






