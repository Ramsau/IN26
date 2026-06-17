import settings
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

def read_imu():
    imu_data = open(settings.imu_file, 'r').readlines()
    return imu_data

def read_gnss():
    raw = open(settings.gnss_file, 'r', encoding='iso-8859-1').readlines()
    gnss_data = [[float(val) for val in entry.split()] for entry in raw[1:]]
    gnss_data = np.array(gnss_data)
    return gnss_data

def plot_gnss(gnss_data):
    cimgt_request = cimgt.OSM()
    ax = plt.axes(projection=cimgt_request.crs)
    ax.set_extent(settings.lat_lon_plotting_bounds)
    ax.add_image(cimgt_request, 5)
    plt.scatter(gnss_data[:, 1], gnss_data[:, 2], transform=ccrs.PlateCarree())
    plt.show()



gnss_data = read_gnss()
plot_gnss(gnss_data)

