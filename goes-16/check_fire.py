import os
from random import randint
import datetime
import csv

import numpy as np
from osgeo import gdal

data_dir = '/app/geoserver/datadir/mosaic_dir'
regions = ['happy_camp', 'bigwood', 'carson', 'oakland']
band = 'wld_map'
csv_header = ['region', 'date', 'confidence']

def __tif_validity(f_path, band):
    ds = gdal.Open(f_path)
    raster_array = np.array(ds.GetRasterBand(1).ReadAsArray())

    file_name = os.path.basename(f_path)
    granuel_prefix = band.replace('_', '') + '_'
    ingestion_timestamp = file_name.replace(granuel_prefix, '').replace('_', '').replace('T', '').replace('Z', '').replace('.tif', '')
    ingestion = datetime.datetime.strptime(ingestion_timestamp, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')

    fire_found_raster = raster_array[raster_array > 0]
    if fire_found_raster.size:
        return ingestion, np.mean(fire_found_raster)
    return None

csv_file = f"{__file__}.csv".replace('.py', '')
with open(csv_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)

    for region in regions:
        region_dir = os.path.join(data_dir, region + '_' + band)
        tif_files = [os.path.join(region_dir, file) for file in os.listdir(region_dir) if file.find('.tif') > -1]
        fire_data = []
        for tif_file in tif_files:
            data = __tif_validity(tif_file, band)
            if data is not None:
                fire_data.append([region, data[0], data[1]])
        fire_data = sorted(fire_data, key = lambda x: str(x[1]))
        writer.writerows(fire_data)