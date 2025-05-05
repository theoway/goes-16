import json
import os
import random
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pprint import pprint
from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import s3fs
from netCDF4 import Dataset
from osgeo import gdal, ogr, osr
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from tqdm import tqdm


class ViirsPoint:
    def __init__(
        self, lat: float, lon: float, brightness: float, frp: float, date: str
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.brightness = brightness
        self.frp = frp
        self.date: datetime = datetime.strptime(date, "%Y/%m/%d %H%M")
        self.key: None | int = None

    @classmethod
    def parse_feature(cls, feature: str):
        properties = json.loads(feature)["properties"]
        date = f"{properties['ACQ_DATE']} {properties['ACQ_TIME']}"
        lat = properties["LATITUDE"]
        lon = properties["LONGITUDE"]
        # bri = properties["BRIGHTNESS"]
        frp = properties["FRP"]

        return cls(lat, lon, 0, frp, date)


class ViirsDataset:
    def __init__(
        self, dir_location: str, eps: float = 0.01, min_samples: int = 4
    ) -> None:
        self.fs = s3fs.S3FileSystem(anon=True)

        file = ogr.Open(dir_location)
        shape = file.GetLayer(0)

        self.eps = eps
        self.min_samples = min_samples

        self.data_points: List[ViirsPoint] = []

        for idx in range(shape.GetFeatureCount()):
            feature = shape.GetFeature(idx)
            point = ViirsPoint.parse_feature(feature.ExportToJson())

            self.data_points.append(point)

        self.unique_dates = set([p.date for p in self.data_points])

    def fit(self, date: datetime):
        self.filtered_data_points = list(
            filter(lambda x: x.date == date, self.data_points)
        )
        tmp_data = [[x.lat, x.lon] for x in self.filtered_data_points]
        self._db = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, algorithm="auto"
        ).fit(tmp_data)

    def parse_filename(self, filename: str) -> dict:
        if filename.startswith("OR_"):
            filename = filename[3:]

        parts = filename.split("_")
        if len(parts) != 5:
            raise ValueError("Invalid filename format")

        channel = parts[0][-2:]
        satellite_id = parts[1]
        start_time = parts[2][1:]

        start_dt = datetime.strptime(start_time, "%Y%j%H%M%S%f")

        parameters = {
            "channel": int(channel),
            "satellite_id": satellite_id,
            "start_time": start_dt,
        }
        return parameters

    def nearest_min(self, time: datetime, dates: Iterable[datetime]):
        diffs = list(map(lambda x: (x - time), dates))
        sorted_diffs = sorted(diffs, key=lambda x: abs(x.total_seconds()))
        return sorted_diffs

    def process_file(self, file):
        file_path = self.__process_band_file(file)
        self.__convert_to_tiff(file_path)

    def __get_band_file(self, dir_path: str, band: int):
        raster_band = None
        for file in os.listdir(dir_path):
            if "output" in file:
                continue
            f = self.parse_filename(file)
            if f["channel"] == band:
                raster_band = file
        if raster_band is None:
            raise ValueError(f"Could not find band: {band}")
        return raster_band

    def __convert_WSG__(self, lat, lon):
        InSR = osr.SpatialReference()
        InSR.SetFromUserInput("EPSG:4326")
        OutSR = osr.SpatialReference()
        OutSR.SetFromUserInput("ESRI:102498")

        transform_epsg = osr.CoordinateTransformation(InSR, OutSR)
        lon, lat, _ = transform_epsg.TransformPoint(lat, lon)
        return lon, lat

    def __patchify_file(self, file, rand_x: int, rand_y: int, save_dir, win_size=32):
        for idx, poly in enumerate(self.__polygons):
            lon, lat = poly.Centroid().GetX(), poly.Centroid().GetY()

            layer = gdal.Open(file)
            transform = layer.GetGeoTransform()
            xOrigin = transform[0]
            yOrigin = transform[3]
            pixelWidth = transform[1]
            pixelHeight = -transform[5]

            col = (int((lon - xOrigin) / pixelWidth)) - win_size / 2 - 1
            row = (int((yOrigin - lat) / pixelHeight)) - win_size / 2 - 1

            __save_dir = os.path.join(save_dir, str(idx))
            if not os.path.exists(__save_dir):
                os.mkdir(__save_dir)

            window = (col + rand_x, row + rand_y, win_size, win_size)
            save_file = os.path.join(
                __save_dir, f"{file.split('/')[-1].split('.')[0]}_{idx}.tiff"
            )
            gdal.Translate(save_file, file, srcWin=window)

    def patch(self, date: datetime, win_size=32):
        if len(self.__polygons) == 0:
            return

        patches_dir = os.path.join(self.base_dir, "patches")
        if not os.path.exists(patches_dir):
            os.mkdir(patches_dir)

        save_dir = os.path.join(patches_dir, str(date))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        x_random = int(random.uniform(0, win_size // 10))
        # x_random = 5
        y_random = int(random.uniform(-1 * (win_size // 3), win_size // 3))

        for file in os.listdir(self.download_dir):
            self.__patchify_file(
                os.path.join(self.download_dir, file),
                x_random,
                y_random,
                save_dir,
                win_size,
            )

    def __clean_up(self, dir: str):
        if not os.path.exists(dir):
            return

        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

    def process_output(self, dir_path: str):
        raster_band_500m = self.__get_band_file(dir_path, 2)
        raster_layer = gdal.Open(os.path.join(dir_path, raster_band_500m))
        cols = raster_layer.RasterXSize
        rows = raster_layer.RasterYSize
        projection = raster_layer.GetProjection()
        geotransform = raster_layer.GetGeoTransform()

        target_layer = gdal.GetDriverByName("MEM").Create(
            "", cols, rows, 1, gdal.GDT_Byte
        )
        target_layer.SetProjection(projection)
        target_layer.SetGeoTransform(geotransform)

        mem_driver = ogr.GetDriverByName("Memory")
        mem_ds = mem_driver.CreateDataSource("mem_data_source")
        InSR = osr.SpatialReference()
        InSR.SetFromUserInput("ESRI:102498")
        mem_layer = mem_ds.CreateLayer(
            "multipolygon", geom_type=ogr.wkbMultiPolygon, srs=InSR
        )
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

        ds = defaultdict(list)
        for idx, p in enumerate(self._db.labels_):
            ds[p].append(idx)

        self.__polygons = []
        for k, v in ds.items():
            if k == -1:
                continue

            polygon_points = [
                [self.filtered_data_points[x].lon, self.filtered_data_points[x].lat]
                for x in v
            ]

            if len(polygon_points) > 0:
                hull = ConvexHull(polygon_points)
                ring = ogr.Geometry(ogr.wkbLinearRing)

                for p in hull.vertices:
                    lon = polygon_points[p][0]
                    lat = polygon_points[p][1]
                    lon, lat = self.__convert_WSG__(lat, lon)
                    ring.AddPoint(lon, lat)

                # Complete ring
                if len(hull.vertices) > 0:
                    lon = polygon_points[hull.vertices[0]][0]
                    lat = polygon_points[hull.vertices[0]][1]
                    lon, lat = self.__convert_WSG__(lat, lon)
                    ring.AddPoint(lon, lat)

                    poly = ogr.Geometry(ogr.wkbPolygon)

                    poly.AddGeometry(ring)
                    self.__polygons.append(poly)
                    multipolygon.AddGeometry(poly)

        feature_defn = mem_layer.GetLayerDefn()
        feature = ogr.Feature(feature_defn)
        feature.SetGeometry(multipolygon)
        mem_layer.CreateFeature(feature)
        gdal.RasterizeLayer(
            target_layer,
            [1],
            mem_layer,
            burn_values=[255],
            options=["ALL_TOUCHED=TRUE"],
        )
        gdal.Translate(f"{dir_path}/output.tiff", target_layer, format="GTiff")

    def process_dir(self, dir_path: str):
        with Pool() as pp:
            files = os.listdir(dir_path)
            files_paths = [os.path.join(dir_path, file) for file in files]
            with tqdm(total=len(files_paths), leave=False) as pbar:
                for _ in pp.imap_unordered(self.process_file, files_paths):
                    pbar.update()

    def __convert_to_tiff(self, file_path):
        layer = gdal.Open(file_path)
        options = gdal.WarpOptions(
            format="GTiff",
            width=10000,
            height=6000,
        )
        file_name = file_path.replace(".tiff", ".tif")
        gdal.Warp(file_name, layer, options=options)
        os.remove(file_path)

        return file_name

    def __process_band_file(self, file_path: str, band="Rad") -> str:
        file = self.parse_filename(file_path.split("/")[-1])

        if file["channel"] <= 6:
            raster_layer = gdal.Open("NETCDF:{0}:{1}".format(file_path, band))
            ds = Dataset(file_path)

            kappa = ds.variables["kappa0"][:]
            Field = kappa * ds.variables["Rad"][:]

            os.remove(file_path)
            file_path = file_path.replace(".nc", ".tiff")
            driver = gdal.GetDriverByName("Gtiff")
            output_dataset = driver.Create(
                file_path,
                raster_layer.RasterXSize,
                raster_layer.RasterYSize,
                1,
                gdal.GDT_Float32,
            )

            # Copy geotransform and projection
            output_dataset.SetGeoTransform(raster_layer.GetGeoTransform())
            output_dataset.SetProjection(raster_layer.GetProjection())

            # Write the calculated data to the new GeoTIFF file
            output_dataset.GetRasterBand(1).WriteArray(Field)
            output_dataset.FlushCache()

            raster_layer = None
            output_dataset = None

            return file_path

        else:
            raster_layer = gdal.Open("NETCDF:{0}:{1}".format(file_path, band))
            ds = Dataset(file_path)

            planck_fk1 = ds.variables["planck_fk1"][:]
            planck_fk2 = ds.variables["planck_fk2"][:]
            planck_bc1 = ds.variables["planck_bc1"][:]
            planck_bc2 = ds.variables["planck_bc2"][:]
            Field = (
                planck_fk2 / (np.log((planck_fk1 / ds.variables["Rad"][:]) + 1))
                - planck_bc1
            ) / planck_bc2

            os.remove(file_path)
            file_path = file_path.replace(".nc", ".tiff")
            driver = gdal.GetDriverByName("Gtiff")
            output_dataset = driver.Create(
                file_path,
                raster_layer.RasterXSize,
                raster_layer.RasterYSize,
                1,
                gdal.GDT_Float32,
            )

            # Copy geotransform and projection
            output_dataset.SetGeoTransform(raster_layer.GetGeoTransform())
            output_dataset.SetProjection(raster_layer.GetProjection())

            # Write the calculated data to the new GeoTIFF file
            output_dataset.GetRasterBand(1).WriteArray(Field)
            output_dataset.FlushCache()

            raster_layer = None
            output_dataset = None

            return file_path

    def download(
        self,
        base_dir: str,
        date_range: int = 5,
        param: str = "ABI-L1b-RadC",
        process=True,
    ):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.download_dir = os.path.join(self.base_dir, "downloads")
        if not os.path.exists(self.download_dir):
            os.mkdir(self.download_dir)

        for date in self.unique_dates:
            self.fit(date)
            self.download_datetime(date, self.download_dir, date_range, param, process)
            self.process_output(self.download_dir)
            self.patch(date, win_size=128)

    def download_datetime(
        self,
        date: datetime,
        save_dir: str,
        date_range: int = 5,
        param: str = "ABI-L1b-RadC",
        process=True,
    ):
        days_since_year_start = (
            datetime(date.year, date.month, date.day) - datetime(date.year, 1, 1)
        ).days + 1

        try:
            data_hour = self.fs.ls(
                f"s3://noaa-goes16/{param}/{date.year}/{str(days_since_year_start).zfill(3)}/{str(date.hour).zfill(2)}"
            )

            dates = set(
                map(
                    lambda x: self.parse_filename(x.split("/")[-1])["start_time"],
                    data_hour,
                )
            )

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            nearest_time = self.nearest_min(date, dates)

            self.__clean_up(self.download_dir)
            for diff in nearest_time[:date_range]:
                closest_date = date + diff
                files = list(
                    filter(
                        lambda x: f"s{closest_date.strftime('%Y%j%H%M%S')}" in x,
                        data_hour,
                    )
                )

                assert len(files) == 16
                self.fs.get(files, save_dir)

            if process:
                self.process_dir(save_dir)

        except Exception as e:
            print(f"Unable to query aws for {str(date).zfill(3)}: {param}")
            raise ValueError(f"Unable to load aws due to {e}")

    def plot(self, file_name: str):
        ds = defaultdict(list)
        for idx, p in enumerate(self._db.labels_):
            ds[p].append(idx)

        df = pd.DataFrame({"lon": [], "lat": [], "key": []})

        for k, v in ds.items():
            if k == -1:
                continue

            x = [self.filtered_data_points[x].lon for x in v]
            y = [self.filtered_data_points[x].lat for x in v]
            assert len(x) == len(y)
            k = [k for _ in range(len(x))]

            df2 = pd.DataFrame({"lon": x, "lat": y, "key": k})
            df = pd.concat([df, df2], ignore_index=True)

        fig = px.scatter(df, x="lon", y="lat", color="key")

        maps_dir = os.path.join(self.base_dir, "maps")
        if not os.path.exists(maps_dir):
            os.mkdir(maps_dir)

        fig.write_html(f"data/maps/{file_name}.html")


if __name__ == "__main__":
    shapefile = "./files/viirs_data/J1_VIIRS_C2_USA_contiguous_and_Hawaii_24h.shp"
    dataset = ViirsDataset(shapefile)
    dataset.download("data", date_range=2)
