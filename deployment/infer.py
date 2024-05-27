import os
import torch
import random
import numpy as np
from typing import List
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from osgeo import gdal, ogr, osr


def parse_filename(filename: str) -> dict:
    if filename.startswith("OR_"):
        filename = filename[3:]

    parts = filename.split("_")
    if len(parts) != 5:
        raise ValueError("Invalid filename format")

    channel = None

    band_data = parts[0].split("-")
    product = band_data[2]
    if product == "RadC":
        channel = int(parts[0][-2:])

    start_time = parts[2][1:]
    end_time = parts[3][1:]

    start_dt = datetime.strptime(start_time, "%Y%j%H%M%S%f")
    end_dt = datetime.strptime(end_time, "%Y%j%H%M%S%f")

    return {
        "channel": channel,
        "product": product,
        "start_time": start_dt,
        "end_time": end_dt,
    }


class ModelInput:
    def __init__(self, in_dir) -> None:
        self.in_dir = in_dir

    def create_input(self):
        for band in os.listdir(self.in_dir):
            if ".tif" not in band:
                continue

            f = parse_filename(band)

            if f["channel"] == 1:
                band_1 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_1 = (band_1 - 0.0) / (1 - 0)

            elif f["channel"] == 2:
                raster_layer = gdal.Open(os.path.join(self.in_dir, band))

                driver = gdal.GetDriverByName("Gtiff")
                output_dataset = driver.Create(
                    os.path.join(self.in_dir, f"{f['start_time']}.tiff"),
                    raster_layer.RasterXSize,
                    raster_layer.RasterYSize,
                    1,
                    gdal.GDT_Byte,
                )

                # Copy geotransform and projection
                output_dataset.SetGeoTransform(raster_layer.GetGeoTransform())
                output_dataset.SetProjection(raster_layer.GetProjection())

                drv = ogr.GetDriverByName("ESRI Shapefile")
                dst_ds = drv.CreateDataSource(os.path.join(self.in_dir, f"{f['start_time']}.shp"))

                band_2 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_2 = (band_2 - 0.0) / (1 - 0)

            elif f["channel"] == 3:
                band_3 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_3 = (band_3 - 0.0) / (1 - 0)

            elif f["channel"] == 4:
                band_4 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_4 = (band_4 - 0.0) / (1 - 0)

            elif f["channel"] == 5:
                band_5 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_5 = (band_5 - 0.0) / (1 - 0)

            elif f["channel"] == 6:
                band_6 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=0,
                    max=1,
                )
                band_6 = (band_6 - 0.0) / (1 - 0)

            elif f["channel"] == 7:
                mn, mm = 197.31, 411.86
                band_7 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_7 = (band_7 - mn) / (mm - mn)

            elif f["channel"] == 8:
                mn, mm = 138.05, 311.06
                band_8 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_8 = (band_8 - mn) / (mm - mn)

            elif f["channel"] == 9:
                mn, mm = 137.7, 311.08
                band_9 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_9 = (band_9 - mn) / (mm - mn)

            elif f["channel"] == 10:
                mn, mm = 126.91, 331.2
                band_10 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_10 = (band_10 - mn) / (mm - mn)

            elif f["channel"] == 11:
                mn, mm = 127.69, 341.3
                band_11 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_11 = (band_11 - mn) / (mm - mn)

            elif f["channel"] == 12:
                mn, mm = 117.49, 311.06
                band_12 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_12 = (band_12 - mn) / (mm - mn)

            elif f["channel"] == 13:
                mn, mm = 89.62, 341.28
                band_13 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_13 = (band_13 - mn) / (mm - mn)

            elif f["channel"] == 14:
                mn, mm = 96.19, 341.28
                band_14 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_14 = (band_14 - mn) / (mm - mn)

            elif f["channel"] == 15:
                mn, mm = 97.38, 341.28
                band_15 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_15 = (band_15 - mn) / (mm - mn)

            elif f["channel"] == 16:
                mn, mm = 92.7, 318.26
                band_16 = torch.clamp(
                    v2.ToImage()(Image.open(os.path.join(self.in_dir, band))),
                    min=mn,
                    max=mm,
                )
                band_16 = (band_16 - mn) / (mm - mn)

        inputs = torch.concatenate(
            (
                band_1,
                band_2,
                band_3,
                band_4,
                band_5,
                band_6,
                band_7,
                band_8,
                band_9,
                band_10,
                band_11,
                band_12,
                band_13,
                band_14,
                band_15,
                band_16,
            )
        )

        return inputs, [output_dataset, dst_ds]


class CustomDataset(Dataset):
    def __init__(
        self, datalist: List[ModelInput], transforms=None, target_transforms=None
    ) -> None:
        super().__init__()
        self.datalist = datalist
        self.transform = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        input_images, label = self.datalist[index].create_input()

        if self.transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            input_images = self.transform(input_images)

        if self.target_transforms is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            label = self.target_transforms(label)

        return input_images, label


if __name__ == "__main__":
    MODEL = torch.load("./training/models/training/models/R2U/model19_0.024963259883224963.pth", map_location=torch.device('cpu'))
    MODEL.eval()
    threshold = 0.25

    fires = []
    patch_dir = "./tmp/patches/"
    for fire in os.listdir(patch_dir):
        fires.append(ModelInput(os.path.join(patch_dir, fire)))

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False),
        ]
    )

    dataset = CustomDataset(fires, transforms=transform, target_transforms=transform)

    for i in tqdm(dataset):
        inputs, save = i

        with torch.no_grad():
            img = torch.unsqueeze(inputs, 0)
            img = transform(img)
            out = torch.sigmoid(MODEL(img))
            out[out < threshold] = 0.0
            out[out >= threshold] = 1.0
            out = torch.squeeze(out, 0)
            out = to_pil_image(out)
            im_label = np.array(out)

            gtiff, shp = save

            gtiff.GetRasterBand(1).WriteArray(im_label)
            gtiff.FlushCache()

            data = gtiff.GetRasterBand(1)

            srs = osr.SpatialReference()
            srs.ImportFromWkt(gtiff.GetProjection())

            dst_layer = shp.CreateLayer("output", srs=srs)
            gdal.Polygonize(data, None, dst_layer, -1, [], callback=None)
            shp.Destroy()