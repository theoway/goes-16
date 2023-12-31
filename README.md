# GOES-16

## Downloading Latest

```
python3 goes-16/main.py -s /tmp/DATA/ latest
```

Will Download the latest bounding boxes in the save directory



## Bulk Downloading By datetime

```
python3 goes-16/main.py -s /tmp/DATA/ date -d 29-9-2023 1-10-2023
```
Will Download the specified date range bounding boxes in the save directory

# Docker setup
## Building
```
docker build -t [image-name]:[image-tag] .
```
Builds the docker image

## Running
```
docker run  --rm  -v "/repo/path/goes-16:/app" goes_downloader:stable python3 goes-16/main.py -s DATA/ latest
```
Downloads latest images and saves them in `DATA` folder

```
docker run --rm  -v "/path/to/repo/goes-16:/app" [image-name]:[image-tag] python3 goes-16/main.py -s DATA/ date -d 2023-09-01 2023-10-02
```
Downloads images between start and end date, in above example 29-9-2023 & 1-10-2023, and saves them in `DATA` folder


```
docker run --network goes-16_default --rm  -v "/path/to/repo/goes-16:/app" goes_downloader:stable python3 goes-16/mosaic_update.py
```
Downloads images between start and end date, in above example 29-9-2023 & 1-10-2023, and saves them in `DATA` folder