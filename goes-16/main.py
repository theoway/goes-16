import os, shutil
import argparse
from goes_16_latest import GoesDownloaderLatest
from goes_16_date import GoesDownloaderDate, GoesDownloaderIndividualBboxDate
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename="goes_downloader.log", 
    filemode="w"
)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Type of Download")
    parser.add_argument("-s", "--save", required=True)

    latest_parser = subparsers.add_parser("latest")

    datetime_parser = subparsers.add_parser("date")
    datetime_parser.add_argument("-d", "--date", nargs=2, type=str, required=False)
    datetime_parser.add_argument("-g", "--geojson", action='store_false', required=False)
    args = parser.parse_args()
    try:
        if 'geojson' in args:
            logging.info(f"Bulk Downloading based on bbox geojson start & end dates")

            down = GoesDownloaderIndividualBboxDate(args.save)
            #down.wildfire_map()
            down.run("ABI-L2-ACHAC", "cloud", "HT")
            #down.run("ABI-L2-FDCC", "mask", "Mask")

        elif 'date' in args:
            logging.info(f"Bulk Downloading")
            down = GoesDownloaderDate(args.save,
                                      datetime.strptime(args.date[0], '%Y-%m-%d'),
                                      datetime.strptime(args.date[1], '%Y-%m-%d'))
            down.wildfire_map()
            down.run("ABI-L2-ACHAC", "cloud", "HT")
            down.run("ABI-L2-FDCC", "mask", "Mask")
            #down.run("ABI-L2-FDCC", "area", "Area")
            #down.run("ABI-L2-FDCC", "power", "Power")
            #down.run("ABI-L2-FDCC", "temp", "Temp")
        
        elif 'date' not in args:
            logging.info(f"Downloading data {datetime.now()}")
            down = GoesDownloaderLatest(args.save)
            down.wildfire_map()
            down.run("ABI-L2-ACHAC", "cloud", "HT")
            down.run("ABI-L2-FDCC", "mask", "Mask")
            #down.run("ABI-L2-FDCC", "area", "Area")
            #down.run("ABI-L2-FDCC", "power", "Power")
            #down.run("ABI-L2-FDCC", "temp", "Temp")
            down.cloud_json()

        logging.info("Finished process")
    except Exception as e:
        logging.error(e, exc_info=True)
        raise
    finally:
        down.clean_root_dir()

if __name__ == "__main__":
    main()