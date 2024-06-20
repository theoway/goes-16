from download import Downloader
from node import Node, parse_json
from datetime import datetime, timedelta
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", required=True)
    parser.add_argument("-e", "--end", required=True)
    parser.add_argument("-d", "--diff", required=True)

    args = parser.parse_args()

    START_TIME  = datetime.strptime(args.start, "%d/%m/%Y-%H:%M:%S")
    END_TIME = datetime.strptime(args.end, "%d/%m/%Y-%H:%M:%S")

    dates = [START_TIME + timedelta(minutes=x*float(args.diff)) for x in range(int((END_TIME - START_TIME).total_seconds()/(60 * float(args.diff)) + 1))]

    down = Downloader()
    for idx, i in enumerate(dates):
        down.download_datetime(i)

    nodes = parse_json("./deployment/locations.json")
        
    for i, node in enumerate(nodes):
        node.crop()