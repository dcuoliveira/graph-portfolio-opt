import dropbox
from pathlib import Path
import os
from tqdm import tqdm
import time
import requests
import argparse

parser = argparse.ArgumentParser()

def download_folder(dbx, folder, local_path, max_retries=3, delay=5):
    """
    dbx: dropbox.Dropbox object instance
    folder: Path to the Dropbox folder
    local_path: Local path where the files will be downloaded
    max_retries: Maximum number of retries if a download fails
    delay: Time to wait between retries
    """
    # List all files in the Dropbox folder
    result = dbx.files_list_folder(folder)

    # Iterate over all files in the Dropbox folder
    for entry in result.entries:
        # Construct the full local path
        local_file = Path(local_path) / entry.name

        # Make sure the local file's directory exists
        local_file.parent.mkdir(parents=True, exist_ok=True)

        # If the entry is a file, download it
        if isinstance(entry, dropbox.files.FileMetadata):
            # Open the local file in write-binary mode
            with local_file.open("wb") as f:
                retries = 0
                while retries < max_retries:
                    try:
                        # Download the Dropbox file to the local file
                        metadata, res = dbx.files_download(str(Path(folder) / entry.name))
                        f.write(res.content)
                        # If the download is successful, break out of the loop
                        break
                    except requests.exceptions.ReadTimeout:
                        # If a ReadTimeout occurs, wait for the delay and retry
                        print("ReadTimeout occurred, waiting for {} seconds before retrying...".format(delay))
                        time.sleep(delay)
                        retries += 1
        # If the entry is a folder, recurse
        elif isinstance(entry, dropbox.files.FolderMetadata):
            download_folder(dbx, str(Path(folder) / entry.name), str(local_file), max_retries, delay)

def main():

    parser.add_argument('-tk', '--token', type=str, help='access token to be generated', default=None)
    parser.add_argument('-sy', '--start_year', type=int, help='start year of the crsp data', default=2000)
    parser.add_argument('-ey', '--end_year', type=int, help='end year of the crsp data', default=2021 + 1)

    args = parser.parse_args()

    # initialize a Dropbox object instance
    dbx = dropbox.Dropbox(args.token)

    years = [str(y) for y in range(args.start_year, args.end_year)]

    # check if exists
    local_path = os.path.join(os.path.dirname(__file__), "inputs", "US_CRSP_NYSE")
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    for year in tqdm(years, total=len(years), desc="Downloading CRSP Data"):

        # specify the Dropbox folder and the local path
        folder = "/US_CRSP_NYSE/Yearly/{}".format(year)
        year_local_path = os.path.join(local_path, year)

        # check if exists
        if not os.path.exists(year_local_path):
            os.makedirs(year_local_path)

        # call the download_folder function
        download_folder(dbx, folder, year_local_path)

if __name__ == "__main__":
    main()