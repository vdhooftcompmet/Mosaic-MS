import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import requests
import sys
import shutil

from tqdm import tqdm
import zipfile
from urllib.parse import urlparse


def ask_for_download(url, filename):
    response = input(
            f"ATTENTION: file {filename} is missing.\n" 
            f"It can be downloaded from '{url}'. \n"
            f"Do you want to download it now? [Y/n]\n"
            f"> " 
        )
    if response.strip() in ["", "Y", "y"]:
        return True
    return False


def download_from_link(url, target_folder, extract):
    files_produced = []

    parsed_url = urlparse(url)
    raw_filename = os.path.basename(parsed_url.path)
    if not raw_filename:
        raw_filename = "downloaded_file"
        
    os.makedirs(target_folder, exist_ok=True)
    local_filename = os.path.join(target_folder, raw_filename)
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 

    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(local_filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if not extract:
        return [local_filename]
    
    if not zipfile.is_zipfile(local_filename):
        print(f"WARNING: could not extract file {local_filename}!")
        return [local_filename]


    print("Extracting files...")
    temp_extract_dir = os.path.join(target_folder, "temp_extraction")
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
    

    for filename in os.listdir(temp_extract_dir):
        source = os.path.join(temp_extract_dir, filename)
        destination = os.path.join(target_folder, filename)
        
        if os.path.exists(destination):
            if os.path.isdir(destination):
                shutil.rmtree(destination)
            else:
                os.remove(destination)
        
        shutil.move(source, target_folder)
        files_produced.append(destination)
        
    shutil.rmtree(temp_extract_dir)
    os.remove(local_filename)

    print(f"Done! Processed {len(files_produced)} file(s).")
    return files_produced
