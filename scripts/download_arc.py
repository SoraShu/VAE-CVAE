# download files from https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip
import os
import urllib.request
import zipfile
import shutil
import uuid
def download_arc_dataset(destination_folder='data'):
    os.makedirs(destination_folder, exist_ok=True)
    url = 'https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.zip'
    tmp_dir = os.path.join(destination_folder, f'tmp_{uuid.uuid4().hex}')
    os.makedirs(tmp_dir, exist_ok=True)
    zip_path = os.path.join(tmp_dir, 'ARC-AGI-2-main.zip')
    print(f'Downloading ARC-AGI-2 dataset from {url}...')
    urllib.request.urlretrieve(url, zip_path)
    print(f'Extracting to {tmp_dir}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
    os.remove(zip_path)
    data_dst_path = os.path.join(destination_folder, 'arc2')
    data_src_path = os.path.join(tmp_dir,'ARC-AGI-2-main', 'data')
    if os.path.exists(data_dst_path):
        shutil.rmtree(data_dst_path)
    shutil.move(data_src_path, data_dst_path)
    shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    download_arc_dataset()