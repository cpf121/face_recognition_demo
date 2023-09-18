import stow
import tarfile
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO


def download_and_unzip(url, extract_to='Datasets', chunk_size=1024 * 1024):
    http_response = urlopen(url)

    data = b''
    iterations = http_response.length
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    tar_file = tarfile.open(fileobj=BytesIO(data), mode='r|bz2')
    tar_file.extractall(path=extract_to)
    tar_file.close()


dataset_path = stow.join('Datasets', 'LJSpeech-1.1')
if not stow.exists(dataset_path):
    download_and_unzip('https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2', extract_to='Datasets')
