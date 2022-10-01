import gzip
import requests

import numpy as np

from pathlib import Path


def download_mnist(folder):
    """ Downloads mnist data from: yann.lecun.com """
    
    folder.mkdir(exist_ok=True)
    urls= ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']

    for url in urls:
        r = requests.get(url)
        
        if (r.status_code == 200):
            open(Path(folder) / Path(str(url)).name, 'wb').write(r.content)
        else:
            raise Exception("Could not download MNIST dataset")


def read_mnist(folder):
    """ Reads the folder of the MNIST dataset and returns array of mnist data"""
    data_paths = list(Path(folder).glob('*.gz'))
    data = []
    for path in data_paths:
        with open(path, "rb") as f:
            temp = f.read()
            temp = np.frombuffer(gzip.decompress(temp), dtype=np.uint8).copy()
            data.append(temp)

    return data