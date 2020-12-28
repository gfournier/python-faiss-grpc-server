import os
import shutil
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import smart_open

from ann_grpc.indexes.annoy_index import AnnoyIndexWrapper
from ann_grpc.indexes.faiss_index import FaissIndexWrapper


class IndexLoader:

    # TODO add time to refresh file when updated

    def __init__(self):
        pass

    def load_index(self, url, **kwargs):
        # Sync file
        filename = self.sync_index(url)
        print(f"Index synced to {filename}")
        # Load index
        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        return self.load_from_ext(filename, ext, **kwargs)

    @staticmethod
    def load_from_ext(filename, ext, **kwargs):
        if ext == '.faiss':
            return FaissIndexWrapper.load_index(filename, **kwargs)
        elif ext == '.ann':
            return AnnoyIndexWrapper.load_index(filename, **kwargs)

    @staticmethod
    def sync_index(url):
        print(f"Sync index from {url}")
        with NamedTemporaryFile(delete=False) as fout:
            with smart_open.open(url, 'rb') as fin:
                shutil.copyfileobj(fin, fout)
            return fout.name
