import json
import os
import shutil
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import smart_open

from ann_grpc.indexes.annoy_index import AnnoyIndexWrapper
from ann_grpc.indexes.faiss_index import FaissIndexWrapper


def load_from_ext(filename, ext, **kwargs):
    if ext == 'faiss':
        return FaissIndexWrapper.load_index(filename, **kwargs)
    elif ext == 'ann':
        return AnnoyIndexWrapper.load_index(filename, **kwargs)
    elif ext == 'json':
        return load_composite_index(filename, **kwargs)


def load_composite_index(filename, **kwargs):
    with open(filename, "rt") as f:
        jso = json.load(f)
    assert isinstance(jso, list)
    indexes = {}
    for elem in jso:
        assert isinstance(elem, dict)
        assert "name" in elem
        assert "type" in elem
        assert "url" in elem
        index_kwargs = kwargs.copy()
        if "params" in elem:
            index_kwargs.update(elem["params"])
        indexes[elem["name"]] = load_from_ext(
            sync_file(elem["url"]), elem["type"], **index_kwargs
        )
    return indexes


def sync_file(url):
    print(f"Sync index from {url}")
    with NamedTemporaryFile(delete=False) as fout:
        with smart_open.open(url, 'rb') as fin:
            shutil.copyfileobj(fin, fout)
        print(f"Index synced to {fout.name}")
        return fout.name


class IndexLoader:

    # TODO add timer to refresh file when updated

    def __init__(self):
        pass

    def load_index(self, url, **kwargs):
        # Sync file
        filename = sync_file(url)
        # Load index
        path = urlparse(url).path
        ext = os.path.splitext(path)[1][1:]
        return load_from_ext(filename, ext, **kwargs)
