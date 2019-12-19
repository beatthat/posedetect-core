import re
import os
from detectron.utils.io import download_url


def clear_cache_for(url, cache_dir=None):
    cache_file_path = cache_url(url, cache_dir, download_if_missing=False)

    if os.path.exists(cache_file_path):
        print("will remove {}".format(cache_file_path))
        did_remove = True
        os.remove(cache_file_path)
    else:
        print("not exists remove {}".format(cache_file_path))
        did_remove = False

    return did_remove, cache_file_path


def cache_url(url, cache_dir=None, download_if_missing=True):
    """Get the local/cached file path for a url, typically pre-trained weights.
    """
    if cache_dir is None:
        cache_dir = "/cache"

    cache_file_path = re.sub("^http[s]?://", "", url)
    cache_file_path = os.path.join(cache_dir, re.sub("/", "-", cache_file_path))

    if os.path.exists(cache_file_path) or not download_if_missing:
        return cache_file_path

    cache_file_dir = os.path.dirname(cache_file_path)
    if not os.path.exists(cache_file_dir):
        os.makedirs(cache_file_dir)

    download_url(url, cache_file_path)
    return cache_file_path
