import os
import urllib.request
from typing import List

import labelbox as lb
import numpy as np
import supervisely as sly
from PIL import Image

import src.globals as g
from src.exceptions import handle_lb_exceptions
import urllib.request

@handle_lb_exceptions
def get_projects() -> List[lb.Project]:
    projects = g.STATE.client.get_projects()
    return projects


@handle_lb_exceptions
def download_mask(url: str, save_path: str, client):
    """Download mask from url and return as numpy array

    :param url: url to mask
    :type url: str
    :param save_path: path to save mask
    :type save_path: str
    :param client: labelbox client
    :type client: labelbox.client.Client
    :return: mask as numpy array
    :rtype: np.ndarray
    """

    req = urllib.request.Request(url, headers=client.headers)
    mask_np = np.asarray(Image.open(urllib.request.urlopen(req)), dtype=np.uint8)
    return mask_np
