import os
import urllib.request
from typing import List

import labelbox as lb
import numpy as np
import supervisely as sly
from labelbox.data.serialization import COCOConverter
from labelbox.data.serialization.labelbox_v1.converter import LBV1Converter
from PIL import Image

import src.globals as g


def get_projects() -> List[lb.Project]:
    projects = g.STATE.client.get_projects()
    return projects


def download_coco_format_project(project: lb.Project) -> bool:
    """Download project from labelbox and convert to COCO format

    :param project: labelbox project
    :type project: lb.Project
    :return: path to project directory or False if error
    :rtype: Union[str, bool]
    """

    project_src_dir = os.path.join(g.COCO_DIR, project.name)
    src_img_dir = os.path.join(project_src_dir, "images")
    src_ann_dir = os.path.join(project_src_dir, "annotations")

    sly.fs.mkdir(project_src_dir, remove_content_if_exists=True)
    sly.fs.mkdir(src_img_dir, remove_content_if_exists=True)
    sly.fs.mkdir(src_ann_dir, remove_content_if_exists=True)

    try:
        project = g.STATE.client.get_project(project.uid)
        project_export = project.export_labels(download=True)
        if len(project_export) == 0:
            sly.logger.warning(f"Project {project.name} has no labels.")
            return False
        sly.logger.debug(f"Project '{project.name}' has image labels.")
        labels = LBV1Converter.deserialize(project_export)
        coco_labels = COCOConverter.serialize_instances(
            labels, image_root=src_img_dir, ignore_existing_data=True, max_workers=0
        )
        image_path = coco_labels.get("info").get("image_root")
        coco_labels["info"]["image_root"] = image_path.as_posix()
        sly.json.dump_json_file(coco_labels, os.path.join(src_ann_dir, "instances.json"))
        sly.logger.info(
            f"Project {project.name} was downloaded successfully and converted to COCO format."
        )
        return project_src_dir
    except Exception as e:
        sly.logger.error(f"Can't process the project {project.name}: {e}")
        return False


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
