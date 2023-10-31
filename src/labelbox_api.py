import os
from typing import List

import labelbox as lb
import supervisely as sly
from labelbox.data.serialization import COCOConverter
from labelbox.data.serialization.labelbox_v1.converter import LBV1Converter

import src.globals as g


def get_projects() -> List[lb.Project]:
    projects = g.STATE.client.get_projects()
    return projects


def download_coco_format_project(project: lb.Project) -> bool:
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
            sly.logger.error(f"Project {project.name} has no labels.")
            return False
        labels = LBV1Converter.deserialize(project_export)
        if not src_img_dir.endswith("/"):
            src_img_dir += "/"

        coco_labels = COCOConverter.serialize_instances(
            labels, image_root=src_img_dir, ignore_existing_data=True, max_workers=0
        )
        image_path = coco_labels.get("info").get("image_root")
        coco_labels["info"]["image_root"] = image_path.as_posix()
        sly.json.dump_json_file(coco_labels, os.path.join(src_ann_dir, "instances.json"))
        return project_src_dir
    except Exception as e:
        sly.logger.error(f"Can't download project {project.name} as COCO format.")
        sly.logger.error(e)
        return False
