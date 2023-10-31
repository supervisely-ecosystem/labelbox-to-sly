import os
import shutil
from copy import deepcopy
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervisely as sly
from pycocotools.coco import COCO


def coco_to_supervisely(src_path: str, dst_path: str, ignore_bbox: bool = False) -> str:
    """Convert COCO project from src_path to Supervisely project in dst_path.

    :param src_path: path to COCO project.
    :type src_path: str
    :param dst_path: path to Supervisely project.
    :type dst_path: str
    :param ingore_bbox: if True, bounding boxes will be ignored, defaults to False
    :type ingore_bbox: bool, optional
    :return: path to Supervisely project.
    :rtype: str
    """
    project_meta = sly.ProjectMeta()

    dataset_name = os.path.basename(os.path.normpath(src_path))

    coco_ann_dir = os.path.join(src_path, "annotations")
    coco_ann_path = os.path.join(coco_ann_dir, "instances.json")
    if coco_ann_path is not None:
        try:
            coco_instances = COCO(annotation_file=coco_ann_path)
        except Exception as e:
            sly.logger.warning(f"File {coco_ann_path} has been skipped due to error: {e}")
            return None

        categories = coco_instances.loadCats(ids=coco_instances.getCatIds())
        coco_images = coco_instances.imgs
        coco_anns = coco_instances.imgToAnns

        # * Creating directories for Supervisely project.
        dst_dataset_path = os.path.join(dst_path, dataset_name)
        sly.fs.mkdir(dst_dataset_path)
        img_dir = os.path.join(dst_dataset_path, "img")
        ann_dir = os.path.join(dst_dataset_path, "ann")
        sly.fs.mkdir(img_dir)
        sly.fs.mkdir(ann_dir)

        project_meta = update_meta(project_meta, dst_path, categories, dataset_name)

        for img_id, img_info in coco_images.items():
            image_name = img_info["file_name"]
            if "/" in image_name:
                image_name = os.path.basename(image_name)
            if sly.fs.file_exists(os.path.join(src_path, "images", image_name)):
                img_ann = coco_anns[img_id]
                img_size = (img_info["height"], img_info["width"])
                ann = coco_to_sly_ann(
                    meta=project_meta,
                    coco_categories=categories,
                    coco_ann=img_ann,
                    image_size=img_size,
                    ignore_bbox=ignore_bbox,
                )
                move_trainvalds_to_sly_dataset(
                    dataset_dir=src_path,
                    coco_image=img_info,
                    ann=ann,
                    img_dir=img_dir,
                    ann_dir=ann_dir,
                )

    sly.logger.info(f"COCO dataset converted to Supervisely project: {dst_path}")
    return dst_path


def update_meta(
    meta: sly.ProjectMeta, dst_path: str, coco_categories: List[dict], ann_types=None
) -> sly.ProjectMeta:
    """Create Supervisely ProjectMeta from COCO categories.

    :param meta: ProjectMeta of Supervisely project.
    :type meta: sly.ProjectMeta
    :param dst_path: path to Supervisely project.
    :type dst_path: str
    :param coco_categories: List of COCO categories.
    :type coco_categories: List[dict]
    :param dataset_name: name of dataset.
    :type dataset_name: str
    :return: Updated ProjectMeta.
    :rtype: sly.ProjectMeta
    """
    path_to_meta = os.path.join(dst_path, "meta.json")
    if not os.path.exists(path_to_meta):
        colors = []
        for category in coco_categories:
            if category["name"] in [obj_class.name for obj_class in meta.obj_classes]:
                continue
            new_color = sly.color.generate_rgb(colors)
            colors.append(new_color)
            obj_class = sly.ObjClass(category["name"], sly.AnyGeometry, new_color)
            meta = meta.add_obj_class(obj_class)
        meta_json = meta.to_json()
        sly.json.dump_json_file(meta_json, path_to_meta)
    return meta


def coco_to_sly_ann(
    meta: sly.ProjectMeta,
    coco_categories: List[dict],
    coco_ann: List[Dict],
    image_size: Tuple[int, int],
    ignore_bbox: bool = False,
) -> sly.Annotation:
    """Convert COCO annotation to Supervisely annotation.

    :param meta: ProjectMeta of Supervisely project.
    :type meta: sly.ProjectMeta
    :param coco_categories: List of COCO categories.
    :type coco_categories: List[dict]
    :param coco_ann: List of COCO annotations.
    :type coco_ann: List[Dict]
    :param image_size: size of image.
    :type image_size: Tuple[int, int]
    :param ignore_bbox: if True, bounding boxes will be ignored, defaults to False
    :type ignore_bbox: bool, optional
    :return: Supervisely annotation.
    :rtype: sly.Annotation
    """

    labels = []
    imag_tags = []
    name_cat_id_map = coco_category_to_class_name(coco_categories)
    for object in coco_ann:
        curr_labels = []
        segm = object.get("segmentation")
        bbox = object.get("bbox")

        if segm is not None and len(segm) > 0:
            obj_class_name = name_cat_id_map[object["category_id"]]
            obj_class = meta.get_obj_class(obj_class_name)
            if type(segm) is dict:
                polygons = convert_rle_mask_to_polygon(object)
                for polygon in polygons:
                    figure = polygon
                    label = sly.Label(figure, obj_class)
                    labels.append(label)
            elif type(segm) is list and not is_segm_equal_to_bbox(segm[0], bbox):
                figures = convert_polygon_vertices(object, image_size)
                curr_labels.extend([sly.Label(figure, obj_class) for figure in figures])
            elif type(segm) is list and is_segm_equal_to_bbox(segm[0], bbox):
                x, y, w, h = bbox
                rectangle = sly.Label(sly.Rectangle(y, x, y + h, x + w), obj_class)
                curr_labels.append(rectangle)
        labels.extend(curr_labels)

        if not ignore_bbox:
            if bbox is not None and len(bbox) == 4:
                obj_class_name = name_cat_id_map[object["category_id"]]
                obj_class = meta.get_obj_class(obj_class_name)
                if len(curr_labels) > 1:
                    for label in curr_labels:
                        bbox = label.geometry.to_bbox()
                        labels.append(sly.Label(bbox, obj_class))
                else:
                    x, y, w, h = bbox
                    rectangle = sly.Label(sly.Rectangle(y, x, y + h, x + w), obj_class)
                    labels.append(rectangle)

        caption = object.get("caption")
        if caption is not None:
            imag_tags.append(sly.Tag(meta.get_tag_meta("caption"), caption))

    return sly.Annotation(image_size, labels=labels, img_tags=imag_tags)


def is_segm_equal_to_bbox(polygon: list, bbox: list) -> bool:
    """Check if polygon is equal to bounding box.

    :param polygon: List of polygon vertices.
    :type polygon: list
    :param bbox: List of bounding box vertices.
    :type bbox: list
    :return: True if polygon is equal to bounding box, False otherwise.
    :rtype: bool
    """
    if len(polygon) != 8 or len(bbox) != 4:
        return False
    same_left = polygon[0] == bbox[0]
    same_top = polygon[1] == bbox[1]
    same_right = polygon[2] == bbox[0] + bbox[2]
    same_bottom = polygon[7] == bbox[1] + bbox[3]
    return same_left and same_top and same_right and same_bottom


def convert_rle_mask_to_polygon(coco_ann: List[Dict]) -> List[sly.Polygon]:
    """Convert RLE mask to List of Supervisely Polygons.

    :param coco_ann: List of COCO annotations.
    :type coco_ann: List[Dict]
    :return: List of Supervisely Polygons.
    :rtype: List[sly.Polygon]
    """
    if type(coco_ann["segmentation"]["counts"]) is str:
        coco_ann["segmentation"]["counts"] = bytes(
            coco_ann["segmentation"]["counts"], encoding="utf-8"
        )
        mask = mask_util.decode(coco_ann["segmentation"])
    else:
        rle_obj = mask_util.frPyObjects(
            coco_ann["segmentation"],
            coco_ann["segmentation"]["size"][0],
            coco_ann["segmentation"]["size"][1],
        )
        mask = mask_util.decode(rle_obj)
    mask = np.array(mask, dtype=bool)
    return sly.Bitmap(mask).to_contours()


def convert_polygon_vertices(
    coco_ann: List[Dict], image_size: Tuple[int, int]
) -> List[sly.Polygon]:
    """Convert polygon vertices to Supervisely Polygons.

    :param coco_ann: List of COCO annotations.
    :type coco_ann: List[Dict]
    :param image_size: size of image.
    :type image_size: Tuple[int, int]
    :return: List of Supervisely Polygons.
    :rtype: List[sly.Polygon]
    """
    polygons = coco_ann["segmentation"]
    if all(type(coord) is float for coord in polygons):
        polygons = [polygons]

    exteriors = []
    for polygon in polygons:
        polygon = [polygon[i * 2 : (i + 1) * 2] for i in range((len(polygon) + 2 - 1) // 2)]
        exteriors.append([(width, height) for width, height in polygon])

    interiors = {idx: [] for idx in range(len(exteriors))}
    id2del = []
    for idx, exterior in enumerate(exteriors):
        temp_img = np.zeros(image_size + (3,), dtype=np.uint8)
        geom = sly.Polygon([sly.PointLocation(y, x) for x, y in exterior])
        geom.draw_contour(temp_img, color=[255, 255, 255])
        im = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        for idy, exterior2 in enumerate(exteriors):
            if idx == idy or idy in id2del:
                continue
            results = [cv2.pointPolygonTest(contours[0], (x, y), False) > 0 for x, y in exterior2]

            if all(results):
                interiors[idx].append(deepcopy(exteriors[idy]))
                id2del.append(idy)

    for j in sorted(id2del, reverse=True):
        del exteriors[j]

    figures = []
    for exterior, interior in zip(exteriors, interiors.values()):
        exterior = [sly.PointLocation(y, x) for x, y in exterior]
        interior = [[sly.PointLocation(y, x) for x, y in points] for points in interior]
        figures.append(sly.Polygon(exterior, interior))

    return figures


def move_trainvalds_to_sly_dataset(
    dataset_dir: str, coco_image: Dict, ann: sly.Annotation, img_dir: str, ann_dir: str
) -> None:
    """Move images and annotations to Supervisely dataset.

    :param dataset_dir: path to COCO dataset.
    :type dataset_dir: str
    :param coco_image: COCO image.
    :type coco_image: Dict
    :param ann: Supervisely annotation.
    :type ann: sly.Annotation
    :param img_dir: path to Supervisely images.
    :type img_dir: str
    :param ann_dir: path to Supervisely annotations.
    :type ann_dir: str
    """
    image_name = coco_image["file_name"]
    if "/" in image_name:
        image_name = os.path.basename(image_name)
    ann_json = ann.to_json()
    coco_img_path = os.path.join(dataset_dir, "images", image_name)
    sly_img_path = os.path.join(img_dir, image_name)
    if sly.fs.file_exists(os.path.join(coco_img_path)):
        sly.json.dump_json_file(ann_json, os.path.join(ann_dir, f"{image_name}.json"))
        shutil.copy(coco_img_path, sly_img_path)


def coco_category_to_class_name(coco_categories: List[dict]) -> Dict:
    """Create dictionary with COCO category id as key and category name as value.

    :param coco_categories: List of COCO categories.
    :type coco_categories: List[dict]
    :return: Dictionary with COCO category id as key and category name as value.
    :rtype: Dict
    """
    return {category["id"]: category["name"] for category in coco_categories}
