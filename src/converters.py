import os
import shutil
from copy import deepcopy
from typing import Dict, List, Tuple

import cv2
import labelbox as lb
import numpy as np
import pycocotools.mask as mask_util
import supervisely as sly
from labelbox.data.serialization.labelbox_v1.converter import LBV1VideoIterator
from pycocotools.coco import COCO

import src.globals as g
from src.labelbox_api import download_mask


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

        project_meta = update_sly_meta(project_meta, dst_path, categories)

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


def update_sly_meta(
    meta: sly.ProjectMeta, dst_path: str, coco_categories: List[dict]
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


def convert_object_to_sly_geometry(lb_obj: dict, obj_cls: sly.ObjClass = None):
    """Convert Labelbox object to Supervisely geometry.

    :param lb_obj: Labelbox object.
    :type lb_obj: dict
    :param obj_cls: Supervisely object class.
    :type obj_cls: sly.ObjClass
    :return: Supervisely geometry.
    :rtype: sly.Geometry
    """

    if obj_cls is None:
        return None
    elif obj_cls.geometry_type == sly.Rectangle and lb_obj.get("bbox"):
        bbox = lb_obj["bbox"]
        top, left = int(bbox["top"]), int(bbox["left"])
        bottom, right = (int(top) + bbox["height"], int(left) + bbox["width"])
        geometry = sly.Rectangle(top, left, bottom, right)
    elif obj_cls.geometry_type == sly.Bitmap and lb_obj.get("instanceURI"):
        mask_url = lb_obj["instanceURI"]
        local_path = os.path.join(g.TEMP_DIR, "mask.png")
        mask_np = download_mask(mask_url, local_path, g.STATE.client)
        geometry = sly.Bitmap(mask_np)
    elif obj_cls.geometry_type == sly.Polygon and lb_obj.get("polygon"):
        exterior = [(point["y"], point["x"]) for point in lb_obj["polygon"]]
        geometry = sly.Polygon(exterior=exterior)
    elif obj_cls.geometry_type == sly.Polyline and lb_obj.get("line"):
        exterior = [(point["y"], point["x"]) for point in lb_obj["line"]]
        geometry = sly.Polyline(exterior=exterior)
    elif obj_cls.geometry_type == sly.Point and lb_obj.get("point"):
        geometry = sly.Point(lb_obj["point"]["y"], lb_obj["point"]["x"])
    else:
        return None
    return geometry


def process_video_project(project: lb.Project):
    """Process video project. Get video links and annotations from Labelbox.
    Convert annotations to Supervisely format and upload to Supervisely project.

    :param project: Labelbox project.
    :type project: lb.Project
    :return: Supervisely project if success, False otherwise.
    :rtype: Union[sly.Project, bool]
    """

    try:
        sly_project = g.api.project.create(
            g.STATE.selected_workspace,
            project.name,
            type=sly.ProjectType.VIDEOS,
            change_name_if_conflict=True,
        )
        sly.logger.debug(f"Created project '{project.name}' in Supervisely.")

        sly_ds = g.api.dataset.create(sly_project.id, "ds0")
        sly.logger.debug(f"Created dataset 'ds0' in project '{project.name}'.")

        sly_meta = create_sly_meta_from_lb(project, sly_project)

        project = g.STATE.client.get_project(project.uid)
        project_export = project.export_labels(download=True)
        if len(project_export) == 0:
            sly.logger.error(f"Project {project.name} has no labels.")
            return False
        sly.logger.debug(f"Project '{project.name}' has image labels.")
        data = LBV1VideoIterator(project_export, g.STATE.client)
        for video_data in data:
            video_url = video_data["Labeled Data"]
            video_name = video_data["External ID"]
            sly_video = g.api.video.upload_link(
                sly_ds.id, video_url, video_name, skip_download=True
            )
            video_objects_map = {}
            video_frames = []
            labels_info = video_data["Label"]
            for idx, frame_info in enumerate(labels_info):
                figures = []
                for lb_obj in frame_info["objects"]:
                    cls_name = lb_obj["title"]
                    vobj_id = lb_obj["featureId"]

                    lbl_obj_cls = sly_meta.get_obj_class(cls_name)
                    geometry = convert_object_to_sly_geometry(lb_obj, lbl_obj_cls)
                    if geometry is None:
                        continue
                    vobj = video_objects_map.get(vobj_id)
                    if vobj is None:
                        vobj = sly.VideoObject(lbl_obj_cls, class_id=lbl_obj_cls.sly_id)
                        video_objects_map[vobj_id] = vobj

                    figure = sly.VideoFigure(vobj, geometry, idx, class_id=vobj_id)

                    figures.append(figure)
                sly_frame = sly.Frame(idx, figures=figures)
                video_frames.append(sly_frame)
            video_objects = sly.VideoObjectCollection(list(video_objects_map.values()))
            video_frames = sly.FrameCollection(video_frames)
            img_size = sly_video.frame_height, sly_video.frame_width
            video_ann = sly.VideoAnnotation(img_size, len(labels_info), video_objects, video_frames)
            g.api.video.annotation.append(sly_video.id, video_ann)
        sly.logger.info(f"Project {project.name} was successfully uploaded to Supervisely.")
        return sly_project
    except Exception as e:
        sly.logger.error(f"Can't process the project {project.name}.")
        sly.logger.error(e)
        g.api.project.remove(sly_project.id)
        return False


def create_sly_meta_from_lb(project: lb.Project, sly_project: sly.Project):
    """Create and update Supervisely ProjectMeta from Labelbox project.

    :param project: Labelbox project.
    :type project: lb.Project
    :param sly_project: Supervisely project.
    :type sly_project: sly.Project
    :return: Supervisely ProjectMeta.
    :rtype: sly.ProjectMeta
    """

    ontology = project.ontology()
    tools = ontology.tools()
    obj_classes = []
    sly.logger.debug(f"Updating meta for project '{project.name}'.")
    for tool in tools:
        obj_class = sly.ObjClass(tool.name, g.GEOMETRIES_MAPPING[tool.tool.name])
        obj_classes.append(obj_class)

        sly.logger.debug(f"   - Added object class '{tool.name}' ({obj_class.geometry_type.name}).")

    project_meta = sly.ProjectMeta(obj_classes=obj_classes)
    g.api.project.update_meta(sly_project.id, project_meta)
    return project_meta
