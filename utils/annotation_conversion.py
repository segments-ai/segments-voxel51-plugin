import fiftyone as fo
import numpy as np

from .helpers import Point3D


def create_51_cuboid(instance: dict, category_name: str):
    position = Point3D(**instance["position"]).array().tolist()
    dims = Point3D(**instance["dimensions"]).array().tolist()

    detection = fo.Detection(
        label=category_name,
        location=position,
        dimensions=dims,
        rotation=[0, 0, instance["yaw"]],
    )

    return detection


def create_51_bbox(
    instance: dict, image_size: np.ndarray, category_name: str
) -> fo.Detection:
    points = np.array(instance["points"])
    points = points / image_size[None, :]
    width = points[1, 0] - points[0, 0]
    height = points[1, 1] - points[0, 1]
    detection = fo.Detection(
        bounding_box=[points[0, 0], points[0, 1], width, height], label=category_name
    )

    return detection


def create_51_polyline(
    instance: dict,
    image_size: np.ndarray,
    category_name: str,
    is_polygon: bool,
) -> fo.Polyline:
    points = np.asarray(instance["points"])
    points = points / image_size[None, :]

    polygon = fo.Polyline(
        label=category_name,
        points=[points.tolist()],
        closed=is_polygon,
        filled=is_polygon,
    )
    return polygon


def create_51_keypoint(
    instance: dict, image_size: np.ndarray, category_name: str
) -> fo.Keypoint:
    points = np.asarray(instance["points"])
    points = points / image_size[None, :]

    point = fo.Keypoint(points=points.tolist(), label=category_name)
    return point


def create_51_3dpolygon(
    instance: dict, category_name: str, is_polygon: bool
) -> fo.Polyline:
    points = np.array(instance["points"])
    line = fo.Polyline(
        label=category_name, points3d=[points.tolist()], closed=is_polygon
    )
    return line
