"""
Operators for integrating with segments.ai
"""

import enum
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import numpy as np
import requests
import segments
import segments.typing
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from segments import SegmentsClient, SegmentsDataset

SEGMENTS_FRONTEND_URL = "https://app.segments.ai"
SEGMENTS_METADATA_KEY = "segments_metadata"


class TargetSelection(enum.Enum):
    DATASET = "full_dataset"
    SELECTED = "selected"
    CURRENT_VIEW = "current_view"


class SegmentsDatasetType(enum.Enum):
    SEGMENTATION_BITMAP = "segmentation-bitmap"
    SEGMENTATION_BITMAP_HIGHRES = "segmentation-bitmap-highres"
    BBOXES = "bboxes"
    VECTOR = "vector"
    KEYPOINTS = "keypoints"
    # IMAGE_SEGMENTATION_SEQUENCE = "image-segmentation-sequence"
    # IMAGE_VECTOR_SEQUENCE = "image-vector-sequence"
    POINTCLOUD_CUBOID = "pointcloud-cuboid"
    POINTCLOUD_SEGMENTATION = "pointcloud-segmentation"
    POINTCLOUD_VECTOR = "pointcloud-vector"
    POINTCLOUD_CUBOID_SEQUENCE = "pointcloud-cuboid-sequence"
    POINTCLOUD_SEGMENTATION_SEQUENCE = "pointcloud-segmentation-sequence"
    POINTCLOUD_VECTOR_SEQUENCE = "pointcloud-vector-sequence"
    MULTISENSOR_SEQUENCE = "multisensor-sequence"

class UploadLabelType(str, enum.Enum):
    DETECTIONS = "detections"
    DETECTION = "detection"
    POLYLINES = "polylines"
    POLYLINE = "polyline"
    DETECTIONS_3D = "3d-detections"
    DETECTION_3D = "3d-detection"
    POLYLINES_3D = "3d-polylines"
    POLYLINE_3D = "3d-polyline"


POINTCLOUD_TASKS = {
    SegmentsDatasetType.POINTCLOUD_CUBOID,
    SegmentsDatasetType.POINTCLOUD_SEGMENTATION,
    SegmentsDatasetType.POINTCLOUD_VECTOR,
    SegmentsDatasetType.POINTCLOUD_CUBOID_SEQUENCE,
    SegmentsDatasetType.POINTCLOUD_SEGMENTATION_SEQUENCE,
    SegmentsDatasetType.POINTCLOUD_VECTOR_SEQUENCE,
}

IMAGE_TASKS = {
    SegmentsDatasetType.SEGMENTATION_BITMAP,
    SegmentsDatasetType.SEGMENTATION_BITMAP_HIGHRES,
    SegmentsDatasetType.BBOXES,
    SegmentsDatasetType.VECTOR,
    SegmentsDatasetType.KEYPOINTS,
}

SEQUENCE_TASKS = {
    SegmentsDatasetType.POINTCLOUD_CUBOID_SEQUENCE,
    SegmentsDatasetType.POINTCLOUD_SEGMENTATION_SEQUENCE,
    SegmentsDatasetType.POINTCLOUD_VECTOR_SEQUENCE,
    SegmentsDatasetType.MULTISENSOR_SEQUENCE,
}

POINTCLOUD_SEQUENCE_TASKS = {
    task for task in POINTCLOUD_TASKS if task in SEQUENCE_TASKS
}

SINGLE_POINTCLOUD_TASKS = {
    task for task in POINTCLOUD_TASKS if task not in SEQUENCE_TASKS
}

SINGLE_IMAGE_TASKS = {task for task in IMAGE_TASKS if task not in SEQUENCE_TASKS}


# NOTE: this is a partial duplication. Do we need this set for the segments.typing types?
IMAGE_TASKS_SEGMENTS = {
    segments.typing.TaskType.SEGMENTATION_BITMAP,
    segments.typing.TaskType.SEGMENTATION_BITMAP_HIGHRES,
    segments.typing.TaskType.IMAGE_SEGMENTATION_SEQUENCE,
    segments.typing.TaskType.BBOXES,
    segments.typing.TaskType.VECTOR,
    segments.typing.TaskType.IMAGE_VECTOR_SEQUENCE,
    segments.typing.TaskType.KEYPOINTS,
}

POINTCLOUD_TASKS_SEGMENTS = {
    segments.typing.TaskType.POINTCLOUD_CUBOID,
    segments.typing.TaskType.POINTCLOUD_SEGMENTATION,
    segments.typing.TaskType.POINTCLOUD_VECTOR,
    segments.typing.TaskType.POINTCLOUD_CUBOID_SEQUENCE,
    segments.typing.TaskType.POINTCLOUD_SEGMENTATION_SEQUENCE,
    segments.typing.TaskType.POINTCLOUD_VECTOR_SEQUENCE,
}

_ANNO_LABEL_MAP = {
    "bboxes": "bbox",
    "keypoints": "point",
    "vector": "polyline",
    "vector-filled": "polygon",
    "pointcloud-cuboid": "cuboid",
    "pointcloud-vector": "polyline",
}


class DatasetUploadTarget(enum.Enum):
    NEW = "New"
    APPEND = "Append"


class RequestAnnotations(foo.Operator):
    """This operator uploads samples from fiftyone to Segments.ai. It can setup a new Segments dataset or append data to an existing dataset."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="request_annotations",
            label="Request Segments.ai annotations",
            light_icon="/assets/Black icon.svg",
            dark_icon="/assets/White icon.svg",
            dynamic=True,
        )

    @staticmethod
    def target_data_selector(ctx, inputs):
        has_selected = bool(ctx.selected)
        has_view = ctx.view != ctx.dataset.view()
        default_choice = "full_dataset"

        target_choices = types.RadioGroup(orientation="horizontal")
        target_choices.add_choice(
            TargetSelection.DATASET.value,
            label="Entire dataset",
            description="Upload the entire dataset",
        )
        if has_view:
            target_choices.add_choice(
                TargetSelection.CURRENT_VIEW.value,
                label="Current view",
                description="Upload only the current view",
            )
            default_choice = TargetSelection.CURRENT_VIEW.value
        if has_selected:
            target_choices.add_choice(
                TargetSelection.SELECTED.value,
                label="Selected samples",
                description="Upload only the selected samples.",
            )
            default_choice = TargetSelection.SELECTED.value
        inputs.enum(
            "target",
            target_choices.values(),
            required=True,
            label="Target",
            view=target_choices,
            default=default_choice,
        )

    @staticmethod
    def dataset_type_selector(ctx, inputs, media_type: str, is_sequence):
        if media_type == "image":
            labelmap = {
                SegmentsDatasetType.SEGMENTATION_BITMAP: "Segmentation bitmap",
                SegmentsDatasetType.SEGMENTATION_BITMAP_HIGHRES: "Segmentation bitmap highres",
                SegmentsDatasetType.BBOXES: "Bounding boxes",
                SegmentsDatasetType.VECTOR: "Vector",
                SegmentsDatasetType.KEYPOINTS: "Keypoints",
            }

            default_selection = SegmentsDatasetType.SEGMENTATION_BITMAP.value

        elif media_type == "point-cloud" and not is_sequence:
            labelmap = {
                SegmentsDatasetType.POINTCLOUD_CUBOID: "Pointcloud cuboid",
                SegmentsDatasetType.POINTCLOUD_VECTOR: "Pointcloud vector",
                SegmentsDatasetType.POINTCLOUD_SEGMENTATION: "Pointcloud segmentation",
            }

            default_selection = SegmentsDatasetType.POINTCLOUD_CUBOID.value
        elif media_type == "point-cloud" and is_sequence:
            labelmap = {
                SegmentsDatasetType.POINTCLOUD_CUBOID_SEQUENCE: "Pointcloud cuboid sequence",
                SegmentsDatasetType.POINTCLOUD_VECTOR_SEQUENCE: "Pointcloud vector sequence",
                SegmentsDatasetType.POINTCLOUD_SEGMENTATION_SEQUENCE: "Pointcloud segmentation sequence",
            }

            default_selection = SegmentsDatasetType.POINTCLOUD_CUBOID.value
        elif media_type == "group":
            labelmap = {
                SegmentsDatasetType.MULTISENSOR_SEQUENCE: "Multisensor sequence",
                SegmentsDatasetType.POINTCLOUD_CUBOID_SEQUENCE: "Pointcloud cuboid sequence",
                SegmentsDatasetType.POINTCLOUD_VECTOR_SEQUENCE: "Pointcloud vector sequence",
                SegmentsDatasetType.POINTCLOUD_SEGMENTATION_SEQUENCE: "Pointcloud segmentation sequence",
            }
            default_selection = SegmentsDatasetType.MULTISENSOR_SEQUENCE.value
        else:
            raise ValueError(f"Not implemented for media type: {media_type}")

        data_choices = types.Dropdown()
        for datatype in labelmap.keys():
            data_choices.add_choice(datatype.value, label=labelmap[datatype])

        inputs.enum(
            "dataset_type",
            data_choices.values(),
            label="Dataset type",
            default=default_selection,
            view=data_choices,
        )

    def get_labels_list(self, ctx, inputs):
        dataset_view = self.target_dataset_view(ctx)
        label_fields = dataset_view.get_field_schema(embedded_doc_type = fo.Label)
        
        label_choices = types.Dropdown()
        for field in label_fields:
            label_choices.add_choice(field, label=field)

        inputs.enum(
            "label_field",
            label_choices.values(),
            label="Label field to upload",
            view=label_choices,
            description="Select the label field to upload to Segments.ai",
        )

    
    def _fetch_fo_label_points(self, single_label, width, height):
        if self.upload_label_type in (UploadLabelType.DETECTIONS, UploadLabelType.DETECTION):
            bbox = single_label.bounding_box
            all_pts = [[
                    [bbox[0] * width, bbox[1] * height],  # x0, y0 (upper left corner of bbox)
                    [(bbox[0] + bbox[2]) * width, (bbox[1] + bbox[3]) * height]  # x1, y1 (lower right corner of bbox)
                ]]
        elif self.upload_label_type in (UploadLabelType.POLYLINES, UploadLabelType.POLYLINE):
            scaled_points_list = single_label.points
            all_pts = []
            for scaled_points in scaled_points_list:
                points = []
                points = [[x * width, y * height] for x, y in scaled_points]
                all_pts.append(points)
        else:
            raise ValueError(f"Unsupported label type for upload: {self.upload_label_type}")
        return all_pts
    
    def _fetch_fo_label_points_3d(self, single_label, track_id, cat_map):
        if self.upload_label_type in (UploadLabelType.DETECTION_3D, UploadLabelType.DETECTIONS_3D):
            location = single_label.location
            dimensions = single_label.dimensions
            rotation_euler = single_label.rotation
            rotation = Rotation.from_euler("xyz", rotation_euler)
            qx, qy, qz, qw = rotation.as_quat()
            anno = [{
                "track_id": track_id,
                "id": track_id,
                "category_id": cat_map[single_label.label],
                "type": "cuboid",
                "position" : {
                    "x": location[0],
                    "y": location[1],
                    "z": location[2]
                },
                "dimensions": {
                    "x": dimensions[0],
                    "y": dimensions[1],
                    "z": dimensions[2]
                },
                "yaw": 0,
                "rotation": {
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "qw": qw
                }
            }]
            track_id += 1
        elif self.upload_label_type in (UploadLabelType.POLYLINES_3D, UploadLabelType.POLYLINE_3D):
            anno = []
            for pt_list in single_label.points3d:
                pts = [pt for pt in pt_list]
                anno_single = {
                    "track_id": track_id,
                    "id": track_id,
                    "category_id": cat_map[single_label.label],
                    "type": "polyline",
                    "points": pts
                }
                track_id += 1
                anno.append(anno_single)
        return anno, track_id

    def _fetch_fo_label_iterable(self, sample, label_field_name):
        if sample[label_field_name] is not None:
            if self.upload_label_type in (UploadLabelType.DETECTIONS, UploadLabelType.DETECTIONS_3D):
                return sample[label_field_name].detections
            elif self.upload_label_type in (UploadLabelType.POLYLINES, UploadLabelType.POLYLINES_3D):
                return sample[label_field_name].polylines
            else:
                return [sample[label_field_name]]
        else:
            return None
    
    def upload_selected_label_field(self, client, ctx, dataset_view, task_type):
        cat_map = self.cat_map
        label_field_name = ctx.params["label_field"]
        for sample in dataset_view:
            label_list = self._fetch_fo_label_iterable(sample, label_field_name)
            if label_list is None:
                continue
            if "3d" not in self.upload_label_type:
                width = sample.metadata.width
                height = sample.metadata.height
                annotations = []
                det_idx = 0
                for single_label in label_list:
                    if single_label.label not in cat_map:
                        continue
                    points = self._fetch_fo_label_points(single_label, width, height)
                    segments_task_type = task_type.value
                    if getattr(single_label, "filled",  False):
                            segments_task_type += "-filled"
                    for pts in points:
                        anno = {"track_id": det_idx + 1, 
                                "id": det_idx + 1,  #this is a legacy field and is always equal to track_id.
                                "category_id": cat_map[single_label.label], # the category id
                                "type": _ANNO_LABEL_MAP[segments_task_type], # refers to the annotation type (bounding box)
                                "points": pts}
                        annotations.append(anno)
                        det_idx += 1
                if len(annotations) > 0:
                    client.add_label(sample.segments_uuid, "ground-truth", {'format_version': '0.1', 'annotations': annotations})
            else:
                det_idx = 0
                annotations = []
                for single_label in label_list:
                    if single_label.label not in cat_map:
                        continue
                    anno, det_idx = self._fetch_fo_label_points_3d(single_label, det_idx + 1, cat_map)
                    annotations.extend(anno)
                if len(annotations) > 0:
                    if "detection" in self.upload_label_type:
                        client.add_label(sample.segments_uuid, "ground-truth", {'format_version': '0.2', 'annotations': annotations})
                    elif "polyline" in self.upload_label_type:
                        client.add_label(sample.segments_uuid, "ground-truth", {'format_version': '0.1', 'annotations': annotations})

    def _check_types_get_upload_label_type(self, dataset_view, label_field_name, task_type):
        if task_type in (SegmentsDatasetType.BBOXES.value, SegmentsDatasetType.POINTCLOUD_CUBOID.value):
            if label_field_name in dataset_view.get_field_schema(embedded_doc_type=fo.Detections):
                self.upload_label_type  = UploadLabelType.DETECTIONS
                if dataset_view.media_type == "point-cloud":
                    self.upload_label_type = UploadLabelType.DETECTIONS_3D
            elif label_field_name in dataset_view.get_field_schema(embedded_doc_type=fo.Detection):
                self.upload_label_type = UploadLabelType.DETECTION
                if dataset_view.media_type == "point-cloud":
                    self.upload_label_type = UploadLabelType.DETECTION_3D
            else:
                raise ValueError(
                    f"Label field '{label_field_name}' is not a valid Detections field"
                )
        elif task_type in (SegmentsDatasetType.VECTOR.value, SegmentsDatasetType.POINTCLOUD_VECTOR.value):
            if label_field_name in dataset_view.get_field_schema(embedded_doc_type=fo.Polylines):
                self.upload_label_type  = UploadLabelType.POLYLINES
                if dataset_view.media_type == "point-cloud":
                    self.upload_label_type = UploadLabelType.POLYLINES_3D
            elif label_field_name in dataset_view.get_field_schema(embedded_doc_type=fo.Polyline):
                self.upload_label_type = UploadLabelType.POLYLINE
                if dataset_view.media_type == "point-cloud":
                    self.upload_label_type = UploadLabelType.POLYLINE_3D
            else:
                raise ValueError(
                    f"Label field '{label_field_name}' is not a valid Polylines field"
                )
        else:
            raise ValueError(
                f"Upload of labels of type {task_type} not supported through the plugin"
            )
    
    def execute(self, ctx):
        data_upload_target = DatasetUploadTarget(ctx.params["dataset_choice"])
        dataset_view = self.target_dataset_view(ctx)
        label_field_name = ctx.params.get("label_field", None)

        client = get_client(ctx)
        if data_upload_target == DatasetUploadTarget.NEW:
            task_type = ctx.params["dataset_type"]
            label_classes = ctx.params["classes"]
            if label_field_name is not None:
                self._check_types_get_upload_label_type(dataset_view, label_field_name, task_type)
                if ctx.params["classes"] == []:
                    _, label_path = dataset_view._get_label_field_path(label_field_name, "label")
                    label_classes = sorted(set(dataset_view._dataset.distinct(label_path)) | set(dataset_view.distinct(label_path)))
            cat_map={}
            attributes = {"format_version": "0.1", "categories": []}
            for idx, cls in enumerate(label_classes):
                attributes["categories"].append({"id": idx + 1, "name": cls})
                cat_map[cls] = idx + 1
            self.cat_map = cat_map

            organization = None
            if ctx.params.get("in_organization"):
                organization = ctx.params["dataset_owner"]
            dataset = client.add_dataset(
                ctx.params["dataset_name"],
                description="Created by the segments fiftyone plugin.",
                metadata={"created_by": "fiftyone_plugin"},
                task_type=task_type,
                task_attributes=attributes,
                organization=organization,
            )
        elif data_upload_target == DatasetUploadTarget.APPEND:
            dataset_name, _ = _fetch_selected_dataset_name(ctx)
            dataset = client.get_dataset(dataset_name)

        task_type = SegmentsDatasetType(dataset.task_type)
        upload_dataset(client, dataset_view, dataset.full_name, task_type, ctx)
        if label_field_name is not None:
            self.upload_selected_label_field(client, ctx, dataset_view, task_type)

        url = urljoin(SEGMENTS_FRONTEND_URL, dataset.full_name)
        return {"segments_dataset": dataset.full_name, "url": url}

    def target_dataset_view(self, ctx):
        target = TargetSelection(ctx.params["target"])
        if target == TargetSelection.DATASET:
            return ctx.dataset
        elif target == TargetSelection.SELECTED:
            return ctx.view.select(ctx.selected)
        elif target == TargetSelection.CURRENT_VIEW:
            return ctx.view
        else:
            raise ValueError(f"Could not get target for {target=}")

    def resolve_input(self, ctx):
        inputs = types.Object()
        dataset_name, dataset_type = _fetch_selected_dataset_name(ctx)

        self.target_data_selector(ctx, inputs)

        default_dataset_choice = DatasetUploadTarget.NEW.value
        seg_dataset_choices = types.RadioGroup(orientation="horizontal")
        if dataset_name is not None:
            seg_dataset_choices.add_choice(
                DatasetUploadTarget.APPEND.value,
                label=DatasetUploadTarget.APPEND.value,
                description="Append samples to segments.ai dataset.",
            )
            default_dataset_choice = DatasetUploadTarget.APPEND.value

        seg_dataset_choices.add_choice(
            DatasetUploadTarget.NEW.value,
            label=DatasetUploadTarget.NEW.value,
            description="Create a new segments.ai dataset from these samples",
        )
        inputs.enum(
            "dataset_choice",
            seg_dataset_choices.values(),
            view=seg_dataset_choices,
            label="New or existing dataset?",
            default=default_dataset_choice,
        )
        label_field_name = ctx.params.get("label_field", None)

        if ctx.params.get("dataset_choice", "") == DatasetUploadTarget.NEW.value:
            inputs.str("dataset_name", label="Dataset Name")
            inputs.bool("in_organization", label="Add to organization")
            if ctx.params.get("in_organization", ""):
                user = get_client(ctx).get_user()
                org_choices = types.Choices()
                for org in user.organizations:
                    org_choices.add_choice(org.username)

                inputs.enum(
                    "dataset_owner",
                    org_choices.values(),
                    label="Dataset owner",
                    required=True,
                )
            is_sequence = dataset_has_dynamic_groups(self.target_dataset_view(ctx))
            self.dataset_type_selector(
                ctx, inputs, ctx.dataset.media_type, is_sequence
            )
            self.get_labels_list(ctx, inputs)

            inputs.list(
                "classes",
                types.String(),
                label="Classes",
                description="The annotation labels",
            )
            dataset_type = ctx.params.get("dataset_type", "")
        else:
            dset = types.Notice(
                label=f"Appending data to segments.ai dataset: {dataset_name}"
            )
            inputs.view("dataset_name", dset)

        if dataset_type == SegmentsDatasetType.MULTISENSOR_SEQUENCE.value:
            inputs.bool(
                "add_image_sensors",
                label="Add cameras as seperate sensors",
                description="This will add cameras as annotation tasks in the multisensor interface",
                views=types.CheckboxView(),
            )
        if ctx.params.get("target", ""):
            dataset_view = self.target_dataset_view(ctx)
            if label_field_name is not None:
                if dataset_type in (SegmentsDatasetType.BBOXES.value, SegmentsDatasetType.VECTOR.value, SegmentsDatasetType.KEYPOINTS.value):
                    if None in dataset_view.values("metadata"):
                        error_metadata = types.Error(
                            label="Some samples do not have metadata. Please compute metadata for all samples before uploading."
                        )
                        inputs.view("warning_no_metadata", error_metadata, invalid=True)

        sequence_tasks_str = set(map(lambda x: x.value, SEQUENCE_TASKS))
        if dataset_type in sequence_tasks_str:
            if ctx.params.get("target", "") == TargetSelection.DATASET.value:
                error_target = types.Error(
                    label=f"Can't upload the full dataset to segments for dataset type {dataset_type}. Please create a view using the dynamic grouping feature."
                )
                inputs.view("warning_no_full_dataset", error_target, invalid=True)
            else:
                if ctx.params.get("target", ""):
                    try:
                        next(dataset_view.iter_dynamic_groups())
                    except ValueError:
                        error_no_dynamic = types.Error(
                            label="No dynamic groups found. Please use the dynamic grouping feature to create sequences."
                        )
                        inputs.view(
                            "warning_no_full_dataset", error_no_dynamic, invalid=True
                        )

        return types.Property(inputs)

    def resolve_output(self, ctx):
        outputs = types.Object()
        view = types.View(label="New dataset created")
        outputs.str("url", label="Dataset URL", view=types.MarkdownView())
        return types.Property(outputs, view=view)


class FetchAnnotations(foo.Operator):
    """Fetches annotations from a Segments.ai release and attaches them to the fiftyone samples."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="fetch_annotations",
            light_icon="/assets/Black icon.svg",
            dark_icon="/assets/White icon.svg",
            label="Fetch Segments.ai annotations",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        dataset_name, _ = _fetch_selected_dataset_name(ctx)
        if dataset_name is None:
            return _no_dset_selected_warning(inputs)

        dset = types.Notice(
            label=f"Fetching annotations from segments.ai dataset: {dataset_name}"
        )
        inputs.view("dataset_name", dset)

        client = get_client(ctx)
        releases = client.get_releases(dataset_name)
        choices_releases = types.Choices()
        for release in releases:
            choices_releases.add_choice(release.name, label=release.name)

        inputs.enum(
            "release",
            choices_releases.values(),
            view=choices_releases,
            label="Release",
            required=True,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        run_result = ctx.dataset.load_run_results(SEGMENTS_METADATA_KEY, cache=False)
        dataset_name = run_result.dataset_full_name

        client = get_client(ctx)

        dataset_sdk = client.get_dataset(dataset_name)
        uuid_sample_map = create_uuid_sample_map(ctx.dataset, client, dataset_sdk)
        release = client.get_release(dataset_sdk.full_name, ctx.params["release"])

        dataset_type = SegmentsDatasetType(dataset_sdk.task_type)
        # Pointcloud-vector and multisensor are incompatible with SegmentsDataset, handle them seperately
        if dataset_type in (
            SegmentsDatasetType.POINTCLOUD_VECTOR,
            SegmentsDatasetType.MULTISENSOR_SEQUENCE,
            SegmentsDatasetType.POINTCLOUD_VECTOR_SEQUENCE,
            SegmentsDatasetType.POINTCLOUD_CUBOID_SEQUENCE,
        ):
            response = requests.get(release.attributes.url)
            response.raise_for_status()
            releasefile = response.json()

            if dataset_type == SegmentsDatasetType.POINTCLOUD_VECTOR:
                insert_cuboid_labels(releasefile, ctx.dataset, uuid_sample_map)
            else:
                insert_multisensor_labels(
                    releasefile, ctx.dataset, uuid_sample_map, dataset_type
                )
            # else:
            #     raise ValueError(f"Unexpected datset_type: {dataset_type}")
        else:
            dataloader = SegmentsDataset(release, preload=False, load_images=False)
            if dataset_type in (
                SegmentsDatasetType.SEGMENTATION_BITMAP,
                SegmentsDatasetType.SEGMENTATION_BITMAP_HIGHRES,
            ):
                insert_segmentation_labels(dataloader, ctx.dataset, uuid_sample_map)
            elif dataset_type in (
                SegmentsDatasetType.BBOXES,
                SegmentsDatasetType.KEYPOINTS,
                SegmentsDatasetType.VECTOR,
            ):
                insert_vector_labels(dataloader, ctx.dataset, uuid_sample_map)
            elif dataset_type == SegmentsDatasetType.POINTCLOUD_CUBOID:
                insert_cuboid_labels(dataloader, ctx.dataset, uuid_sample_map)
            elif dataset_type == SegmentsDatasetType.POINTCLOUD_SEGMENTATION:
                raise ValueError(
                    "Importing pointcloud segmentation projects not yet supported"
                )
            else:
                raise ValueError(
                    f"Dataset type '{dataset_type.value}' not yet supported"
                )

        ctx.ops.reload_dataset()

    def resolve_output(self, ctx):
        outputs = types.Object()
        view = types.View(label="Succesfully pulled annotations")
        return types.Property(outputs, view=view)


class AddIssue(foo.Operator):
    """Adds an issue to a Segments.ai sample from within fiftyone."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="add_issue",
            light_icon="/assets/Black icon.svg",
            dark_icon="/assets/White icon.svg",
            label="Add issue to Segments.ai sample",
            dynamic=False,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        if not bool(ctx.selected) or len(ctx.selected) > 1:
            warning = types.Warning(label="Please select 1 sample")
            prop = inputs.view("warning", warning)
            prop.invalid = True
            return types.Property(
                inputs, view=types.View(label="Add issue to segments.ai")
            )

        dataset_full_name, _ = _fetch_selected_dataset_name(ctx)
        if dataset_full_name is None:
            return _no_dset_selected_warning(inputs)

        dset = types.Notice(
            label=f"Making issue in segments.ai dataset: {dataset_full_name}"
        )
        inputs.view("dataset_name", dset)
        inputs.str(
            "description",
            allow_empty=False,
            view=types.TextFieldView(label="Issue description"),
        )
        return types.Property(inputs)

    def execute(self, ctx):
        s_id = ctx.selected[0]
        selected_sample = ctx.dataset[s_id]

        client = get_client(ctx)
        client.add_issue(selected_sample["segments_uuid"], ctx.params["description"])

    def resolve_output(self, ctx):
        pass


class SelectDataset(foo.Operator):
    """Select the corresponding Segments.ai dataset for this fiftyone dataset. This is required for other operators that interact with Segments.ai."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="select_segments_dataset",
            light_icon="/assets/Black icon.svg",
            dark_icon="/assets/White icon.svg",
            label="Select corresponding Segments.ai dataset",
            dynamic=False,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        client = get_client(ctx)

        datasets = client.get_datasets()
        filtered_dataset = []
        for dataset in datasets:
            if task_type_matches(ctx.dataset.media_type, dataset.task_type):
                filtered_dataset.append(
                    {"full_name": dataset.full_name, "name": dataset.name}
                )

        choices_dataset = types.AutocompleteView()
        for dataset in filtered_dataset:
            choices_dataset.add_choice(dataset["full_name"], label=dataset["full_name"])

        inputs.str("dataset", view=choices_dataset, label="Dataset", required=True)

        return types.Property(inputs)

    def execute(self, ctx):
        try:
            config = fo.RunConfig()
            ctx.dataset.register_run(SEGMENTS_METADATA_KEY, config)
        except ValueError:
            # Run config already exists, no operation necessary
            pass

        client = get_client(ctx)
        results = ctx.dataset.init_run_results(SEGMENTS_METADATA_KEY)
        results.dataset_full_name = ctx.params["dataset"]
        results.dataset_type = str(client.get_dataset(ctx.params["dataset"]).task_type)

        ctx.dataset.save_run_results(SEGMENTS_METADATA_KEY, results, overwrite=True)


## Helper functions

AssetInfo = namedtuple("AssetInfo", "url filename")
SequenceMapKey = namedtuple("SequenceMapKey", "uuid frame_idx sensor_name")


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def array(self):
        return np.array((self.x, self.y, self.z))

    def transform(self, tmat: np.ndarray):
        t_self = self.array()
        t_homog = np.concatenate((t_self, [1]))
        transformed = tmat @ t_homog

        return Point3D(*transformed[:3])


def pcd_filename_map(dataset: fo.Dataset) -> dict[str, fo.Sample]:
    if dataset.media_type != "3d":
        return {Path(s.filepath).name: s for s in dataset}
    else:
        try:
            return {s["segments_pc_filename"]: s for s in dataset}
        except KeyError:
            raise KeyError(
                "Expected to find 'source_pcd_filename' attribute in sample. This is required to match segments.ai annotations with fiftyone samples."
            )


def create_uuid_sample_map(
    dataset: fo.Dataset,
    client: segments.SegmentsClient,
    segments_dataset: segments.typing.Dataset,
) -> dict[str, fo.Sample]:
    """Creates a dictionary mapping a Segments uuid string to a fiftyone sample."""
    if dataset.media_type == "group":
        map_ = create_uuid_sample_map_grouped(dataset)
    elif segments_dataset.task_type in (
        segments.typing.TaskType.POINTCLOUD_CUBOID_SEQUENCE,
        segments.typing.TaskType.POINTCLOUD_SEGMENTATION_SEQUENCE,
        segments.typing.TaskType.POINTCLOUD_VECTOR_SEQUENCE,
    ):
        map_ = create_uuid_sample_map_sequence(dataset)
    else:
        map_ = create_uuid_sample_map_local(dataset)
        reversed_maps = {value.id: key for (key, value) in map_.items()}

        segments_samples = None
        # Extend map by matching filenames
        for sample in dataset:
            if sample.id in reversed_maps:
                # Already matched
                continue

            if segments_samples is None:
                # Lazily fetch the samples
                segments_samples = client.get_samples(segments_dataset.full_name)
                sample_name_to_id = {s.name: s.uuid for s in segments_samples}

            fo_name = Path(sample.filepath).name
            if fo_name in sample_name_to_id:
                map_[sample_name_to_id[fo_name]] = sample

    return map_


def create_uuid_sample_map_local(dataset: fo.Dataset) -> dict[str, fo.Sample]:
    """Creates a dictionary mapping a Segments uuid string to a fiftyone sample."""
    map_ = {}
    for sample in dataset:
        uuid = sample["segments_uuid"]
        if uuid is not None:
            map_[uuid] = sample

    return map_


def create_uuid_sample_map_sequence(dataset) -> dict[SequenceMapKey, fo.Sample]:
    sample_mapping = {}
    for sample in dataset:
        add_sample_to_uuid_map(sample_mapping, sample)

    return sample_mapping


def create_uuid_sample_map_grouped(
    dataset: fo.Dataset,
) -> dict[SequenceMapKey, fo.Sample]:
    sample_mapping = {}

    # Iterate over all groups in the dataset
    for group in dataset.iter_groups():
        # Iterate over all slices in the group
        for sample in group.values():
            add_sample_to_uuid_map(sample_mapping, sample)

    return sample_mapping


def add_sample_to_uuid_map(sample_mapping, sample):
    if "segments_uuid" not in sample:
        return

    # Extract the necessary fields
    sensor_name = sample.segments_sensor_name
    uuid = sample.segments_uuid
    frame_idx = sample.segments_frame_idx

    # Create a unique key for the dictionary
    key = SequenceMapKey(sensor_name=sensor_name, uuid=uuid, frame_idx=frame_idx)

    # Map the key to the sample
    sample_mapping[key] = sample


def is_cloud_storage(path) -> bool:
    parse_result = urlparse(path)
    if parse_result.scheme == "":
        # no parsed scheme, assume local file
        return False
    else:
        # If scheme provided, assume cloud storage
        # TODO: Check for supported schemes
        return True


def insert_segmentation_labels(
    dataloader: SegmentsDataset, dataset: fo.Dataset, sample_map: dict[str, fo.Sample]
):
    catmap = {x.id: x.name for x in dataloader.categories}
    dataset.mask_targets["ground_truth_segmentation"] = catmap
    dataset.save()
    annotation_count = 0
    for annotation in dataloader:
        if annotation["segmentation_bitmap"] is None:
            # No segmentation annotation, skip
            continue

        sample = sample_map[annotation["uuid"]]
        segmap_instance = np.asarray(annotation["segmentation_bitmap"])
        id_id_map = {x["id"]: x["category_id"] for x in annotation["annotations"]}
        if 0 not in id_id_map:
            id_id_map[0] = 0

        id_id_func = np.vectorize(lambda x: id_id_map[x])
        segmap = id_id_func(segmap_instance)

        label = fo.Segmentation(mask=segmap)
        sample.add_labels(label, label_field="ground_truth_segmentation")
        sample.save()
        annotation_count += 1


def insert_vector_labels(
    dataloader: SegmentsDataset, dataset: fo.Dataset, sample_map: dict[str, fo.Sample]
):
    id_cat_map = {x.id: x.name for x in dataloader.categories}
    for annotation in dataloader:
        if annotation["annotations"] is None:
            continue

        sample = sample_map[annotation["uuid"]]
        if sample.metadata is None:
            sample.compute_metadata()

        image_width = sample.metadata.width
        image_height = sample.metadata.height
        image_size = np.array((image_width, image_height))

        detections = []
        polygons = []
        polylines = []
        keypoints = []

        for instance in annotation["annotations"]:
            category_name = id_cat_map[instance["category_id"]]
            if instance["type"] == "bbox":
                detection = create_51_bbox(instance, image_size, category_name)
                detections.append(detection)
            elif instance["type"] == "polygon":
                polygon = create_51_polyline(
                    instance, image_size, category_name, is_polygon=True
                )
                polygons.append(polygon)
            elif instance["type"] == "polyline":
                polyline = create_51_polyline(
                    instance, image_size, category_name, is_polygon=False
                )
                polylines.append(polyline)
            elif instance["type"] == "point":
                keypoint = create_51_keypoint(instance, image_size, category_name)
                keypoints.append(keypoint)
            else:
                raise ValueError(f"Could not parse annotation type: {instance['type']}")

        if detections:
            det_sample = fo.Detections(detections=detections)
            sample["ground_truth_bboxes"] = det_sample
        if polygons:
            pol_sample = fo.Polylines(polylines=polygons)
            sample["ground_truth_polygons"] = pol_sample
        if polylines:
            pol_sample = fo.Polylines(polylines=polylines)
            sample["ground_truth_polylines"] = pol_sample
        if keypoints:
            kp_sample = fo.Keypoints(keypoints=keypoints)
            sample["ground_truth_points"] = kp_sample

        sample.save()


def insert_cuboid_labels(
    dataloader: Union[SegmentsDataset, dict],
    dataset: fo.Dataset,
    sample_map: dict[str, fo.Sample],
):
    if isinstance(dataloader, SegmentsDataset):
        id_cat_map = {x.id: x.name for x in dataloader.categories}
        iterable = dataloader
    else:
        categories = dataloader["dataset"]["task_attributes"]["categories"]
        id_cat_map = {x["id"]: x["name"] for x in categories}
        iterable = dataloader["dataset"]["samples"]

    for annotation in iterable:
        if annotation["labels"]["ground-truth"] is None:
            continue

        sample = sample_map[annotation["uuid"]]

        _insert_sample_annotations_cuboid(
            sample,
            annotation["labels"]["ground-truth"]["attributes"]["annotations"],
            id_cat_map,
        )


def _insert_sample_annotations_cuboid(
    sample: fo.Sample,
    annotation: dict,
    id_cat_map: Dict[int, str],
    egomotion: Optional[np.ndarray] = None,
):
    cuboids = []
    polygons = []
    polylines = []
    keypoints = []
    for instance in annotation:
        category_name = id_cat_map[instance["category_id"]]
        type_ = instance["type"]

        if type_ == "cuboid":
            cuboid = create_51_cuboid(instance, category_name, egomotion)
            cuboids.append(cuboid)
        elif type_ == "polygon":
            polygon = create_51_3dpolygon(instance, category_name, is_polygon=True)
            polygons.append(polygon)
        elif type_ == "polyline":
            polyline = create_51_3dpolygon(instance, category_name, is_polygon=False)
            polylines.append(polyline)
        elif type_ == "point":
            pass  # Not supported by fiftyone, somehow warn the user?
        else:
            raise ValueError(f"Not implemented for annoation type: {type_}")

    if cuboids:
        det_sample = fo.Detections(detections=cuboids)
        sample["ground_truth_cuboids"] = det_sample
    if polygons:
        pol_sample = fo.Polylines(polylines=polygons)
        sample["ground_truth_polygons"] = pol_sample
    if polylines:
        pol_sample = fo.Polylines(polylines=polylines)
        sample["ground_truth_polylines"] = pol_sample
    if keypoints:
        pol_sample = fo.Polylines(polylines=keypoints)
        sample["ground_truth_points"] = pol_sample

    sample.save()


def insert_multisensor_labels(
    dataloader: dict,
    dataset: fo.Dataset,
    sample_map: dict[SequenceMapKey, fo.Sample],
    dataset_type: SegmentsDatasetType,
):
    def ego_to_transmat(ego: dict):
        rot = Quaternion(
            x=ego["heading"]["qx"],
            y=ego["heading"]["qy"],
            z=ego["heading"]["qz"],
            w=ego["heading"]["qw"],
        )
        pos = np.array(
            [
                ego["position"]["x"],
                ego["position"]["y"],
                ego["position"]["z"],
            ]
        )
        tmat = np.eye(4)
        tmat[:3, :3] = rot.rotation_matrix
        tmat[:3, 3] = pos

        return tmat

    def get_egomotion_frames(sensors):
        for sensor in sensors:
            if "ego_pose" in sensor["attributes"]["frames"][0]:
                return sensor["attributes"]["frames"]

        return None

    def get_egomotion(sample: dict):
        if "sensors" in sample["attributes"]:
            sensors = sample["attributes"]["sensors"]
            frames = get_egomotion_frames(sensors)
            if frames is None:
                # No ego pose found, return None
                return None
        else:
            frames = sample["attributes"]["frames"]
            if "ego_pose" not in frames[0]:
                return None

        ego_poses = [x["ego_pose"] for x in frames]
        ego_poses = [ego_to_transmat(x) for x in ego_poses]
        return ego_poses

    categories = dataloader["dataset"]["task_attributes"]["categories"]
    id_cat_map = {x["id"]: x["name"] for x in categories}
    segments_samples = dataloader["dataset"]["samples"]

    for annotation in segments_samples:
        if (label_dict := annotation["labels"]["ground-truth"]) is None:
            continue
        uuid = annotation["uuid"]
        egomotion = get_egomotion(annotation)

        if dataset_type == SegmentsDatasetType.MULTISENSOR_SEQUENCE:
            insert_multisensor_annotations(
                sample_map, id_cat_map, label_dict, uuid, egomotion
            )
        else:
            insert_pointcloud_sequence_annotations(
                sample_map, id_cat_map, label_dict, uuid, egomotion
            )


def insert_pointcloud_sequence_annotations(
    sample_map, id_cat_map, label_dict, sample_uuid, egomotion
):
    for f_idx, frame in enumerate(label_dict["attributes"]["frames"]):
        ann = frame["annotations"]
        key = SequenceMapKey(sample_uuid, f_idx, "sample")
        if key not in sample_map:
            continue

        sample = sample_map[key]
        egomotion_this_frame = None if egomotion is None else egomotion[f_idx]
        _insert_sample_annotations_cuboid(sample, ann, id_cat_map, egomotion_this_frame)


def insert_multisensor_annotations(
    sample_map, id_cat_map, label_dict, sample_uuid, egomotion
):
    sensors = label_dict["attributes"]["sensors"]
    for sensor in sensors:
        sensor_name = sensor["name"]
        for f_idx, frame in enumerate(sensor["attributes"]["frames"]):
            ann = frame["annotations"]
            key = SequenceMapKey(sample_uuid, f_idx, sensor_name)
            if key not in sample_map:
                continue

            sample = sample_map[key]
            egomotion_this_frame = None if egomotion is None else egomotion[f_idx]
            _insert_sample_annotations_cuboid(
                sample, ann, id_cat_map, egomotion_this_frame
            )


# Caching the client object, as constructing it is relatively expensive
_CLIENT: Optional[SegmentsClient] = None


def get_client(ctx) -> SegmentsClient:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = ctx.secrets.get("SEGMENTS_API_KEY")
    # Sometimes a missing secret is `None`, sometimes it's an empty string.
    segments_url = ctx.secrets.get("SEGMENTS_URL", None)
    if segments_url:
        client = SegmentsClient(api_key, api_url=segments_url)
    else:
        client = SegmentsClient(api_key)

    _CLIENT = client
    return client


def upload_dataset(
    client: SegmentsClient,
    dataset: fo.Dataset,
    dataset_id: str,
    task_type: SegmentsDatasetType,
    ctx,
):
    if dataset.media_type == "group":
        dataset_iterator = dataset.iter_dynamic_groups()
    else:
        dataset_iterator = dataset

    def needs_bucket_upload(sample):
        no_alternate_filepath = "segments_filepath" not in sample
        not_cloud_storage = not is_cloud_storage(sample.filepath)

        return no_alternate_filepath and not_cloud_storage

    sample = next(iter(dataset))
    upload_limit = 100
    if len(dataset) > upload_limit and needs_bucket_upload(sample):
        raise ValueError(
            f"The dataset is too large to upload using this plugin (larger than {upload_limit}). Please upload the samples to a cloud bucket and provide the URLs in the 'segments_filepath' field."
        )

    for idx, s in enumerate(dataset_iterator):
        ctx.set_progress(
            (idx + 1) / len(dataset), label=f"Uploading {idx+1}/(len(dataset))"
        )

        # If the sample is stored in a cloud bucket, don't upload it to segments.ai. Instead, use the URL directly.
        if isinstance(s, fo.Sample) or isinstance(s, fo.core.sample.SampleView):
            asset_info = [upload_single_sample(client, s)]
        else:
            asset_info = upload_sequence_sample(client, s)

        sample_attrib, sample_name = generate_sample_attribs(
            s, asset_info, task_type, ctx.params.get("add_image_sensors", False)
        )

        segments_sample = client.add_sample(
            dataset_id, sample_name, attributes=sample_attrib
        )
        if task_type == SegmentsDatasetType.MULTISENSOR_SEQUENCE:
            # For each of the samples, record uuid, frame index and sensor name
            for groupname in s.group_slices:
                s.group_slice = groupname
                s.set_values("segments_uuid", [segments_sample.uuid] * len(s))
                s.set_values("segments_frame_idx", range(len(s)))
                s.set_values("segments_sensor_name", [groupname] * len(s))
        elif task_type in SEQUENCE_TASKS:
            s.set_values("segments_uuid", [segments_sample.uuid] * len(s))
            s.set_values("segments_frame_idx", range(len(s)))
            s.set_values("segments_sensor_name", ["sample"] * len(s))
        else:
            s["segments_uuid"] = segments_sample.uuid
            s.save()


def upload_media_sample(client, sample):
    if "segments_filepath" in sample:
        url = sample["segments_filepath"]
        filename = url.rsplit("/", 1)[-1]
    elif is_cloud_storage(sample.filepath):
        url = sample.filepath
        filename = url.rsplit("/", 1)[-1]
    else:
        with open(sample.filepath, "rb") as f:
            asset = client.upload_asset(f, Path(sample.filepath).name)
            url = asset.url
            filename = asset.filename

    return AssetInfo(url, filename)


def upload_sequence_sample(
    client: segments.SegmentsClient, s: fo.DatasetView
) -> List[Dict[str, AssetInfo]]:

    if s.group_field is not None:
        asset_infos = _upload_sequence_sample_groups(client, s)
    else:
        asset_infos = _upload_sequence_sample_nogroup(client, s)

    return asset_infos


def _upload_sequence_sample_groups(client: segments.SegmentsClient, s):
    asset_infos = []
    for sensors in s.iter_groups():
        asset_info = {}
        for key, sample in sensors.items():
            asset_info[key] = upload_media_sample(client, sample)
        asset_infos.append(asset_info)

    return asset_infos


def _upload_sequence_sample_nogroup(client: segments.SegmentsClient, s: fo.DatasetView):
    asset_infos = []
    for idx, frame in enumerate(s):
        asset_info = {}
        asset_info["sample"] = upload_media_sample(client, frame)
        asset_infos.append(asset_info)

    return asset_infos


def upload_single_sample(
    client: segments.SegmentsClient, s: fo.Sample
) -> Dict[str, AssetInfo]:

    info = upload_media_sample(client, s)
    asset_info = {"sample": info}

    return asset_info


def generate_sample_attribs(
    sample_info: Union[fo.DatasetView, fo.Sample],
    asset_infos: List[Dict[str, AssetInfo]],
    task_type: SegmentsDatasetType,
    include_image_sensors: bool = False,
):
    if task_type in SINGLE_IMAGE_TASKS:
        asset_info = asset_infos[0]["sample"]
        sample_attrib = {"image": {"url": asset_info.url}}
        sample_name = asset_info.filename
    elif task_type in SINGLE_POINTCLOUD_TASKS:
        asset_info = asset_infos[0]["sample"]
        sample_attrib = {"pcd": {"url": asset_info.url, "type": "pcd"}}
        sample_name = asset_info.filename
    elif task_type == SegmentsDatasetType.MULTISENSOR_SEQUENCE:
        sensors = []
        sensors.append(_generate_attrib_frames_lidar_group(sample_info, asset_infos))
        if include_image_sensors:
            sensors.extend(
                _generate_attrib_frames_image_group(sample_info, asset_infos)
            )

        sample_attrib = {"sensors": sensors}
        # TODO: Provide a way to customize the sample names
        sample_name = sample_info.first().id

    elif task_type in POINTCLOUD_SEQUENCE_TASKS:
        sample_attrib = _generate_attrib_frames_lidar(sample_info, asset_infos)
        # TODO: Provide a way to customize the sample names
        sample_name = sample_info.first().id
    else:
        # TODO: add support for media type '3d'
        raise ValueError(f"Dataset upload not implemented for media type: {task_type}")

    return sample_attrib, sample_name


def _generate_attrib_frames_image_group(
    sample_info: fo.DatasetView, asset_infos: List[Dict[str, AssetInfo]]
):
    image_sensors = []
    sensors = next(sample_info.iter_groups())
    for sensor_name, sensor_sample in sensors.items():
        if sensor_sample.media_type != "image":
            continue
            # image_sensors[sensor_name] = sensor_sample

        sensor_attribs = {"name": sensor_name, "task_type": "image-vector-sequence"}
        frames = []
        for asset_info in asset_infos:
            frame = {}
            frame["name"] = Path(asset_info[sensor_name].filename).name
            frame["image"] = {"url": asset_info[sensor_name].url}
            frames.append(frame)

        sensor_attribs["attributes"] = {"frames": frames}

        image_sensors.append(sensor_attribs)

    return image_sensors


def _pointcloud_frame_from_sample(lidar_sample, asset_info):
    frame = {}
    frame["name"] = Path(asset_info.filename).name
    frame["pcd"] = {"url": asset_info.url, "type": "pcd"}
    if lidar_sample.metadata is not None and "position" in lidar_sample.metadata:
        frame["ego_pose"] = {
            "position": {
                "x": lidar_sample.metadata.position["x"],
                "y": lidar_sample.metadata.position["y"],
                "z": lidar_sample.metadata.position["z"],
            },
            "heading": {
                "qw": lidar_sample.metadata.heading["qw"],
                "qx": lidar_sample.metadata.heading["qx"],
                "qy": lidar_sample.metadata.heading["qy"],
                "qz": lidar_sample.metadata.heading["qz"],
            },
        }

    return frame


def _generate_attrib_frames_lidar_group(
    sample_info: fo.DatasetView, asset_infos: List[Dict[str, AssetInfo]]
):
    sensors = next(sample_info.iter_groups())
    for sensor_name, sensor_sample in sensors.items():
        if sensor_sample.media_type == "point-cloud":
            break
    else:
        raise ValueError("Could not find a pointcloud slice")

    pc_name = sensor_name

    sensor_attribs = {"name": pc_name, "task_type": "pointcloud-cuboid-sequence"}
    frames = []
    for sensors, asset_info in zip(sample_info.iter_groups(), asset_infos):
        lidar_sample = sensors[pc_name]

        frame = _pointcloud_frame_from_sample(lidar_sample, asset_info[pc_name])

        images = []
        for sensor_name, sensor_sample in sensors.items():
            if sensor_sample.media_type != "image":
                continue

            image_info = {
                "name": sensor_name,
                "url": asset_info[sensor_name].url,
            }
            if sensor_sample.metadata is not None:
                if sensor_sample.metadata.intrinsic_matrix is not None:
                    image_info["intrinsics"] = {
                        "intrinsic_matrix": sensor_sample.metadata.intrinsic_matrix
                    }

                if sensor_sample.metadata.extrinsics_translation is not None:
                    image_info["extrinsics"] = {
                        "translation": sensor_sample.metadata.extrinsics_translation,
                        "rotation": sensor_sample.metadata.extrinsics_rotation,
                    }

                if sensor_sample.metadata.camera_convention is not None:
                    image_info["camera_convention"] = (
                        sensor_sample.metadata.camera_convention
                    )

            images.append(image_info)

        frame["images"] = images

        frames.append(frame)

    sensor_attribs["attributes"] = {"frames": frames}

    return sensor_attribs


def _generate_attrib_frames_lidar(view, asset_infos):
    if len(asset_infos) == 0:
        raise ValueError(f"No frames in sample {view}")

    if "sample" in asset_infos[0]:
        frames = []
        for sample, asset_info in zip(view, asset_infos):
            frame = _pointcloud_frame_from_sample(sample, asset_info["sample"])
            frames.append(frame)

        attributes = {"frames": frames}
    else:
        # Re-use the multisensor attribute generation code. We need to unpack the `attributes` though.
        attributes = _generate_attrib_frames_lidar_group(view, asset_infos)[
            "attributes"
        ]

    return attributes


def task_type_matches(media_type: str, seg_task_type: segments.typing.TaskType) -> bool:
    if media_type == "image":
        return seg_task_type in IMAGE_TASKS_SEGMENTS

    elif media_type == "point-cloud" or media_type == "3d":
        return seg_task_type in POINTCLOUD_TASKS_SEGMENTS
    elif media_type == "group":
        return seg_task_type in (segments.typing.TaskType.MULTISENSOR_SEQUENCE,)
    else:
        raise ValueError(f"Not implemented for media type: {media_type}")


def _no_dset_selected_warning(inputs):
    warning = types.Warning(
        label="No segments.ai dataset selected. Please select one using the select_segments_dataset operator."
    )
    prop = inputs.view("warning", warning)
    prop.invalid = True
    return types.Property(inputs, view=types.View(label="No dataset selected"))


def _fetch_selected_dataset_name(ctx) -> Optional[Tuple[str, str]]:
    try:
        run_result = ctx.dataset.load_run_results(SEGMENTS_METADATA_KEY, cache=False)
        name = run_result.dataset_full_name
        type_ = run_result.dataset_type
        return (name, type_)
    except ValueError:
        return None, None


def create_51_cuboid(
    instance: dict, category_name: str, egomotion: Optional[np.ndarray] = None
):
    position = Point3D(**instance["position"])
    dims = Point3D(**instance["dimensions"])
    rotation = Rotation.from_quat(
        [
            instance["rotation"]["qx"],
            instance["rotation"]["qy"],
            instance["rotation"]["qz"],
            instance["rotation"]["qw"],
        ]
    ).as_euler("xyz")
    if egomotion is not None:
        egomotion = np.linalg.inv(egomotion)
        position = position.transform(egomotion)

        # egomotion[:3, 3] = 0
        rotation_orig = Rotation.from_euler("xyz", rotation)
        rotmatrix = egomotion[:3, :3] @ rotation_orig.as_matrix()
        rotation = Rotation.from_matrix(rotmatrix).as_euler("xyz")

    position = position.array().tolist()
    dims = dims.array().tolist()

    detection = fo.Detection(
        label=category_name,
        location=position,
        dimensions=dims,
        rotation=rotation.tolist(),
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
    if is_polygon:
        points = np.concatenate((points, points[0:1, :]), axis=0)

    line = fo.Polyline(label=category_name, points3d=[points.tolist()])
    return line


def dataset_has_dynamic_groups(dataset):
    try:
        iter = dataset.iter_dynamic_groups()
        next(iter)
        return True
    except (AttributeError, ValueError):
        return False


def register(p):
    p.register(RequestAnnotations)
    p.register(FetchAnnotations)
    p.register(AddIssue)
    p.register(SelectDataset)
