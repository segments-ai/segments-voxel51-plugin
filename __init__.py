"""
Operators for integrating with segments.ai
"""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse, urljoin

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import numpy as np
import requests
import segments
import segments.typing
from segments import SegmentsClient, SegmentsDataset

SEGMENTS_FRONTEND_URL = "https://segments.ai"
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
    # Disabling sequences for now, how would that interface work?
    # IMAGE_SEGMENTATION_SEQUENCE = "image-segmentation-sequence"
    # IMAGE_VECTOR_SEQUENCE = "image-vector-sequence"
    POINTCLOUD_CUBOID = "pointcloud-cuboid"
    POINTCLOUD_SEGMENTATION = "pointcloud-segmentation"
    POINTCLOUD_VECTOR = "pointcloud-vector"
    # POINTCLOUD_CUBOID_SEQUENCE = "pointcloud-cuboid-sequence"
    # POINTCLOUD_SEGMENTATION_SEQUENCE = "pointcloud-segmentation-sequence"
    # POINTCLOUD_VECTOR_SEQUENCE = "pointcloud-vector-sequence"


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
    def dataset_type_selector(ctx, inputs, media_type: str):
        if media_type == "image":
            labelmap = {
                SegmentsDatasetType.SEGMENTATION_BITMAP: "Segmentation bitmap",
                SegmentsDatasetType.SEGMENTATION_BITMAP_HIGHRES: "Segmentation bitmap highres",
                SegmentsDatasetType.BBOXES: "Bounding boxes",
                SegmentsDatasetType.VECTOR: "Vector",
                SegmentsDatasetType.KEYPOINTS: "Keypoints",
            }
            data_choices = types.Dropdown()
            for datatype in labelmap.keys():
                data_choices.add_choice(datatype.value, label=labelmap[datatype])

            default_selection = SegmentsDatasetType.SEGMENTATION_BITMAP.value

        elif media_type == "point-cloud":
            labelmap = {
                SegmentsDatasetType.POINTCLOUD_CUBOID: "Pointcloud cuboid",
                SegmentsDatasetType.POINTCLOUD_VECTOR: "Pointcloud vector",
                SegmentsDatasetType.POINTCLOUD_SEGMENTATION: "Pointcloud segmentation",
            }
            data_choices = types.Dropdown()
            for datatype in labelmap.keys():
                data_choices.add_choice(datatype.value, label=labelmap[datatype])

            default_selection = SegmentsDatasetType.POINTCLOUD_CUBOID.value
        else:
            raise ValueError(f"Not implemented for media type: {media_type}")

        inputs.enum(
            "dataset_type",
            data_choices.values(),
            label="Dataset type",
            default=default_selection,
            view=data_choices,
        )

    def execute(self, ctx):
        data_upload_target = DatasetUploadTarget(ctx.params["dataset_choice"])
        dataset_view = self.target_dataset_view(ctx)

        client = get_client(ctx)
        if data_upload_target == DatasetUploadTarget.NEW:
            task_type = ctx.params["dataset_type"]
            attributes = {"format_version": "0.1", "categories": []}
            for idx, cls in enumerate(ctx.params["classes"]):
                attributes["categories"].append({"id": idx + 1, "name": cls})

            dataset = client.add_dataset(
                ctx.params["dataset_name"],
                description="Created by the segments fiftyone plugin.",
                metadata={"created_by": "fiftyone_plugin"},
                task_type=task_type,
                task_attributes=attributes,
            )
        elif data_upload_target == DatasetUploadTarget.APPEND:
            dataset_name = _fetch_selected_dataset_name(ctx)
            dataset = client.get_dataset(dataset_name)

        upload_dataset(client, dataset_view, dataset.full_name, ctx)

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
        dataset_name = _fetch_selected_dataset_name(ctx)

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

        if ctx.params.get("dataset_choice", "") == DatasetUploadTarget.NEW.value:
            inputs.str("dataset_name", label="Dataset Name")
            self.dataset_type_selector(ctx, inputs, ctx.dataset.media_type)

            inputs.list(
                "classes",
                types.String(),
                label="Classes",
                description="The annotation labels",
            )
        else:
            dset = types.Notice(
                label=f"Appending data to segments.ai dataset: {dataset_name}"
            )
            inputs.view("dataset_name", dset)

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
        dataset_name = _fetch_selected_dataset_name(ctx)
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
        # Pointcloud-vector is incompatible with SegmentsDataset, handle it seperately
        if dataset_type == SegmentsDatasetType.POINTCLOUD_VECTOR:
            response = requests.get(release.attributes.url)
            response.raise_for_status()
            releasefile = response.json()
            insert_cuboid_labels(releasefile, ctx.dataset, uuid_sample_map)
        else:
            dataloader = SegmentsDataset(release, preload=False)

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

        dataset_full_name = _fetch_selected_dataset_name(ctx)
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

        choices_dataset = types.Choices()
        for dataset in filtered_dataset:
            choices_dataset.add_choice(dataset["full_name"], label=dataset["name"])

        inputs.enum(
            "dataset",
            choices_dataset.values(),
            view=choices_dataset,
            label="Dataset",
            required=True,
        )

        return types.Property(inputs)

    def execute(self, ctx):
        try:
            config = fo.RunConfig()
            ctx.dataset.register_run(SEGMENTS_METADATA_KEY, config)
        except ValueError:
            # Run config already exists, no operation necessary
            pass

        results = ctx.dataset.init_run_results(SEGMENTS_METADATA_KEY)
        results.dataset_full_name = ctx.params["dataset"]

        ctx.dataset.save_run_results(SEGMENTS_METADATA_KEY, results, overwrite=True)


## Helper functions


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def array(self):
        return np.array((self.x, self.y, self.z))


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
    map_ = create_uuid_sample_map_local(dataset)
    reversed_maps = {value.id: key for (key, value) in map_.items()}

    segments_samples = None

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
    dataset.mask_targets["ground_truth"] = catmap
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
        sample.add_labels(label, label_field="ground_truth")
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

        cuboids = []
        polygons = []
        polylines = []
        keypoints = []
        for instance in annotation["labels"]["ground-truth"]["attributes"][
            "annotations"
        ]:
            category_name = id_cat_map[instance["category_id"]]
            type_ = instance["type"]

            if type_ == "cuboid":
                cuboid = create_51_cuboid(instance, category_name)
                cuboids.append(cuboid)
            elif type_ == "polygon":
                polygon = create_51_3dpolygon(
                    instance, category_name, is_polygon=True
                )
                polygons.append(polygon)
            elif type_ == "polyline":
                polyline = create_51_3dpolygon(
                    instance, category_name, is_polygon=False
                )
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


def upload_dataset(client: SegmentsClient, dataset: fo.Dataset, dataset_id: str, ctx):
    for idx, s in enumerate(dataset):
        ctx.set_progress(
            (idx + 1) / len(dataset), label=f"Uploading {idx+1}/(len(dataset))"
        )

        # If the sample is stored in a cloud bucket, don't upload it to segments.ai. Instead, use the URL directly.
        if is_cloud_storage(s.filepath):
            url = s.filepath
            filename = url.rsplit("/", 1)[-1]
        else:
            with open(s.filepath, "rb") as f:
                asset = client.upload_asset(f, Path(s.filepath).name)
                url = asset.url
                filename = asset.filename

        if dataset.media_type == "image":
            sample_attrib = {"image": {"url": url}}
        elif dataset.media_type == "point-cloud":
            sample_attrib = {"pcd": {"url": url, "type": "pcd"}}
        else:
            # TODO: add support for media type '3d'
            raise ValueError(
                f"Dataset upload not implemented for media type: {dataset.media_type}"
            )

        segments_sample = client.add_sample(
            dataset_id, filename, attributes=sample_attrib
        )
        s["segments_uuid"] = segments_sample.uuid
        s.save()


def task_type_matches(media_type: str, seg_task_type: segments.typing.TaskType) -> bool:
    if media_type == "image":
        return seg_task_type in (
            segments.typing.TaskType.SEGMENTATION_BITMAP,
            segments.typing.TaskType.SEGMENTATION_BITMAP_HIGHRES,
            segments.typing.TaskType.IMAGE_SEGMENTATION_SEQUENCE,
            segments.typing.TaskType.BBOXES,
            segments.typing.TaskType.VECTOR,
            segments.typing.TaskType.IMAGE_VECTOR_SEQUENCE,
            segments.typing.TaskType.KEYPOINTS,
        )

    elif media_type == "point-cloud" or media_type == "3d":
        return seg_task_type in (
            segments.typing.TaskType.POINTCLOUD_CUBOID,
            segments.typing.TaskType.POINTCLOUD_SEGMENTATION,
            segments.typing.TaskType.POINTCLOUD_VECTOR,
        )
    else:
        raise ValueError(f"Not implemented for media type: {media_type}")


def _no_dset_selected_warning(inputs):
    warning = types.Warning(
        label="No segments.ai dataset selected. Please select one using the select_segments_dataset operator."
    )
    prop = inputs.view("warning", warning)
    prop.invalid = True
    return types.Property(inputs, view=types.View(label="No dataset selected"))


def _fetch_selected_dataset_name(ctx) -> Optional[str]:
    try:
        run_result = ctx.dataset.load_run_results(SEGMENTS_METADATA_KEY, cache=False)
        return run_result.dataset_full_name
    except ValueError:
        return None


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
    if is_polygon:
        points = np.concatenate((points, points[0:1, :]), axis=0)

    line = fo.Polyline(label=category_name, points3d=[points.tolist()])
    return line


def register(p):
    p.register(RequestAnnotations)
    p.register(FetchAnnotations)
    p.register(AddIssue)
    p.register(SelectDataset)
