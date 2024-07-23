"""
Annotation operators.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import enum
import urllib.parse
from pathlib import Path

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import numpy as np
from segments import SegmentsClient, SegmentsDataset

SEGMENTS_FRONTEND_URL = "https://segments.ai"


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


class RequestAnnotations(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="request_annotations",
            label="Request Segments.ai annotations",
            light_icon="/assets/Black icon.svg",
            dark_icon="/assets/White icon.svg",
            dynamic=False,
        )

    @staticmethod
    def target_data_selector(ctx, inputs):
        has_selected = bool(ctx.selected)
        has_view = ctx.view != ctx.dataset.view()
        target_choices = types.RadioGroup(orientation="horizontal")
        target_choices.add_choice(
            TargetSelection.DATASET.value,
            label="Entire dataset",
            description="Upload the entire dataset",
        )
        if has_selected:
            target_choices.add_choice(
                TargetSelection.SELECTED.value,
                label="Selected samples",
                description="Upload only the selected samples.",
            )
        if has_view:
            target_choices.add_choice(
                TargetSelection.CURRENT_VIEW.value,
                label="Current view",
                description="Upload only the current view",
            )
        inputs.enum(
            "target",
            target_choices.values(),
            required=True,
            label="Target",
            view=target_choices,
            default="full_dataset",
        )

    @staticmethod
    def dataset_type_selector(ctx, inputs):
        labelmap = {
            SegmentsDatasetType.SEGMENTATION_BITMAP: "Segmentation bitmap",
            SegmentsDatasetType.SEGMENTATION_BITMAP_HIGHRES: "Segmentation bitmap highres",
            SegmentsDatasetType.BBOXES: "Bounding boxes",
            SegmentsDatasetType.VECTOR: "Vector",
            SegmentsDatasetType.KEYPOINTS: "Keypoints",
        }
        data_choices = types.Dropdown()
        for datatype in SegmentsDatasetType:
            data_choices.add_choice(datatype.value, label=labelmap[datatype])

        inputs.enum(
            "dataset_type",
            data_choices.values(),
            label="Dataset type",
            default=SegmentsDatasetType.SEGMENTATION_BITMAP.value,
            view=data_choices,
        )

    def execute(self, ctx):
        attributes = {"format_version": "0.1", "categories": []}
        for idx, cls in enumerate(ctx.params["classes"]):
            attributes["categories"].append({"id": idx + 1, "name": cls})

        task_type = ctx.params["dataset_type"]

        client = get_client(ctx)
        dataset = client.add_dataset(
            ctx.params["dataset_name"],
            description="Created by the segments fiftyone plugin.",
            task_type=task_type,
            task_attributes=attributes,
        )

        dataset_view = self.target_dataset_view(ctx)
        url = urllib.parse.urljoin(SEGMENTS_FRONTEND_URL, dataset.full_name)
        upload_dataset(client, dataset_view, dataset.full_name, ctx)

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

        self.target_data_selector(ctx, inputs)
        inputs.str("dataset_name", label="Dataset Name")
        self.dataset_type_selector(ctx, inputs)

        inputs.list(
            "classes",
            types.String(),
            label="Classes",
            description="The annotation labels",
        )

        return types.Property(inputs)

    def resolve_output(self, ctx):
        outputs = types.Object()
        view = types.View(label="New dataset created")
        outputs.str("url", label="Dataset URL")
        return types.Property(outputs, view=view)


class FetchAnnotations(foo.Operator):
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
        choices_dataset = types.Choices()
        client = get_client(ctx)

        for dataset in client.get_datasets():
            choices_dataset.add_choice(dataset.full_name, label=dataset.name)

        inputs.enum(
            "dataset",
            choices_dataset.values(),
            view=choices_dataset,
            label="Dataset",
            required=True,
        )

        if (dataset := ctx.params.get("dataset", None)) is not None:
            releases = client.get_releases(dataset)
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
        client = get_client(ctx)

        dataset_sdk = client.get_dataset(ctx.params["dataset"])
        release = client.get_release(dataset_sdk.full_name, ctx.params["release"])
        dataloader = SegmentsDataset(release, preload=False)

        # Ugly hack to make sure SegmentsDataset does not fetch all image files.
        for sample in dataloader.samples:
            image_info = sample["attributes"]["image"]
            image_info["url"] = image_info["url"].replace("https", "s3")

        fn_sample_map = {Path(s.filepath).name: s for s in ctx.dataset}
        dataset_type = SegmentsDatasetType(dataset_sdk.task_type)
        if dataset_type in (
            SegmentsDatasetType.SEGMENTATION_BITMAP,
            SegmentsDatasetType.SEGMENTATION_BITMAP_HIGHRES,
        ):
            insert_segmentation_labels(dataloader, ctx.dataset, fn_sample_map)
        else:
            raise ValueError(f"Dataset type '{dataset_type.value}' not yet supported")

        ctx.ops.reload_dataset()

    def resolve_output(self, ctx):
        outputs = types.Object()
        view = types.View(label="Succesfully pulled annotations")
        return types.Property(outputs, view=view)


def insert_segmentation_labels(
    dataloader: SegmentsDataset, dataset: fo.Dataset, sample_map: dict[str, fo.Sample]
):
    catmap = {x.id: x.name for x in dataloader.categories}
    dataset.mask_targets["ground_truth"] = catmap
    dataset.save()
    annotation_count = 0
    for annotation in dataloader:
        if annotation["segmentation_bitmap"] is None:
            # No annotation, skip
            continue

        name = annotation["name"]
        sample = sample_map[name]
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


def get_client(ctx) -> SegmentsClient:
    api_key = ctx.secrets.get("SEGMENTS_API_KEY")
    if (segments_url := ctx.secrets.get("SEGMENTS_URL", None)) is not None:
        client = SegmentsClient(api_key, api_url=segments_url)
    else:
        client = SegmentsClient(api_key)
    return client


def upload_dataset(client: SegmentsClient, dataset: fo.Dataset, dataset_id: str, ctx):
    for idx, s in enumerate(dataset):
        ctx.set_progress(
            (idx + 1) / len(dataset), label=f"Uploading {idx+1}/(len(dataset))"
        )
        with open(s.filepath, "rb") as f:
            asset = client.upload_asset(f, Path(s.filepath).name)
            sample_attrib = {"image": {"url": asset.url}}
            client.add_sample(dataset_id, asset.filename, attributes=sample_attrib)


def register(p):
    p.register(RequestAnnotations)
    p.register(FetchAnnotations)
