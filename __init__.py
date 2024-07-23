"""
Annotation operators.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from pathlib import Path

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import numpy as np
from segments import SegmentsClient, SegmentsDataset


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

    def execute(self, ctx):
        attributes = {"format_version": "0.1", "categories": []}
        for idx, cls in enumerate(ctx.params["classes"]):
            attributes["categories"].append({"id": idx + 1, "name": cls})

        client = get_client(ctx)
        dataset = client.add_dataset(
            ctx.params["dataset_name"],
            description="Created by the segments fiftyone plugin.",
            task_type="segmentation-bitmap",
            task_attributes=attributes,
        )

        upload_dataset(client, ctx.dataset, dataset.full_name)

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.str("dataset_name", label="Dataset Name")

        inputs.list(
            "classes",
            types.String(),
            label="Classes",
            description="The annotation labels",
        )

        return types.Property(inputs)

    def resolve_output(self, ctx):
        # TODO: Add link to new dataset on segments.ai
        outputs = types.Object()
        view = types.View(label="New dataset created")
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
        for annotation in dataloader:
            if annotation["segmentation_bitmap"] is None:
                # No annotation, skip
                continue

            name = annotation["name"]
            sample = fn_sample_map[name]
            segmap = np.asarray(annotation["segmentation_bitmap"])
            label = fo.Segmentation(mask=segmap)
            sample.add_labels(label, label_field="ground_truth")
            sample.save()

        ctx.ops.reload_dataset()


def get_client(ctx) -> SegmentsClient:
    client = SegmentsClient(
        ctx.secrets.get("SEGMENTS_API_KEY"), api_url=ctx.secrets.get("SEGMENTS_URL")
    )
    return client


def upload_dataset(client: SegmentsClient, dataset: fo.Dataset, dataset_id: str):
    for s in dataset:
        with open(s.filepath, "rb") as f:
            asset = client.upload_asset(f, Path(s.filepath).name)
            sample_attrib = {"image": {"url": asset.url}}
            client.add_sample(dataset_id, asset.filename, attributes=sample_attrib)


def register(p):
    p.register(RequestAnnotations)
    p.register(FetchAnnotations)
