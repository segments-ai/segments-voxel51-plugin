from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import fiftyone as fo
import numpy as np
import segments


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
