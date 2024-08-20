from dataclasses import dataclass
import codecs
import json
import base64
from pathlib import Path

import numpy as np
import fiftyone as fo
import fiftyone.operators.types as types
import segments


@dataclass
class Point3D:
    x: float
    y: float
    z: float

    def array(self):
        return np.array((self.x, self.y, self.z))


def in_cache(ctx, cache_key: str) -> bool:
    return cache_key in ctx.params


def fetch_cache(ctx, cache_key: str) -> dict:
    cache_data = ctx.params[cache_key]
    decoded = base64.decodebytes(cache_data.encode("utf-8"))
    decoded = codecs.decode(decoded, encoding="zlib_codec").decode("utf-8")
    decoded = json.loads(decoded)

    return decoded


def add_cache(inputs, cache_key: str, cache_data: list[dict]):
    encoded = json.dumps(cache_data)
    encoded = codecs.encode(encoded.encode("utf-8"), encoding="zlib_codec")
    encoded = base64.encodebytes(encoded).decode("utf-8")

    inputs.str(cache_key, default=encoded, view=types.HiddenView())


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
