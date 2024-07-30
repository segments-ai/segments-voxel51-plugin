from dataclasses import dataclass
import codecs
import json
import base64

import numpy as np
import fiftyone.operators.types as types


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