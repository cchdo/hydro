from collections import ChainMap
from itertools import chain
from typing import Hashable, Tuple

from .containers import Exchange


def ordered_merge(*tuples: Tuple[Hashable, ...]):
    return tuple(dict.fromkeys(chain(*tuples)).keys())


def merge_ex(*excahnges: Exchange) -> Exchange:
    return Exchange(
        file_type=excahnges[0].file_type,
        comments=excahnges[0].comments,
        parameters=ordered_merge(*(ex.parameters for ex in excahnges)),
        keys=ordered_merge(*(ex.keys for ex in excahnges)),
        flags=ordered_merge(*(ex.flags for ex in excahnges)),
        errors=ordered_merge(*(ex.errors for ex in excahnges)),
        coordinates=dict(ChainMap(*(ex.coordinates for ex in reversed(excahnges)))),
        data=dict(ChainMap(*(ex.data for ex in reversed(excahnges)))),
    )
