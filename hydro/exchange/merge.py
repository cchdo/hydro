from itertools import chain
from typing import Hashable, Tuple

from .containers import Exchange


def ordered_merge(*tuples: Tuple[Hashable, ...]):
    return tuple(dict.fromkeys(chain(*tuples)).keys())


def merge_ex(*excahnges: Exchange) -> Exchange:
    coordinates2 = dict(coord for ex in excahnges for coord in ex.coordinates.items())
    data2 = dict(data for ex in excahnges for data in ex.data.items())
    return Exchange(
        file_type=excahnges[0].file_type,
        comments=excahnges[0].comments,
        parameters=ordered_merge(*(ex.parameters for ex in excahnges)),
        keys=ordered_merge(*(ex.keys for ex in excahnges)),
        flags=ordered_merge(*(ex.flags for ex in excahnges)),
        errors=ordered_merge(*(ex.errors for ex in excahnges)),
        coordinates=coordinates2,
        data=data2,
    )
