from itertools import chain
from typing import Hashable, Tuple

from .containers import Exchange
from .exceptions import ExchangeInconsistentMergeType


def ordered_merge(*tuples: Tuple[Hashable, ...]):
    return tuple(dict.fromkeys(chain(*tuples)).keys())


def merge_ex(*exchanges: Exchange) -> Exchange:

    if len({ex.file_type for ex in exchanges}) != 1:
        raise ExchangeInconsistentMergeType

    comments = "\n--FileBreak--\n".join({ex.comments for ex in exchanges})

    coordinates2 = dict(coord for ex in exchanges for coord in ex.coordinates.items())
    data2 = dict(data for ex in exchanges for data in ex.data.items())
    return Exchange(
        file_type=exchanges[0].file_type,
        comments=comments,
        parameters=ordered_merge(*(ex.parameters for ex in exchanges)),
        keys=ordered_merge(*(ex.keys for ex in exchanges)),
        flags=ordered_merge(*(ex.flags for ex in exchanges)),
        errors=ordered_merge(*(ex.errors for ex in exchanges)),
        coordinates=coordinates2,
        data=data2,
    )
