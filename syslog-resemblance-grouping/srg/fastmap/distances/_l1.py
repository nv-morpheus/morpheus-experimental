from srg.fastmap.distances._distance import Distance, InputError
from pandas.api.types import is_list_like

from srg.fastmap.distances._helpers import _match_vec_inputs


def _d(x, y):
    if isinstance(x, dict):
        all_keys = {*x}.union({*y})
        diffs = [x.get(key, 0)-y.get(key, 0) for key in all_keys]
    else:
        diffs = [xi - yi for (xi, yi) in zip(x, y)]
    return sum([abs(diff) for diff in diffs])


class L1(Distance):

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "L1"

    def calculate(self, x, y) -> float:

        if not (is_list_like(x, allow_sets=False) and is_list_like(y, allow_sets=False)):
            raise InputError("L1 distance needs to be non-set, list like objects")

        x, y = _match_vec_inputs(x, y)
        d = _d(x, y)

        return d
