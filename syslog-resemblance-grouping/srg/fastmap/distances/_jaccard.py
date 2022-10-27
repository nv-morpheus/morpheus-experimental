from srg.fastmap.distances._distance import Distance, InputError
from srg.utils import shingler


def _weight_set(x):
    return {el: 1 for el in x}


def _match_inputs(x, y, shingle_size):
    if type(x) == type(y):
        return x, y
    elif isinstance(x, str) or isinstance(y, str):
        coll, string = (x, y) if isinstance(y, str) else (y, x)
        shingled = shingler(string, shingle_size=shingle_size)
        if isinstance(coll, set):
            return shingled, coll
        else:
            return _weight_set(shingled), coll
    else:
        dictionary, coll = (x, y) if isinstance(x, dict) else (y, x)
        return _weight_set(coll), dictionary


class Jaccard(Distance):

    def __init__(self, shingle_size=4):
        self._shingle_size = shingle_size

    @property
    def shingle_size(self):
        return self._shingle_size

    def _d(self, x, y):

        _JAC_SWITCH = {
            set: self._jac_set,
            str: self._jac_str,
            dict: self._jac_dict
        }

        func = _JAC_SWITCH[type(x)]
        return func(x, y)

    @staticmethod
    def get_name():
        return 'Jaccard'

    def calculate(self, x, y) -> float:
        x, y = _match_inputs(x, y, self.shingle_size)
        d = self._d(x, y)
        return d

    def _jac_set(self, x, y):
        intersect = len(x.intersection(y))
        size1 = len(x)
        size2 = len(y)
        union = size1 + size2 - intersect
        return 1 - intersect / union

    def _jac_str(self, x, y):
        x_set = shingler(x, shingle_size=self._shingle_size)
        y_set = shingler(y, shingle_size=self._shingle_size)
        return self._jac_set(x_set, y_set)

    def _jac_dict(self, x, y):
        keyset = {*x.keys()}.union(*y.keys())
        min_maxes = [(min(x.get(key, 0), y.get(key, 0)), max(x.get(key, 0), y.get(key, 0))) for key in keyset]
        min_sum = 0
        max_sum = 0
        for (mini, maxi) in min_maxes:
            min_sum += mini
            max_sum += maxi
        return 1 - min_sum / max_sum
        