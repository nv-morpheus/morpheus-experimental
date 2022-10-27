from srg.fastmap.distances._distance import Distance, InputError


def _match_inputs(x, y):
    if type(x) == type(y):
        return x, y
    elif isinstance(x, str) or isinstance(y, str):
        coll, string = (x, y) if isinstance(y, str) else (y, x)
        return coll, [ch for ch in string]
    else:
        return x, y


class Lev(Distance):

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return 'Levenshtein'

    def calculate(self, x, y) -> float:
        s1, s2 = _match_inputs(x, y)

        s1_len = len(s1)
        s2_len = len(s2)
        if s1_len == 0 or s2_len == 0:
            return max(s1_len, s2_len)
        if s1_len > s2_len:
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for index2, char2 in enumerate(s2):
            new_distances = [index2 + 1]
            for index1, char1 in enumerate(s1):
                if char1 == char2:
                    new_distances.append(distances[index1])
                else:
                    new_distances.append(1 + min((distances[index1],
                                                  distances[index1 + 1],
                                                  new_distances[-1])))
            distances = new_distances
        return distances[-1]
        