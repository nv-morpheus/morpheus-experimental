def _to_dict(x):
    if isinstance(x, dict):
        return x
    else:
        return {idx: count for (idx, count) in enumerate(x) if count != 0}


def _match_vec_inputs(x, y):

    if type(x) == type(y):
        return x, y
    elif isinstance(x, dict) or isinstance(y, dict):
        d, coll = (x, y) if isinstance(x, dict) else (y, x)
        return d, _to_dict(coll)
    else:
        assert len(x) == len(y), "Non-set, non-dict list-like inputs must have the same length"
        return x, y
        