import networkx as nx
from enum import Enum
from scipy import stats


class Distances(Enum):
    GED = 'GED'  # Networkx Graph Edit Distance
    GED_OPT = 'GED_OPT'  # Networkx Optimize Graph Edit Distance - approximation
    DEGREE_EMD = 'DEGREE_EMD'  # Degree Earth Mover Distance


def degree_earthmover_distance(u, v):
    """Compute the earthmover distance between the sorted degrees of two graphs."""
    u_degrees = sorted([d for n, d in u.out_degree()])
    v_degrees = sorted([d for n, d in v.out_degree()])
    return stats.wasserstein_distance(u_degrees, v_degrees)


# Helpers
def get_nth_or_last(iterator, n):
    curr = prev = next(iterator, None)
    for _ in range(n):
        prev = curr
        curr = next(iterator, None)
        if curr is None:
            return prev
    return curr


def get_similatiry_measure(distance, **kwargs):
    if distance == Distances.GED:
        return nx.graph_edit_distance
    elif distance == Distances.GED_OPT:
        opt = kwargs['opt'] if 'opt' in kwargs else 1
        return lambda u, v: get_nth_or_last(
            nx.optimize_graph_edit_distance(u, v), opt)
    elif distance == Distances.DEGREE_EMD:
        return degree_earthmover_distance
    else:
        raise ValueError('Unknown distance measure: {}'.format(distance))
