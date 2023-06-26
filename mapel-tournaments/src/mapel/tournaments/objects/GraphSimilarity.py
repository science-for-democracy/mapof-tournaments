import marshal
import pickle
import types
from enum import Enum
from functools import partial
from mapel.core.matchings import solve_matching_vectors
from mapel.core.inner_distances import emd
import gurobipy as gp
import numpy as np

from scipy import stats

import networkx as nx
import tqdm


class Distances(Enum):
    GED = 'GED'  # Networkx Graph Edit Distance
    GED_OPT = 'GED_OPT'  # Networkx Optimize Graph Edit Distance - approximation
    GED_BLP = 'GED_BLP'  # Graph Edit Distance - BLP
    DEGREE_EMD = 'DEGREE_EMD'  # Degree Earth Mover Distance
    SP = 'SP'  # Shortest Paths


def degree_earthmover_distance(u, v):
    """Compute the earthmover distance between the sorted degrees of two graphs."""
    u_degrees = sorted([d for n, d in u.graph.out_degree()])
    v_degrees = sorted([d for n, d in v.graph.out_degree()])
    return emd(u_degrees, v_degrees)


def shortestpathswise_distance(u, v, inner_distance=emd):
    cost_table = get_matching_cost_positionwise(u, v, inner_distance)
    return solve_matching_vectors(cost_table)[0]


# Helpers
def get_nth_or_last(iterator, n):
    curr = prev = next(iterator, None)
    for _ in range(n):
        prev = curr
        curr = next(iterator, None)
        if curr is None:
            return prev
    return curr


def parallel_runner(args):
    return args[0](*args[1:])


def get_similarity_measure(distance, **kwargs):
    if distance == Distances.GED:
        return nx.graph_edit_distance
    elif distance == Distances.GED_OPT:
        opt = kwargs['opt'] if 'opt' in kwargs else 1
        return lambda u, v: get_nth_or_last(nx.optimize_graph_edit_distance(u, v), opt)
    elif distance == Distances.DEGREE_EMD:
        return degree_earthmover_distance
    elif distance == Distances.SP:
        return shortestpathswise_distance
    elif distance == Distances.GED_BLP:
        return ged_blp_wrapper
    else:
        raise ValueError('Unknown distance measure: {}'.format(distance))


def get_matching_cost_positionwise(u, v, inner_distance):
    sp1 = u.to_shortest_paths_matrix()
    sp1.sort(axis=1)
    # sp1 /= sp1.sum(axis=1, keepdims=True)
    sp2 = v.to_shortest_paths_matrix()
    sp2.sort(axis=1)
    # sp2 /= sp2.sum(axis=1, keepdims=True)
    n = len(sp1)
    return [[inner_distance(sp1[i], sp2[j]) for i in range(n)] for j in range(n)]


def ged_blp_wrapper(u, v):
    return ged_blp(u.graph, v.graph)


def ged_blp(u, v):
    """Compute the graph edit distance between two graphs using a binary linear program. """
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    n = len(u.nodes())
    m = gp.Model('ged_blp', env=env)
    # Vertex matching variables
    x = np.ndarray(shape=(n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            x[i, j] = m.addVar(vtype=gp.GRB.BINARY, name='x_{},{}'.format(i, j))

    # Edge matching variables
    y = np.ndarray(shape=(n, n, n, n), dtype=object)
    for i, j in u.edges():
        for k, l in v.edges():
            y[i, j, k, l] = m.addVar(vtype=gp.GRB.BINARY, name='y_{},{},{},{}'.format(i, j, k, l))

    # Edge reversed variables
    r = np.ndarray(shape=(n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            r[i, j] = m.addVar(vtype=gp.GRB.BINARY, name='r_{},{}'.format(i, j))

    # Objective
    m.setObjective(gp.quicksum(r[i, j] for i in range(n) for j in range(n)), gp.GRB.MINIMIZE)

    # Constraints
    # Vertex matching constraints
    for i in range(n):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)
        m.addConstr(gp.quicksum(x[j, i] for j in range(n)) == 1)

    # Edge matching constraints
    for i, j in u.edges():
        m.addConstr(gp.quicksum(y[i, j, k, l] for (k, l) in v.edges) == 1)
    for k, l in v.edges():
        m.addConstr(gp.quicksum(y[i, j, k, l] for (i, j) in u.edges()) == 1)

    # Topological constraints
    for i, j in u.edges():
        for k, l in v.edges():
            m.addConstr(y[i, j, k, l] <= x[i, k] + x[i, l] * r[i, j])
            m.addConstr(y[i, j, k, l] <= x[j, l] + x[j, k] * r[i, j])

    # Solve
    m.optimize()

    # Print all variables
    # for v in m.getVars():
    #     print('%s %g' % (v.varName, v.x))

    # # Get matching
    # matching = []
    # for i in range(n):
    #     for j in range(n):
    #         if x[i, j].x > 0.5:
    #             matching.append((i, j))
    # return m.objVal, matching
    return m.objVal


if __name__ == '__main__':
    # Test ged_blp
    from mapel.tournaments.objects.TournamentExperiment import TournamentInstance
    t1 = nx.from_numpy_array(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    t2 = nx.from_numpy_array(np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]))
    print(ged_blp(t1, t2))
