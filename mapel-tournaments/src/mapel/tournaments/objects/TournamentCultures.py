"""This module contains functions to generate tournaments from different cultures.

>>> print(list(registered_cultures.keys()), sep="\\n")
['ordered', 'rock-paper-scissor', 'uniform', 'condorcet_noise', 'mallows', 'urn', 'diffused', 'nauty']
"""
import itertools
import subprocess
from random import uniform
from subprocess import check_output

import networkx as nx
import numpy as np

registered_cultures = {}
aliases = {}


def get(culture_id: str):
    """Return the culture function for the given culture id."""
    culture_id = culture_id.lower()
    if culture_id in aliases:
        return registered_cultures[aliases[culture_id]]
    else:
        raise ValueError(f"No such culture id: {culture_id}")


def exists(culture_id: str):
    """Return true if the culture exists."""
    culture_id = culture_id.lower()
    return culture_id in aliases


def register(name: str | list[str]):

    def decorator(func):
        if isinstance(name, str):
            names = [name]
        else:
            names = name
        if isinstance(names, list):
            map(lambda n: n.lower(), names)
            registered_cultures[names[0]] = func
            for n in names:
                aliases[n] = names[0]
        else:
            raise ValueError("name must be a string or a list of strings")
        return func

    return decorator


## Compass tournaments
@register(["ordered", "condorcet"])
def ordered(num_participants, _count, _params):
    adjacency_matrix = np.zeros((num_participants, num_participants))
    for i in range(num_participants):
        for j in range(i + 1, num_participants):
            adjacency_matrix[i, j] = 1
    return [nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)]


@register(["rock-paper-scissors", "rps", "chaos"])
def rock_paper_scissors(num_participants, _count, _params):
    adjacency_matrix = np.zeros((num_participants, num_participants))
    for jump_length in range(1, num_participants // 2 + 1):
        # For the last iteration with even number of participants, we only set half of the edges.
        for i in range(num_participants if jump_length < (num_participants + 1) // 2 else num_participants // 2):
            j = (i + jump_length) % num_participants
            adjacency_matrix[i, j] = 1
    return [nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)]


@register("all-rps")
def all_rps(num_participants, max_count, _params):
    return nauty(num_participants, max_count, {'opt': f'-d{(num_participants-1)//2}D{num_participants//2}'})


@register(["o2rps"])
def o2rps(num_participants, _count, _params):
    tournaments = []
    for i, n in enumerate(range(1, num_participants - 2)):
        g1 = ordered(n, 1, {})[0]
        g2 = rock_paper_scissors(num_participants - n, 1, {})[0]
        g2 = nx.relabel_nodes(g2, {i: i + n for i in g2.nodes})

        g = nx.compose(g1, g2)
        for i in g1.nodes:
            for j in g2.nodes:
                g.add_edge(i, j)
        tournaments.append(g)
    return tournaments


@register(["rps2o"])
def rps2o(num_participants, _count, _params):
    tournaments = []
    for i, n in enumerate(range(1, num_participants - 2)):
        g1 = ordered(n, 1, {})[0]
        g2 = rock_paper_scissors(num_participants - n, 1, {})[0]
        g2 = nx.relabel_nodes(g2, {i: i + n for i in g2.nodes})

        g = nx.compose(g1, g2)
        for i in g1.nodes:
            for j in g2.nodes:
                g.add_edge(j, i)
        tournaments.append(g)
    return tournaments


### Statistical


# @register(["uniform", "random"])
def uniform_random(num_participants, count, _params):
    # superseded by uniform_weighted!!!
    """Choose each edge direction with probability 0.5"""
    tournaments = []
    for i in range(count):
        adjacency_matrix = np.zeros((num_participants, num_participants))
        for i in range(num_participants):
            for j in range(i + 1, num_participants):
                if np.random.rand() < 0.5:
                    adjacency_matrix[i, j] = 1
                else:
                    adjacency_matrix[j, i] = 1
        tournaments.append(nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph))
    return tournaments


def _noise(graph, count, params):
    """Create a family of tournaments by randomly reversaling edges. The
  probability of reversaling an edge is p."""
    if 'p' not in params:
        raise ValueError("p must be specified for the 'condorcet_noise' culture inside 'params'")
    p = params['p']
    tournaments = []
    if isinstance(p, float):
        p = [p] * count
    elif not isinstance(p, list):
        raise ValueError("p must be a float or a list of floats")
    for i in range(count):
        g = graph.copy()
        for e in list(g.edges()):
            if uniform(0, 1) < p[i]:
                g.remove_edge(*e)
                g.add_edge(*reversed(e))
        tournaments.append(g)
    return tournaments


# @register(["ordered_noise", "condorcet_noise"])
# def condorcet_noise(num_participants, count, params):
#   """Start with an initial ordered tournament and reversal each edge with probability p"""
#   if 'p' not in params:
#     raise ValueError(
#         "p must be specified for the 'condorcet_noise' culture inside 'params'")
#   p = params['p']
#   if isinstance(p, float):
#     p = [p]
#   elif not isinstance(p, list):
#     raise ValueError("p must be a float or a list of floats")
#   tournaments = []
#   for c in range(count):
#     adjacency_matrix = np.zeros((num_participants, num_participants))
#     for i in range(num_participants):
#       for j in range(i + 1, num_participants):
#         if np.random.rand() < p[c]:
#           adjacency_matrix[j, i] = 1
#         else:
#           adjacency_matrix[i, j] = 1
#     tournaments.append(nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph))
#   return tournaments


@register(["ordered_noise", "condorcet_noise"])
def condorcet_noise(num_participants, count, params):
    """Start with an initial ordered tournament and reverse each edge with probability p"""
    graph = ordered(num_participants, 1, {})[0]
    return _noise(graph, count, params)


def _reversal(graph, count, params):
    """Create a family of tournaments by reversing exactly params['reversals'] edges."""
    if 'reversals' not in params:
        raise ValueError("reversals must be specified for the 'condorcet_reversal' culture inside 'params'")
    reversals = params['reversals']
    tournaments = []
    if isinstance(reversals, int):
        reversals = [reversals] * count
    elif not isinstance(reversals, list):
        raise ValueError("reversals must be an int or a list of ints")
    for i in range(count):
        g = graph.copy()
        edges = list(g.edges())
        np.random.shuffle(edges)
        for e in edges[:reversals[i]]:
            g.remove_edge(*e)
            g.add_edge(*reversed(e))
        tournaments.append(g)
    return tournaments


@register(["ordered_reversal", "condorcet_reversal"])
def condorcet_reversal(num_participants, count, params):
    """Start with an initial ordered tournament and reverse exactly params['reversals'] edges."""
    graph = ordered(num_participants, 1, {})[0]
    return _reversal(graph, count, params)


def _weighted(weights, count):
    """Create a family of tournaments from a given weight distribution."""
    tournaments = []
    n = len(weights)
    for c in range(count):
        adjacency_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < weights[i] / (weights[i] + weights[j]):
                    adjacency_matrix[i, j] = 1
                else:
                    adjacency_matrix[j, i] = 1
        tournaments.append(nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph))
    return tournaments


@register("pow2_weighted")
def pow2_weighted(num_participants, count, params):
    """Create a family of tournaments with a weight distribution of 2^i."""
    weights = [2**i for i in range(num_participants)]
    return _weighted(weights, count)


@register("rand_pow_weighted")
def rand_pow_weighted(num_participants, count, params):
    """Create a family of tournaments with a weight distribution of x^i, where x is a random number from [1, 2)."""
    res = []
    for i in range(count):
        r = np.random.rand() + 1
        weights = [r**i for i in range(num_participants)]
        res.append(_weighted(weights, 1)[0])
    return res


@register(["exponential", "exp_weighted"])
def exp_weighted(num_participants, count, params):
    """Create a family of tournaments with a weight distribution of e^i."""
    weights = [np.exp(i) for i in range(num_participants)]
    return _weighted(weights, count)


@register(["linear", "lin_weighted"])
def lin_weighted(num_participants, count, params):
    """Create a family of tournaments with a weight distribution of i."""
    weights = [i for i in range(num_participants)]
    return _weighted(weights, count)


@register(["logarithmic", "log_weighted"])
def log_weighted(num_participants, count, params):
    """Create a family of tournaments with a weight distribution of log(i)."""
    weights = [np.log(i) for i in range(1, num_participants + 1)]
    return _weighted(weights, count)


@register(["sqrt", "sqrt_weighted"])
def sqrt_weighted(num_participants, count, params):
    """Create a family of tournaments with a weight distribution of sqrt(i)."""
    weights = [np.sqrt(i) for i in range(1, num_participants + 1)]
    return _weighted(weights, count)


@register(["uniform", "random", "uniform_weighted"])
def uniform_weighted(num_participants, count, params):
    """Create a family of tournaments with a weight distribution of 1."""
    weights = [1 for i in range(num_participants)]
    return _weighted(weights, count)


### From elections
import mapel.elections.objects.OrdinalElection as oe


def from_ordinal_election(culture_id, num_participants, count, params):
    """Create tournaments from a registered ordinal election culture."""

    def single(election):
        """Create a tournament from an ordinal election."""
        n = election.num_candidates
        election.prepare_instance()
        adjacency_matrix = np.zeros((n, n))
        pairwise_matrix = election.votes_to_pairwise_matrix()
        for i in range(n):
            for j in range(i + 1, n):
                if pairwise_matrix[i, j] > pairwise_matrix[j, i]:
                    adjacency_matrix[i, j] = 1
                else:
                    adjacency_matrix[j, i] = 1
        return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)

    if 'num_voters' not in params:
        raise ValueError("num_voters must be specified for the 'ordinal_election' culture inside 'params'")
    num_voters = int(params['num_voters'])
    params.pop('num_voters')
    tournaments = []
    for i in range(count):
        election = oe.OrdinalElection(culture_id=culture_id,
                                      num_voters=num_voters,
                                      num_candidates=num_participants)
        tournaments.append(single(election))
    return tournaments


# @register("mallows")
# def mallows_ordinal(num_participants, count, params):
#   return from_ordinal_election('mallows', num_participants, count, params)

# @register("urn")
# def urn_ordinal(num_participants, count, params):
#   return from_ordinal_election('urn', num_participants, count, params)


### Special
@register("diffused")
def diffused(tournament, count, alpha=None, ev=None):
    """Create a family of tournaments by randomly reversaling edges. The
  probability of reversaling an edge is alpha. If ev is specified, alpha is
  computed as ev / |E|."""
    if ev is not None:
        alpha = ev / len(tournament.graph.edges())
    if alpha is None:
        raise ValueError("Either alpha or ev must be specified for the 'diffused' culture")
    tournaments = {}
    for i in range(count):
        g = tournament.graph.copy()
        c = 0
        for e in g.edges():
            if uniform(0, 1) < alpha:
                c += 1
                g.remove_edge(*e)
                g.add_edge(*reversed(e))
        tournaments[f"diffused_a={alpha}_c={c}_{i}"] = g
    return tournaments


tournament_count_lookup = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 12,
    6: 56,
    7: 456,
    8: 6880,
    9: 191536,
    10: 9733056,
    11: 903753248,
    12: 154108311168,
    13: 48542114686912,
    14: 28401423719122304,
    15: 31021002160355166848,
    16: 63530415842308265100288,
    17: 244912778438520759443245824,
}


def nauty_decode(output, n, size, seed=0):
    graphs = []
    lines = output.decode('utf-8').split('\n')[:-1]
    if len(lines) > size:
        np.random.seed(seed)
        lines = np.random.choice(lines, size, replace=False)
    for line in lines:
        arr = np.zeros((n, n), dtype=np.int8)
        ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                if line[ind] == '1':
                    arr[i, j] = 1
                elif line[ind] == '0':
                    arr[j, i] = 1
                ind += 1
        graphs.append(nx.from_numpy_array(arr, create_using=nx.DiGraph))
    return graphs


@register("nauty")
def nauty(n, size, params):
    """Generate size random tournaments with n participants using nauty."""
    args = ["nauty-gentourng", str(n)]
    if 'opt' in params:
        args = ["nauty-gentourng", params['opt'], str(n)]
    elif 'resmod' in params:
        args = ["nauty-gentourng", str(n), params['resmod']]
    elif size < tournament_count_lookup[n]:
        try:
            args.append(f'0/{tournament_count_lookup[n]*2/size}')
        except KeyError:
            raise ValueError(f"n={n} is too big, please supply res/mod manually")
    else:
        args = ["nauty-gentourng", str(n)]
    if 'quiet' in params and params['quiet']:
        args.append('-q')
    proc = subprocess.run(args, stdout=subprocess.PIPE, bufsize=10)
    return nauty_decode(proc.stdout, n, size)


@register("nauty-simple")
def nauty_simple(n, size, _params):
    time_limit = 5
    mod = 1
    args = ["nauty-gentourng", "-u", str(n), f"0/{mod}"]
    while True:
        try:
            output = subprocess.check_output(args, timeout=time_limit, bufsize=0)
            break
        except subprocess.TimeoutExpired:
            mod *= 4
            args = ["nauty-gentourng", str(n), f"0/{mod}"]

    return nauty(n, size, {'resmod': f"0/{mod}"})


import mapel.tournaments.objects.generate_profiles as pl
#### TODO: Code below is taken from https://github.com/uschmidtk/MoV/blob/master/experiments.py
import pandas as pd


def condorcet_tournament(n, m, p):
    all_edges = []

    for i in range(n):
        for s in itertools.combinations(range(m), 2):
            s = list(s)
            if s[0] != s[1]:
                coin = np.random.rand()
                if ((s[0] < s[1]) and (coin <= p)) or ((s[1] < s[0]) and (coin > p)):
                    all_edges.append((s[0], s[1]))
                else:
                    all_edges.append((s[1], s[0]))

    edge_count = pd.Series(all_edges).value_counts()
    agg_edges = list(edge_count[edge_count > int(n / 2)].index)

    T = nx.DiGraph()
    T.add_nodes_from(range(m))
    T.add_edges_from(agg_edges)
    if nx.tournament.is_tournament(T):
        return T
    else:
        print("There has been a mistake, this is not a tournament!")


@register('mov_paper_condorcet_tournament')
def mov_paper_condorcet_tournament(n, size, _params):
    """Generate the condorcet noise model with voters as described in the MoV paper."""
    # We're using 51 voters and p=0.55, same as they did
    return [condorcet_tournament(51, n, 0.55) for _ in range(size)]


def condorcet_tournament_direct(m, p):
    all_edges = []
    for s in itertools.combinations(range(m), 2):
        s = list(s)
        if s[0] != s[1]:
            coin = np.random.rand()
            if ((s[0] < s[1]) and (coin <= p)) or ((s[1] < s[0]) and (coin > p)):
                all_edges.append((s[0], s[1]))
            else:
                all_edges.append((s[1], s[0]))
    T = nx.DiGraph()
    T.add_nodes_from(range(m))
    T.add_edges_from(all_edges)
    if nx.tournament.is_tournament(T):
        return T
    else:
        print("There has been a mistake, this is not a tournament!")


@register('mov_paper_condorcet_direct')
def mov_paper_condorcet_direct(n, size, _params):
    """Generate the condorcet noise model without voters as described in the MoV paper."""
    # We're using 51 voters and p=0.55, same as they did
    return [condorcet_tournament(51, n, 0.55) for _ in range(size)]


def impartial_culture(n, m):
    all_edges = []
    for i in range(n):
        order = list(np.random.permutation(range(m)))
        for s in itertools.combinations(range(m), 2):
            s = list(s)
            if s[0] != s[1]:
                if (order.index(s[0]) < order.index(s[1])):
                    all_edges.append((s[0], s[1]))
                else:
                    all_edges.append((s[1], s[0]))

    edge_count = pd.Series(all_edges).value_counts()
    agg_edges = list(edge_count[edge_count > int(n / 2)].index)

    T = nx.DiGraph()
    T.add_nodes_from(range(m))
    T.add_edges_from(agg_edges)
    if nx.tournament.is_tournament(T):
        return T
    else:
        print("There has been a mistake, this is not a tournament!")


@register('mov_paper_impartial_culture')
def mov_paper_impartial_culture(n, size, _params):
    """Generate the impartial culture model as described in the MoV paper."""
    # We're using 51 voters, same as they did
    return [impartial_culture(51, n) for _ in range(size)]


def mallows(n, m, phi):
    candmap = {i: i for i in range(m)}
    rankmapcounts = pl.gen_mallows(n, candmap, [1], [phi], [list(range(m))])
    all_edges = []
    for i in range(len(rankmapcounts[1])):
        for s in itertools.combinations(range(m), 2):
            if s[0] != s[1]:
                if rankmapcounts[0][i][s[0]] < rankmapcounts[0][i][s[1]]:
                    for j in range(rankmapcounts[1][i]):
                        all_edges.append((s[0], s[1]))
                else:
                    for j in range(rankmapcounts[1][i]):
                        all_edges.append((s[1], s[0]))

    edge_count = pd.Series(all_edges).value_counts()
    agg_edges = list(edge_count[edge_count > int(n / 2)].index)

    T = nx.DiGraph()
    T.add_nodes_from(range(m))
    T.add_edges_from(agg_edges)
    if nx.tournament.is_tournament(T):
        return T
    else:
        print("There has been a mistake, this is not a tournament!")


@register('mov_paper_mallows')
def mov_paper_mallows(n, size, _params):
    """Generate the mallows model as described in the MoV paper."""
    # We're using 51 voters, same as they did
    return [mallows(51, n, 0.95) for _ in range(size)]


def urn(n, m, replace):
    candmap = {i: i for i in range(m)}
    rankmapcounts = pl.gen_urn_strict(n, replace, candmap)
    #print(rankmapcounts)
    all_edges = []
    for i in range(len(rankmapcounts[1])):
        for s in itertools.combinations(range(m), 2):
            if s[0] != s[1]:
                if rankmapcounts[0][i][s[0]] < rankmapcounts[0][i][s[1]]:
                    for j in range(rankmapcounts[1][i]):
                        all_edges.append((s[0], s[1]))
                else:
                    for j in range(rankmapcounts[1][i]):
                        all_edges.append((s[1], s[0]))

    edge_count = pd.Series(all_edges).value_counts()
    agg_edges = list(edge_count[edge_count > int(n / 2)].index)
    #print(edge_count)

    T = nx.DiGraph()
    T.add_nodes_from(range(m))
    T.add_edges_from(agg_edges)
    if nx.tournament.is_tournament(T):
        return T
    else:
        print("There has been a mistake, this is not a tournament!")


@register('mov_paper_urn')
def mov_paper_urn(n, size, _params):
    """Generate the urn model as described in the MoV paper."""
    # We're using 51 voters, same as they did
    return [urn(51, n, True) for _ in range(size)]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
