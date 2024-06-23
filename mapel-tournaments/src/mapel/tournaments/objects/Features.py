import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fuzzywuzzy import fuzz
from mapel.tournaments.objects.helpers import log2_ceil
from mapel.tournaments.objects.TournamentCultures import (ordered,
                                                          rock_paper_scissors)
from mapel.tournaments.objects.TournamentInstance import TournamentInstance
from mapel.tournaments.objects.TournamentSimilarity import (ged_blp,
                                                            get_distance)
from mapel.tournaments.objects.TournamentSolutions import (
    copeland_winners, single_elimination_can_player_win, slater_winner,
    slater_winners, top_cycle_winners)

registered_features = {}
aliases = {}


def list_features():
    return registered_features.keys()


def _get_closest_string(s, strings):
    """Return the string in strings that is closest to s."""
    return max(strings, key=lambda x: fuzz.ratio(s, x))


def get_feature(feature_id: str):
    """Return the feature function corresponding to the given feature id."""
    if feature_id in aliases:
        return registered_features[aliases[feature_id]]
    else:
        closest = _get_closest_string(feature_id, aliases.keys())
        yn = input(f"No such feature id: {feature_id}. Did you mean {closest}? (y/n) ")
        if yn == "y":
            return registered_features[aliases[closest]]
        else:
            print("Then try again...")


def register(name: str | list[str] = [], parallel=False, reps=1, skip_function_name=False):
    """Decorator for registering a feature function.
  @param name: The name of the feature. If a list is given, the first name is the canonical name and the rest are aliases.
  @param parallel: Whether the feature can be computed in parallel.
  @param skip_function_name: Whether to skip adding the function name as an alias.
  """

    def decorator(func):
        if isinstance(name, str):
            names = [name]

        else:
            names = name
        if isinstance(names, list):
            if not skip_function_name:
                names.append(func.__name__)
            map(lambda n: n.lower(), names)
            registered_features[names[0]] = func
            for n in names:
                aliases[n] = names[0]
            if parallel:
                registered_features[names[0] + "_parallel"] = func
                for n in names:
                    aliases[n + "_parallel"] = names[0] + "_parallel"
            func.reps = reps
        else:
            raise ValueError("name must be a string or a list of strings")
        return func

    return decorator


@register("distance_from_ordered")
def distance_from_ordered(tournament: TournamentInstance, experiment):
    if tournament.instance_id == 'ordered_0':
        return 0
    return experiment.distances[tournament.instance_id]['ordered_0']


@register("local_transitivity")
def local_transitivity(tournament: TournamentInstance, _experiment):
    """Calculates how far from a locally transitive graph the graph is, by
  computing ged for each of the two subgraphs induced by each node's in and
  out-degrees.
  """
    g = tournament.graph
    total_distance = 0
    ord_graphs = [ordered(i, 1, {})[0] for i in range(len(g.nodes()))]
    for u in g.nodes():
        g1 = g.subgraph(g.pred[u].keys())
        g2 = g.subgraph(g.succ[u].keys())
        total_distance += ged_blp(ord_graphs[len(g1)], g1)[0]
        total_distance += ged_blp(ord_graphs[len(g2)], g2)[0]
    return total_distance / (len(g.nodes()))


@register("longest_cycle_length")
def longest_cycle_length(tournament, _experiment):
    """Calculates the longest cycle in the tournament. Uses the fact that
  tournaments always have a Hamiltonian path which is also a cycle if and only
  if the tournament is strongly connected."""
    largest = max(nx.strongly_connected_components(tournament.graph), key=len)
    return len(largest)


@register("slater_winners_count", parallel=True)
def slater_winner_count(tournament, _experiment):
    """Calculates the number of Slater winners of a given tournament"""
    return len(slater_winners(tournament))


@register("slater_winner_time", parallel=True, reps=5)
def slater_winner_time(tournament, _experiment):
    """Calculates the time needed to find a single Slater winner of a given tournament"""
    start = time.time()
    slater_winner(tournament)
    end = time.time()
    return end - start


@register('slater_equals_copeland', parallel=True)
def slater_equals_copeland(tournament, _experiment):
    """Calculates whether the Slater winners equal the Copeland winners"""
    for s, c in zip(slater_winners(tournament), copeland_winners(tournament)):
        if s != c:
            return 0
    return 1


@register('unique_slater_equals_copeland', parallel=True)
def unique_slater_equals_copeland(tournament, _experiment):
    """Calculates whether the Slater winners equal the Copeland winners"""
    copeland = copeland_winners(tournament)
    if len(copeland) != 1:
        return 0
    slater = slater_winners(tournament)
    if len(slater) != 1:
        return 0
    return 1 if slater[0] == copeland[0] else 0


@register("copeland_winners_count")
def copeland_winner_count(tournament, _experiment):
    """Calculates the number of copeland winners of a given tournament"""
    return len(copeland_winners(tournament))


@register("top_cycle_winners_count")
def top_cycle_winner_count(tournament, _experiment):
    """Calculates the number of top cycle winners of a given tournament"""
    return len(top_cycle_winners(tournament))


@register("highest_degree")
def highest_degree(tournament, _experiment):
    """Calculates the highest degree of a given tournament"""
    return max(tournament.graph.out_degree(), key=lambda x: x[1])[1]


@register("distortion", parallel=True, reps=5)
def distortion(tournament, experiment):
    """Calculates the distortion of the embedding of a given tournament"""
    eps = 1e-6
    tournament_id = tournament.instance_id
    tournament_coordinates = experiment.coordinates[tournament_id]
    ids = list(experiment.distances.keys())
    ids.remove(tournament_id)

    n = len(tournament.graph.nodes())
    coord_diameter = np.linalg.norm(
        np.array(experiment.coordinates['ordered_0']) - np.array(experiment.coordinates['rps_0']))
    coordinates = np.array([experiment.coordinates[e] for e in ids])
    coordinates_distances = np.linalg.norm(tournament_coordinates - coordinates, axis=1)
    # norm_coordinates_distances = coordinates_distances / np.max(
    #     np.linalg.norm(coordinates[None, :, :] - coordinates[:, None, :], axis=-1))
    norm_coordinates_distances = coordinates_distances / coord_diameter

    dist_diameter = experiment.distances['ordered_0']['rps_0']
    distances = np.array([experiment.distances[tournament_id][i] for i in ids])
    # norm_distances = distances / max(experiment.distances[tournament_id].values())
    norm_distances = distances / dist_diameter

    mins = np.minimum(norm_coordinates_distances, norm_distances)
    maxs = np.maximum(norm_coordinates_distances, norm_distances)
    ratios = maxs[mins > eps] / mins[mins > eps]
    # ratios = np.nan_to_num(ratios, nan=1)
    return np.mean(ratios)


@register("distortion_restricted", parallel=True, reps=5)
def distortion_restricted(tournament, experiment):
    """Calculates the distortion of the embedding of a given tournament"""
    eps = 1e-6
    tournament_id = tournament.instance_id
    tournament_coordinates = experiment.coordinates[tournament_id]
    ids = list(experiment.distances.keys())
    ids.remove(tournament_id)

    n = len(tournament.graph.nodes())
    coord_diameter = np.linalg.norm(
        np.array(experiment.coordinates['ordered_0']) - np.array(experiment.coordinates['rps_0']))
    coordinates = np.array([experiment.coordinates[e] for e in ids])
    coordinates_distances = np.linalg.norm(tournament_coordinates - coordinates, axis=1)
    # norm_coordinates_distances = coordinates_distances / np.max(
    #     np.linalg.norm(coordinates[None, :, :] - coordinates[:, None, :], axis=-1))
    norm_coordinates_distances = coordinates_distances / coord_diameter

    dist_diameter = experiment.distances['ordered_0']['rps_0']
    distances = np.array([experiment.distances[tournament_id][i] for i in ids])
    # norm_distances = distances / max(experiment.distances[tournament_id].values())
    norm_distances = distances / dist_diameter

    cutoff = 0.15
    mins = np.minimum(norm_coordinates_distances, norm_distances)[norm_distances > cutoff]
    maxs = np.maximum(norm_coordinates_distances, norm_distances)[norm_distances > cutoff]
    ratios = maxs[mins > eps] / mins[mins > eps]
    # ratios = np.nan_to_num(ratios, nan=1)
    return np.mean(ratios)


def _single_elimination_wins_probabilistic(g, sample_size=1000):

    def get_winner(g, node_permutation):
        winners = node_permutation
        while len(winners) > 1:
            new_winners = []
            for i in range(1, len(winners), 2):
                if g.has_edge(winners[i - 1], winners[i]):
                    new_winners.append(winners[i - 1])
                else:
                    new_winners.append(winners[i])
            winners = new_winners
        return winners[0]

    g = g.copy()
    n = g.number_of_nodes()
    existing_nodes = list(g)
    loser_nodes = ["loser_" + str(i) for i in range(log2_ceil(n) - len(existing_nodes))]
    for loser in loser_nodes:
        for node in existing_nodes:
            g.add_edge(node, loser)

    win_count = defaultdict(int)
    for _ in range(sample_size):
        node_permutation = list(g)
        random.shuffle(node_permutation)
        winner = get_winner(g, node_permutation)
        win_count[winner] += 1

    return win_count


@register("single_elimination_winners_count", parallel=True, reps=1)
def single_elimination_winners_count(tournament, _experiment):
    """Calculates the number of single elimination winners of a given tournament"""
    return len(_single_elimination_wins_probabilistic(tournament.graph))


@register("single_elimination_win_chance", parallel=True, reps=1)
def single_elimination_win_chance(tournament, _experiment):
    """Calculates the chance to win the tournament by the most probable winner."""
    win_count = _single_elimination_wins_probabilistic(tournament.graph)
    winner = max(win_count.items(), key=lambda x: x[1])
    print(len(win_count), winner[1])
    return winner[1] / sum(win_count.values())


@register("single_elimination_winners_count_ilp", parallel=True)
def single_elimination_winners_count_ilp(tournament, _experiment):
    """Calculates the number of participants that can win a single elimination tournament."""
    count = 0
    for node in tournament.graph.nodes:
        if single_elimination_can_player_win(tournament, node)[0] > 0.5:
            count += 1
    return count


@register("single_elimination_winners_count_ilp_time", parallel=True, reps=15)
def single_elimination_winners_count_ilp_time(tournament, _experiment):
    """Calculates the average time to check if a player can win a single elimination tournament."""
    start = time.time()
    for node in tournament.graph.nodes:
        single_elimination_can_player_win(tournament, node)[0]
    end = time.time()
    return (end - start)


@register("single_elimination_winner_longest_ilp_time", parallel=False, reps=5)
def single_elimination_winner_longest_ilp_time(tournament, _experiment):
    """Calculates the average time to check if a player can win a single elimination tournament."""
    worst_time = 0
    for node in tournament.graph.nodes:
        start = time.time()
        a = single_elimination_can_player_win(tournament, node)[0]
        print(a)
        end = time.time()
        worst_time = max(worst_time, end - start)
    return worst_time * len(tournament.graph.nodes)


if __name__ == '__main__':
    t = TournamentInstance.raw(nx.tournament.random_tournament(5))
    # plot and show graph
    print("Longest cycle:", longest_cycle_length(t))
    nx.draw_circular(t.graph, with_labels=True)
    plt.show()
