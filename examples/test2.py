import pickle
import mapel
from mapel.elections.objects.OrdinalElection import OrdinalElection
import mapel.tournaments
from mapel.tournaments.objects.GraphSimilarity import Distances
from progress.bar import Bar
# from mapel import generate_ordinal_election
import signal

# Make interrupt work with plots
signal.signal(signal.SIGINT, signal.SIG_DFL)

n = 20
s = 200
experiment = mapel.tournaments.TournamentExperiment()
# weights = list([i * 3 for i in range(1, n + 1)])
# experiment.add_family(
#     culture_id='ordered-random',
#     color='blue',
#     family_id='ordered-random',
#     label='Ordered Random',
#     size=s,
#     # plot_path='graphs',
#     params={'weights': weights})
# weights = [1] * n
# experiment.add_family(
#     culture_id='unordered-random',
#     color='green',
#     family_id='unordered-random',
#     label='Rock-Paper-Scissors Random',
#     size=s,
#     # plot_path='graphs',
#     params={'weights': weights})
# weights = [1] * (n // 2) + [9] * (n // 2)
# experiment.add_family(
#     culture_id='biunordered-random',
#     color='gray',
#     family_id='biunordered-random',
#     label='Biunordered Random',
#     size=s,
#     # plot_path='graphs',
#     params={'weights': weights})
# experiment.add_family(
#     culture_id='Ordered',
#     color='red',
#     family_id='Ordered',
#     label='Ordered',
#     single=True,
#     num_participants=n,
#     # plot_path='graphs',
#     params={'compass': 'ordered'})
# experiment.add_family(
#     culture_id='Rock-Paper-Scissors',
#     color='purple',
#     family_id='Rock-Paper-Scissors',
#     label='Rock-Paper-Scissors',
#     single=True,
#     num_participants=n,
#     # plot_path='graphs',
#     params={'compass': 'unordered'})
# experiment.add_family(
#     culture_id='Biunordered',
#     color='orange',
#     family_id='Biunordered',
#     label='Biunordered',
#     single=True,
#     num_participants=n,
#     # plot_path='graphs',
#     params={'compass': 'biunordered'})
# experiment.add_family(culture_id='urn',
#                       instance_type='ordinal',
#                       color='cyan',
#                       family_id='urn_1',
#                       params={'alpha': 0.1},
#                       size=8)

# experiment.add_family(culture_id='norm-mallows',
#                       instance_type='ordinal',
#                       color='brown',
#                       family_id='mallows_1',
#                       params={'norm-phi': 0.1},
#                       size=8)

# experiment.add_family(culture_id='norm-mallows',
#                       instance_type='ordinal',
#                       color='purple',
#                       family_id='mallows_2',
#                       params={'norm-phi': 0.5},
#                       size=8)
experiment.add_family(
    culture_id='1',
    instance_type='tournament',
    color='red',
    family_id='1',
    params={'adjacency_matrix': [[0, 1, 0], [0, 0, 1], [1, 0, 0]]},
    single=True)
experiment.add_family(
    culture_id='2',
    instance_type='tournament',
    label='2',
    color='blue',
    family_id='2',
    params={'adjacency_matrix': [[0, 1, 1], [0, 0, 1], [0, 0, 0]]},
    single=True)

# experiment.save_tournament_plots()
experiment.compute_distances(Distances.SP, opt=4, parallel=True)
experiment.embed(embedding_id='mds')
experiment.print_map()
# fig.show()
# pickle.dump(fig, open('t.pickle', 'wb'))

# oe = OrdinalElection(experiment_id='test',
#                      election_id='test',
#                      num_candidates=5,
#                      num_voters=5,
#                      votes=[[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1],
#                             [3, 4, 0, 1, 2], [4, 0, 1, 2, 3]])

# print(oe.votes_to_pairwise_matrix())

#papier kamien nożyce
