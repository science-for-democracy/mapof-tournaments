from mapel.elections.objects.OrdinalElection import OrdinalElection
import mapel.tournaments
from mapel.tournaments.objects.GraphSimilarity import Distances
from progress.bar import Bar
# from mapel import generate_ordinal_election
import signal

# Make interrupt work with plots
signal.signal(signal.SIGINT, signal.SIG_DFL)

experiment = mapel.tournaments.TournamentExperiment()
# experiment.add_family(culture_id='impartial_culture',
#                       size=10,
#                       color='green',
#                       marker='x',
#                       label='IC',
#                       instance_type='ordinal')
# experiment.add_family(culture_id='norm-mallows',
#                       size=10,
#                       params={'norm-phi': 0.5},
#                       color='blue',
#                       marker='o',
#                       label='Norm-Mallows',
#                       instance_type='ordinal')
n = 10
weights = [1] * n
experiment.add_family(culture_id='uniform',
                      color='blue',
                      family_id='uniform',
                      label='Uniform',
                      size=100,
                      params={'weights': weights})
experiment.add_family(culture_id='Ordered',
                      color='red',
                      family_id='Ordered',
                      label='Ordered',
                      single=True,
                      num_participants=n,
                      plot_path='graphs',
                      params={'compass': 'ordered'})
experiment.add_family(culture_id='Unordered',
                      color='green',
                      family_id='Unordered',
                      label='Unordered',
                      single=True,
                      num_participants=n,
                      plot_path='graphs',
                      params={'compass': 'unordered'})
experiment.add_family(culture_id='Two-Unordered',
                      color='orange',
                      family_id='Two-Unordered',
                      label='Two-Unordered',
                      single=True,
                      num_participants=n,
                      plot_path='graphs',
                      params={'compass': 'two_unordered'})
experiment.add_family(culture_id='urn',
                      instance_type='ordinal',
                      color='cyan',
                      family_id='urn_1',
                      params={'alpha': 0.1},
                      size=8)

experiment.add_family(culture_id='norm-mallows',
                      instance_type='ordinal',
                      color='brown',
                      family_id='mallows_1',
                      params={'norm-phi': 0.1},
                      size=8)

experiment.add_family(culture_id='norm-mallows',
                      instance_type='ordinal',
                      color='purple',
                      family_id='mallows_2',
                      params={'norm-phi': 0.5},
                      size=8)

# experiment.save_tournament_plots()
experiment.compute_distances(Distances.DEGREE_EMD, opt=4)
experiment.embed(embedding_id='kk')
experiment.print_map()

# oe = OrdinalElection(experiment_id='test',
#                      election_id='test',
#                      num_candidates=5,
#                      num_voters=5,
#                      votes=[[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1],
#                             [3, 4, 0, 1, 2], [4, 0, 1, 2, 3]])

# print(oe.votes_to_pairwise_matrix())
