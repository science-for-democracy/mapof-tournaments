import json
import os
import pickle

import networkx as nx
from mapel.core.utils import make_folder_if_do_not_exist
from mapel.tournaments.objects.TournamentCultures import rock_paper_scissors


def load_dict_from_file(path):
  with open(path, 'r') as f:
    return json.load(f)


def log2_ceil(x):
  return 1 << (x - 1).bit_length()


def fill_with_losers_up_to_a_power_of_two(g):
  n = g.number_of_nodes()
  losers = rock_paper_scissors(log2_ceil(n) - n, 1, {})[0]
  losers = nx.relabel_nodes(losers, {i: "loser_" + str(i)
                                     for i in losers.nodes})

  res_g = nx.compose(g, losers)
  for node in g.nodes:
    for loser in losers.nodes:
      res_g.add_edge(node, loser)
  return res_g


# Create a decorator that takes all function arguments and checks if the
# function has already been called with those by checking if a pickle file
# exists with the same name as the function and the same arguments. If it does,
# load the pickle file and return it. If it doesn't, run the function, save the
# result to a pickle file, and return it.
def cache(experiment_prefix=""):

  def _cache(func):
    DIR = 'caches/'

    def wrapper(*args, **kwargs):
      make_folder_if_do_not_exist(DIR)
      # Get the name of the function
      name = experiment_prefix + func.__name__
      # Get the name of the pickle file
      filename = "".join([name, str(args), str(kwargs)])
      filename += ".pickle"
      filepath = DIR + filename
      # If the pickle file exists
      if os.path.exists(filepath):
        # Load the pickle file
        with open(filepath, 'rb') as f:
          result = pickle.load(f)
      # If the pickle file does not exist
      else:
        # Run the function
        result = func(*args, **kwargs)
        # Save the result to a pickle file
        with open(filepath, 'wb') as f:
          pickle.dump(result, f)
      # Return the result
      return result

    return wrapper

  return _cache
