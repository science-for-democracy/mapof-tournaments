import json


def load_dict_from_file(path):
  with open(path, 'r') as f:
    return json.load(f)
