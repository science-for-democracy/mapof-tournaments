#!/usr/bin/env python

import copy
from mapel.main.objects.Family import Family
from mapel.elections.objects.OrdinalElection import OrdinalElection
from mapel.elections._glossary import *
from mapel.main._utils import *
import mapel.elections.models.mallows as mallows


class ElectionFamily(Family):
    """ Family of elections: a set of elections from the same election model_id """

    def __init__(self,
                 model_id: str = None,
                 family_id='none',
                 params: dict = None,
                 size: int = 1,
                 label: str = "none",
                 color: str = "black",
                 alpha: float = 1.,
                 ms: int = 20,
                 show=True,
                 marker='o',
                 starting_from: int = 0,
                 path: dict = None,
                 single: bool = False,

                 num_candidates=None,
                 num_voters=None,
                 election_ids=None,
                 ballot: str = 'ordinal'):

        super().__init__(model_id=model_id,
                         family_id=family_id,
                         params=params,
                         size=size,
                         label=label,
                         color=color,
                         alpha=alpha,
                         ms=ms,
                         show=show,
                         marker=marker,
                         starting_from=starting_from,
                         path=path,
                         single=single,
                         instance_ids=election_ids)

        self.num_candidates = num_candidates
        self.num_voters = num_voters
        self.ballot = ballot

    def __getattr__(self, attr):
        if attr == 'election_ids':
            return self.instance_ids
        else:
            return self.__dict__[attr]

    def __setattr__(self, name, value):
        if name == "election_ids":
            return setattr(self, 'instance_ids', value)
        else:
            self.__dict__[name] = value

    def prepare_family(self, experiment_id=None, store=None):

        ballot = get_ballot_from_model(self.model_id)

        # if model_id in PARTY_MODELS:
        #     params['party'] = prepare_parties(params=params, model_id=model_id)

        elections = {}
        _keys = []
        for j in range(self.size):

            params = copy.deepcopy(self.params)

            variable = None
            path = self.path
            if path is not None and 'variable' in path:
                new_params, variable = _get_params_for_paths(self, j)
                params = {**params, **new_params}

            if params is not None and 'norm-phi' in params:
                params['phi'] = mallows.phi_from_relphi(
                                    self.num_candidates, relphi=params['norm-phi'])

            if self.model_id in {'all_votes'}:
                params['iter_id'] = j

            if self.model_id in {'crate'}:
                new_params = _get_params_for_crate(j)
                params = {**params, **new_params}

            election_id = get_instance_id(self.single, self.family_id, j)

            election = OrdinalElection(experiment_id, election_id, model_id=self.model_id,
                                         num_voters=self.num_voters,
                                         num_candidates=self.num_candidates,
                                         params=copy.deepcopy(params), ballot=ballot,
                                        variable=variable, _import=False,
                                       )

            election.prepare_instance(store=store, params=params)

            elections[election_id] = election

            _keys.append(election_id)

        self.election_ids = _keys


        return elections


def _get_params_for_crate(j):
    base = []
    my_size = 10
    # with_edge
    for p in range(my_size):
        for q in range(my_size):
            for r in range(my_size):
                a = p / (my_size - 1)
                b = q / (my_size - 1)
                c = r / (my_size - 1)
                d = 1 - a - b - c
                tmp = [a, b, c, d]
                if d >= 0 and sum(tmp) == 1:
                    base.append(tmp)
    params = {'alpha': base[j]}
    return params


def _get_params_for_paths(family, j, extremes=False):
    path = family.path

    variable = path['variable']

    if 'extremes' in path:
        extremes = path['extremes']

    params = {}
    if extremes:
        params[variable] = j / (family.size - 1)
    elif not extremes:
        params[variable] = (j + 1) / (family.size + 1)

    if 'scale' in path:
        params[variable] *= path['scale']

    if 'start' in path:
        params[variable] += path['start']
    else:
        path['start'] = 0.

    if 'step' in path:
        params[variable] = path['start'] + j * path['step']

    return params, variable


def get_ballot_from_model(model_id: str) -> str:
    if model_id in APPROVAL_MODELS:
        return 'approval'
    else:
        return 'ordinal'


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 12.10.2021 #
# # # # # # # # # # # # # # # #
