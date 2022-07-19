
import numpy as np


def generate_approval_partylist_votes(num_voters=None, num_candidates=None, params=None):

    if params is None or 'g' not in params:
        num_groups = 5
    else:
        num_groups = params['g']
    alphas = get_vector('linear', num_groups)

    # for i in range(len(alphas)):
    #     if alphas[i] == 0.:
    #         alphas[i] = 0.00001

    sizes = np.random.dirichlet(alphas)
    cumv = np.cumsum(sizes)
    cumv = np.insert(cumv, 0, 0)
    print(cumv)

    votes = []

    for i in range(0,0):
        print(i)

    for g in range(1, num_groups+1):
        vote = set()
        print(int(num_candidates*cumv[g-1]))
        print(int(num_candidates*cumv[g]))
        for i in range(int(num_candidates*cumv[g-1]), int(num_candidates*cumv[g])):
            print(i)
            vote.add(i)
        for i in range(int(num_candidates * cumv[g - 1]), int(num_candidates * cumv[g])):
            votes.append(vote)
    print(votes)
    return votes



# AUXILIARY (alphas)

def get_vector(type, num_candidates):
    if type == "uniform":
        return [1.] * num_candidates
    elif type == "linear":
        return [(num_candidates - x) for x in range(num_candidates)]
    elif type == "linear_low":
        return [(float(num_candidates) - float(x)) / float(num_candidates) for x in range(num_candidates)]
    elif type == "square":
        return [(float(num_candidates) - float(x)) ** 2 / float(num_candidates) ** 2 for x in
                range(num_candidates)]
    elif type == "square_low":
        return [(num_candidates - x) ** 2 for x in range(num_candidates)]
    elif type == "cube":
        return [(float(num_candidates) - float(x)) ** 3 / float(num_candidates) ** 3 for x in
                range(num_candidates)]
    elif type == "cube_low":
        return [(num_candidates - x) ** 3 for x in range(num_candidates)]
    elif type == "split_2":
        values = [1.] * num_candidates
        for i in range(num_candidates / 2):
            values[i] = 10.
        return values
    elif type == "split_4":
        size = num_candidates / 4
        values = [1.] * num_candidates
        for i in range(size):
            values[i] = 1000.
        for i in range(size, 2 * size):
            values[i] = 100.
        for i in range(2 * size, 3 * size):
            values[i] = 10.
        return values
    else:
        return type