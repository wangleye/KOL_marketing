import random
import numpy as np
import pymysql
import time
import math
import pp
import simulate_hit_pp as sh
import logging

conn = pymysql.connect(host='127.0.0.1',
    user='root',
    passwd='123456',
    db='all0504')

TOTAL_GROUP_NUM = 100
TOTAL_ITEM_NUM = 100

TEST_ITEM_NUM = 20 # the number of items used in the test
TEST_GROUP_NUM = 20 # the number of groups used in the test

SIM = {} # dictionary / numpy matrix to store the item similarity matrix
GROUP_USERS = {}
GROUP_CSD = {}
GROUP_TP_RANK = {}
USER_NUM_COSTS = {}
NET_COSTS = {}
COSTS = {}
GROUPS = []
ALL_ITEMS = []
ITEMS = []
ITEM_FANS = {}
ITEM_REVENUES = {}
USER_FRIENDS = {}
SLOTS = {}
SLOT_NUM = 1
BUDGET = 1
COST_TYPE = 'number' # 'net' or 'number'

alpha = 0.02
epsilon = 0.1

DATA_DIR = './facebook'

CACHE_UTILITY = {}
CACHE_HIT_USERS = {}

def init_slots(k):
    global SLOT_NUM
    SLOT_NUM = k
    for group in GROUP_USERS:
        SLOTS[group] = k

def load_all_simulated_hits():
    groups = GROUP_USERS.keys()
    items = ALL_ITEMS
    print 'group len:', len(groups), 'item len:', len(items)
    for idx, group in enumerate(groups):
        print idx, group
        for item in items:
            load_simulated_hits(group, item)

def load_simulated_hits(group_id, item_id):
    query_str = "select group_id, item_id, hit_users from simulate_group_rec_book where alpha between {} and {} and group_id = '{}' and item_id = '{}' limit 1000".format(alpha-0.001, alpha+0.001, group_id, item_id)
    x = conn.cursor()
    x.execute(query_str)
    results = x.fetchall()
    hit_users_list = list()
    for each_result in results:
        item = str(each_result[1])
        group = str(each_result[0])
        hit_users = set(str(each_result[2]).split(','))
        hit_users_list.append(hit_users)
    CACHE_HIT_USERS[set2key((item_id, group_id))] = hit_users_list

def load_groups():
    global GROUPS
    GROUPS = random.sample(GROUP_USERS.keys(), TEST_GROUP_NUM)
    return GROUPS

def load_all_items():
    """
    read item list from file
    """
    global ALL_ITEMS
    with open("{}/book_list".format(DATA_DIR)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                item, item_count = line.split()
                ALL_ITEMS.append(item)
    ALL_ITEMS = ALL_ITEMS[0:TOTAL_ITEM_NUM]

def load_item_revenues():
    """
    load revenues of items
    """
    global ITEM_REVENUES
    with open("{}/book_price".format(DATA_DIR)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                item, item_price = line.split()
                ITEM_REVENUES[item] = float(item_price)


def load_items():
    """
    read item list from file
    """
    global ITEMS
    ITEMS = random.sample(ALL_ITEMS, TEST_ITEM_NUM)
    return ITEMS

def load_user_item_similarity():
    """
    read an item similarity file
    """
    read_user_item_similarity_file_line_by_line()


def read_user_item_similarity_file_line_by_line():
    global SIM
    global ITEM_FANS
    with open("{}/user_book_aff_score_100_item_only_KOL_complete".format(DATA_DIR)) as inputfile:
        first_line = True
        for line in inputfile:
            if first_line:
                first_line = False # skip the first line
                continue
            line = line.strip()
            if len(line) > 0:
                words = line.split()
                user = words[0]
                item = words[1]
                similarity = float(words[2])
                is_true_like = int(words[3])

                if user not in SIM:
                    SIM[user] = {}
                SIM[user][item] = similarity

                if item not in ITEM_FANS:
                    ITEM_FANS[item] = set()
                
                if is_true_like == 1:
                    ITEM_FANS[item].add(user)

def load_group_costs():
    """
    return group costs from file
    """
    global COSTS
    with open("{}/KOL_number_cost".format(DATA_DIR)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                words = line.strip().split(';')
                group_id = words[0]
                normalized_cost = float(words[1])/BUDGET
                USER_NUM_COSTS[group_id] = normalized_cost

    with open("{}/KOL_net_cost".format(DATA_DIR)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                words = line.strip().split(';')
                group_id = words[0]
                normalized_cost = float(words[1])/BUDGET
                NET_COSTS[group_id] = normalized_cost

    if COST_TYPE == 'num':
        COSTS = USER_NUM_COSTS
    else:
        COSTS = NET_COSTS

def load_group_users_and_csd():
    """
    return groups
    """
    max_user_group_count = 0
    with open("{}/KOL_audience".format(DATA_DIR)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                words = line.strip().split(';')
                group_id = words[0]
                users = words[1].split()
                GROUP_USERS[group_id] = users
                if len(users) > max_user_group_count:
                    max_user_group_count = len(users)

    print('user number of the largest group:', max_user_group_count)

    with open("{}/KOL_CSD".format(DATA_DIR)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                words = line.strip().split(';')
                group_id = words[0]
                csd = float(words[1])
                GROUP_CSD[group_id] = csd

    count = 1
    with open("{}/KOL_tp_rankedlist".format(DATA_DIR)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                group_id = line.strip()
                GROUP_TP_RANK[group_id] = count
                count += 1

def set2key(s):
    """
    change a set to string key
    """
    return ' '.join(str(x) for x in s)

def utility_user_count(hit_users):
    users_set = set()
    for item in hit_users:
        users_set.update(hit_users[item])
    return len(users_set)

def utility_unit_revenue(hit_users):
    revenue = 0
    for item in hit_users:
        if item in ITEM_REVENUES:
            item_r = ITEM_REVENUES[item]
        else:
            item_r = 10
        revenue += len(hit_users[item]) * item_r
    return revenue

def utility_monte_carlo(rec_pairs):
    K = 1000 # simulation times
    if len(rec_pairs) == 0:
        return 0

    # if in the utility cache, directly obtain it
    if set2key(rec_pairs) in CACHE_UTILITY:
        return CACHE_UTILITY[set2key(rec_pairs)]

    utility_sum = 0

    for i in range(K):
        hit_users = {}
        for (item, group) in rec_pairs:
            hit_users[item] = set() # initialize hit users for any item in recommendations

        for (item, group) in rec_pairs:
            sim_hit_users = CACHE_HIT_USERS[set2key((item, group))][i]
            hit_users[item].update(sim_hit_users)

        # use the HIT_USERS to calculate utility
        utility_sum += UTILITY_FUNCTION(hit_users)
    utility = utility_sum * 1.0 / K

    # save to cache
    CACHE_UTILITY[set2key(rec_pairs)] = utility

    return utility

def find_max_utility(groups, items, costs, rho=0):
    max_utility = 0
    for g in groups:
        if COSTS[g] > 1:
            continue
        for i in items:
            recommend_pairs = {(i,g)} # initialize a set containing only one recommendation
            utility_ig = utility_monte_carlo(recommend_pairs)
            #print '({},{}): {}'.format(i,g,utility_ig)
            if utility_ig > max_utility and (rho == 0 or utility_ig*1.0/costs[g] >= rho):
                max_utility = utility_ig
    return max_utility

def sum_cost(costs, selected_pairs):
    sum_c = 0
    for (i, g) in selected_pairs:
        sum_c += costs[g]
    return sum_c

def find_max_utility_increase(groups, items, costs, slots, selected_pairs):
    remain_cost = 1 - sum_cost(costs, selected_pairs)
    max_utility_increase = 0
    max_pair = None
    max_utility = 0
    for g in groups:
        if costs[g] > remain_cost or (not still_has_slot(selected_pairs, slots, g)):
            continue
        for i in items:
            recommend_pair = {(i,g)}
            utility_ig = utility_monte_carlo(recommend_pair.union(selected_pairs)) - utility_monte_carlo(selected_pairs)
            if utility_ig > max_utility_increase:
                max_utility_increase = utility_ig
                max_pair = (i,g)
                max_utility = utility_monte_carlo(recommend_pair.union(selected_pairs))

    return max_pair, max_utility_increase, max_utility

def find_max_utility_per_cost_increase(groups, items, costs, slots, selected_pairs):
    remain_cost = 1 - sum_cost(costs, selected_pairs)
    max_utility_increase_per_cost = 0
    max_utility = 0
    max_pair = None
    for g in groups:
        if costs[g] > remain_cost or (not still_has_slot(selected_pairs, slots, g)):
            continue
        for i in items:
            recommend_pair = {(i,g)}
            utility_ig = utility_monte_carlo(recommend_pair.union(selected_pairs)) - utility_monte_carlo(selected_pairs)
            utility_ig_per_cost = utility_ig * 1.0 / costs[g]
            if utility_ig_per_cost > max_utility_increase_per_cost:
                max_utility_increase_per_cost = utility_ig_per_cost
                max_pair = (i,g)
                max_utility = utility_monte_carlo(recommend_pair.union(selected_pairs))

    return max_pair, max_utility_increase_per_cost, max_utility

def is_over_budget(selected_recs, normalized_costs):
    """
    judge whether the selected recommendations is over the budget
    """
    sum_cost = 0
    for (item, group) in selected_recs:
        sum_cost += normalized_costs[group]
    if sum_cost > 1:
        return True
    else:
        return False

def still_has_slot(selected_recs, slots, group_id):
    """
    judge whether a group still has slot
    """
    max_slot = slots[group_id]
    current_slot = 0
    for (item, group) in selected_recs:
        if group == group_id:
            current_slot += 1
    if current_slot < max_slot:
        return True
    else:
        return False

def is_over_slot_constraint(selected_recs, slots, group_id):
    """
    judge whether a group slot constraint is violated
    """
    max_slot = slots[group_id]
    current_slot = 0
    for (item, group) in selected_recs:
        if group == group_id:
            current_slot += 1
    if current_slot > max_slot:
        return True
    else:
        return False

def near_opt_group_rec(groups, items, normalized_costs, slots):
    M = find_max_utility(groups, items, normalized_costs)
    rho = M*1.0/2
    R = {}
    n = len(groups)*len(items)
    while rho <= n*M:
        # print 'rho:', rho
        print 'max:', get_max_value_from_R(R)
        M_rho = find_max_utility(groups, items, normalized_costs, rho)
        if M_rho == 0:
            break
        tau = M_rho
        S = set();
        while tau  >= epsilon*1.0 / n * M_rho and not is_over_budget(S, normalized_costs):
            # use shuffle to add randomness into the algorithm
            candidates = []
            for g in groups:
                for i in items:
                    candidates.append((i,g))
            random.shuffle(candidates)
            for i, g in candidates:
                # lazy evaluation (speed-up)
                utility_increase_max = utility_monte_carlo({(i,g)})
                if (i, g) in S or utility_increase_max < tau or utility_increase_max / normalized_costs[g] < rho or normalized_costs[g] > 1:
                    continue

                S_prime = S.union({(i,g)})
                if not is_over_slot_constraint(S_prime, slots, g) and \
                utility_monte_carlo(S_prime) - utility_monte_carlo(S) >= tau and \
                (utility_monte_carlo(S_prime) - utility_monte_carlo(S)) / normalized_costs[g] >= rho:
                    if is_over_budget(S_prime, normalized_costs):
                        #R[str(rho)] = (S, utility_monte_carlo(S))
                        # local improvement - using S as base, simple/cost greedy
                        S_sg, u_sg = simple_greedy(groups, items, normalized_costs, slots, S)
                        S_cg, u_cg = cost_greedy(groups, items, normalized_costs, slots, S)
                        if u_sg >= u_cg:
                            R[str(rho)] = (S_sg, u_sg)
                        else:
                            R[str(rho)] = (S_cg, u_cg)

                        #R[str(-rho)] = ({(i,g)}, utility_monte_carlo({(i,g)}))
                        S_neg_sg, u_neg_sg = simple_greedy(groups, items, normalized_costs, slots, {(i,g)})
                        S_neg_cg, u_neg_cg = cost_greedy(groups, items, normalized_costs, slots, {(i,g)})
                        if u_neg_sg >= u_neg_cg:
                            R[str(-rho)] = (S_neg_sg, u_neg_sg)
                        else:
                            R[str(-rho)] = (S_neg_cg, u_neg_cg)
                        # continue with next rho
                    else:
                        S = S_prime
                
                if str(rho) in R:
                    break
            if str(rho) in R:
                break
            tau = 1.0/(1+epsilon) * tau

        if str(rho) not in R:
            #R[str(rho)] = (S, utility_monte_carlo(S))
            #R[str(-rho)] = (set(), 0)
            #local improvement
            S_sg, u_sg = simple_greedy(groups, items, normalized_costs, slots, S)
            S_cg, u_cg = cost_greedy(groups, items, normalized_costs, slots, S)
            if u_sg >= u_cg:
                R[str(rho)] = (S_sg, u_sg)
            else:
                R[str(rho)] = (S_cg, u_cg)

        rho = (1.0+epsilon) * rho
    return get_max_value_from_R(R)

def get_max_value_from_R(R):
    max_S = set()
    max_utility = 0
    max_rho = 0
    for each_rho in R:
        (S, u) = R[each_rho]
        if u > max_utility:
            max_utility = u
            max_S = S
            max_rho = each_rho
    return max_S, max_utility, max_rho

def simple_greedy(groups, items, normalized_costs, slots, S):
    max_utility = utility_monte_carlo(S)
    new_pair, utility_increase, utility = find_max_utility_increase(groups, items, normalized_costs, slots, S)
    while new_pair is not None:
        S = S.union({new_pair})
        max_utility = utility
        new_pair, utility_increase, utility = find_max_utility_increase(groups, items, normalized_costs, slots, S)
    return S, max_utility

def cost_greedy(groups, items, normalized_costs, slots, S):
    max_utility = utility_monte_carlo(S)
    new_pair, utility_increase, utility = find_max_utility_per_cost_increase(groups, items, normalized_costs, slots, S)
    while new_pair is not None:
        S = S.union({new_pair})
        max_utility = utility
        new_pair, utility_increase, utility = find_max_utility_per_cost_increase(groups, items, normalized_costs, slots, S)
    return S, max_utility

def random_greedy(input_groups, items, normalized_costs, slots):
    S = set()
    groups = input_groups[:]
    max_utility = utility_monte_carlo(S)
    rand_group = random.choice(groups)
    item = random.choice(items)
    groups.remove(rand_group)
    new_pair, utility_increase, utility = find_max_utility_increase([rand_group,], [item,], normalized_costs, slots, S)
    while len(groups) > 0: # test all the groups until no one can be added
        if new_pair != None: # sometimes rand_group's cost is too high, then new_pair will be None, but we still can find more groups
            S = S.union({new_pair})
            max_utility = utility
        rand_group = random.choice(groups)
        groups.remove(rand_group)
        item = random.choice(items)
        new_pair, utility_increase, utility = find_max_utility_increase([rand_group,], [item,], normalized_costs, slots, S)
    return S, max_utility

def baseline_greedy(input_groups, items, normalized_costs, slots, sort_reference, is_reverse=True):
    S = set()
    groups = input_groups[:]
    groups = sorted(groups, key=lambda x: sort_reference[x], reverse=is_reverse)
    max_utility = utility_monte_carlo(S)
    for group in groups:
        new_pair, utility_increase, utility = find_max_utility_increase([group,], items, normalized_costs, slots, S)
        if new_pair != None:
            S = S.union({new_pair})
            max_utility = utility
    return S, max_utility

def KOL_prefer_greedy(input_groups, items, normalized_costs, slots, sort_reference, is_reverse=True):
    S = set()
    groups = input_groups[:]
    groups = sorted(groups, key=lambda x: sort_reference[x], reverse=is_reverse)
    for group in groups: # group id is KOL
        similarity = 0
        best_item = random.choice(items)
        for item in items: # select the best time for the KOL
            if group in SIM and item in SIM[group] and SIM[group][item] > similarity:
                best_item = item
                similarity = SIM[group][item]
        new_pair, utility_increase, utility = find_max_utility_increase([group,], [best_item,], normalized_costs, slots, S)
        if new_pair != None:
            S = S.union({new_pair})
            max_utility = utility
    return S, max_utility

def audience_prefer_greedy(input_groups, items, normalized_costs, slots, sort_reference, is_reverse=True):
    S = set()
    groups = input_groups[:]
    groups = sorted(groups, key=lambda x: sort_reference[x], reverse=is_reverse)
    for group in groups: # group id is KOL
        similarity = 0
        best_item = random.choice(items)
        for item in items: # select the best time for the KOL
            audience_score = audience_item_score(group, item)
            if audience_score > similarity:
                best_item = item
                similarity = audience_score
        new_pair, utility_increase, utility = find_max_utility_increase([group,], [best_item,], normalized_costs, slots, S)
        if new_pair != None:
            S = S.union({new_pair})
            max_utility = utility
    return S, max_utility

def audience_item_score(group, item):
    score = 0
    for user in GROUP_USERS[group]:
        if user in SIM and item in SIM[user]:
            score += SIM[user][item]
    return score

def benchmark(input_groups, items, normalized_costs, slots, sort_reference, is_reverse, item_select_method):
    if item_select_method == 'KP': # kol prefer
        return KOL_prefer_greedy(input_groups, items, normalized_costs, slots, sort_reference, is_reverse)
    if item_select_method == 'AP': # audience prefer
        return audience_prefer_greedy(input_groups, items, normalized_costs, slots, sort_reference, is_reverse)
    if item_select_method == 'BU': # best utility
        return baseline_greedy(input_groups, items, normalized_costs, slots, sort_reference, is_reverse)

def CSD_KP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, GROUP_CSD, True, 'KP')

def CSD_AP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, GROUP_CSD, True, 'AP')

def CSD_BU(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, GROUP_CSD, True, 'BU')

def AS_KP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, USER_NUM_COSTS, True, 'KP')

def AS_AP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, USER_NUM_COSTS, True, 'AP')

def AS_BU(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, USER_NUM_COSTS, True, 'BU')

def NV_KP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, NET_COSTS, True, 'KP')

def NV_AP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, NET_COSTS, True, 'AP')

def NV_BU(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, NET_COSTS, True, 'BU')

def TP_KP(input_groups, items, normalized_costs, slots):
    return baseline_greedy(input_groups, items, normalized_costs, slots, GROUP_TP_RANK, False, 'KP')

def TP_AP(input_groups, items, normalized_costs, slots):
    return baseline_greedy(input_groups, items, normalized_costs, slots, GROUP_TP_RANK, False, 'AP')

def TP_BU(input_groups, items, normalized_costs, slots):
    return baseline_greedy(input_groups, items, normalized_costs, slots, GROUP_TP_RANK, False, 'BU')

# for simulating final results
def simulate_final_utility(rec_pairs, simulation_times=10000):
    utility_sum = 0
    for i in range(simulation_times):
        hit_users = {}
        for (item, group) in rec_pairs:
            hit_users[item] = set() # initialize hit users for any item in recommendations

        for (item, group) in rec_pairs:
            hit_users[item].update(simulate_final_hit_users(item, GROUP_USERS[group]))

        # use the HIT_USERS to calculate utility
        utility_sum += UTILITY_FUNCTION(hit_users)
    utility = utility_sum * 1.0 / simulation_times
    return utility

def simulate_final_hit_users(item, users_in_group):
    """
    simulation to get hit users for calculating utility
    """
    hit_users = set()
    share_users = set()
    # hit users in group
    for u in users_in_group:
        if u in ITEM_FANS[item]:
            hit_users.add(u)
            if (random.random()<=alpha):
                share_users.add(u)

    # hit users through social influcence
    while len(share_users) > 0:
        new_share_users = set()
        for u in share_users:
            for f in friends(u):
                if f in hit_users:
                    continue
                if f in ITEM_FANS[item]:
                    hit_users.add(f)
                    if random.random()<=alpha:
                        new_share_users.add(f)
        share_users = new_share_users
    return hit_users

def friends(user_id):
    """
    return the friends' ids of a user
    """
    if user_id in USER_FRIENDS:
        return USER_FRIENDS[user_id]
    # read from DB
    query_statement = "SELECT friendstr FROM user WHERE iduser = '{}'".format(user_id)
    x = conn.cursor()
    x.execute(query_statement)
    results = x.fetchall()
    if len(results) == 0:
        return set()
    else:
        friendstr = results[0][0]
        friend_ids = friendstr.split(';')
        USER_FRIENDS[user_id] = friend_ids
        return friend_ids

def get_result_str(result_array):
    result_str = str(np.mean(np.asarray(result_array)))
    return result_str

def evaluate(test_method, method_name):
    started = time.clock()
    results = test_method(GROUPS, ITEMS, COSTS, SLOTS)
    real_result = simulate_final_utility(results[0])
    print method_name, ":", results, real_result
    # logger.info('CSD greedy: {}'.format(results))
    finished = time.clock()
    print finished - started, ' seconds'
    return real_result


def test_utility_difference():
    uitlity_est_vector = []
    utility_act_vector = []
    for item in ITEMS:
        print item
        for group in GROUPS:
            utility_est = utility_monte_carlo([(item, group),])
            utility_act = simulate_final_utility([(item, group),])
            uitlity_est_vector.append(utility_est*1.0/len(GROUP_USERS[group]))
            utility_act_vector.append(utility_act*1.0/len(GROUP_USERS[group]))

    uitlity_est_vector = np.array(uitlity_est_vector)
    utility_act_vector = np.array(utility_act_vector)

    print 'mean utility_est_vector', np.mean(uitlity_est_vector)
    print 'mean utility_act_vector', np.mean(utility_act_vector)
    print 'mean difference', np.mean(np.abs(uitlity_est_vector-utility_act_vector))

    np.savetxt('estimate_utility', uitlity_est_vector)
    np.savetxt('actual_utility', utility_act_vector)


def evaluate_utility_estimation_effect():
    utility_est = np.loadtxt('estimate_utility').reshape((100,100))
    utility_act = np.loadtxt('actual_utility').reshape((100,100))

    estimate_top_utility_index = np.argmax(utility_est, axis=1)
    true_utility_of_estimate = np.zeros(100)

    for i in range(100):
        true_utility_of_estimate[i] = utility_act[i, estimate_top_utility_index[i]]

    true_top_utility = np.max(utility_act, axis=1)
    print(true_top_utility)
    print(np.mean(true_top_utility))
    print(np.mean(true_utility_of_estimate))

    random_estimate = np.zeros(100)
    for i in range(100):
        random_estimate[i] = utility_act[i, random.randint(0,99)]
    print(np.mean(random_estimate))



if __name__ == '__main__':

    # evaluate_utility_estimation_effect()

    UTILITY_FUNCTION = utility_unit_revenue # utility_user_count, utility_unit_revenue

    # initialize logger file
    logger = logging.getLogger("evaluation_facebook_music")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('evaluation_results.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    start = time.clock()
    load_group_users_and_csd()
    load_group_costs()
    load_user_item_similarity()
    load_all_items()
    load_item_revenues()
    init_slots(SLOT_NUM)
    initialize_finished = time.clock()
    print 'initialization finished: ', initialize_finished - start, ' seconds'

    # print 'start simulation in parallel'
    # ppservers = ()
    # n_worker = 4
    # job_server = pp.Server(n_worker, ppservers=ppservers) # create 8 processes
    # k_worker = 250
    # # n_worker * k_worker is the number of simulations for each (item, group) pair
    # dependent_funcs = (sh.sim_hit_users, sh.similarity, sh.friends, sh.sim_to_hit_prob, sh.save_hit_users_to_db)
    # jobs = [job_server.submit(sh.simulate_hit_users_monte_carlo,(ALL_ITEMS, GROUP_USERS, alpha, SIM, k_worker, i), dependent_funcs, ("math","random","time","pymysql","logging")) for i in range(8)]
    # # load cache hit users (may delete)
    # for job in jobs:
    #     hit_users = job()

    load_all_simulated_hits()
    simulation_finished = time.clock()
    print 'simulation ended', simulation_finished - initialize_finished, 'seconds'

    # simulate groups and items
    repeat_times = 10

    # test utility difference
    # print 'test utility difference...'
    # load_items()
    # load_groups()
    # test_utility_difference()
    # print 'test utility difference donw'

    # candidate_group_items = []
    # for i in range(repeat_times):
       # candidate_group_items.append((load_groups(), load_items()))

    # for s in [1,]:
    # for bud in [10,]:
    for item_num, group_num in [(20, 20), (40, 40), (60, 60), (80, 80), (100, 100)]:

        #### for varying item and group numbers
        TEST_GROUP_NUM = group_num
        TEST_ITEM_NUM = item_num
        
        if TEST_GROUP_NUM == TOTAL_GROUP_NUM:
             repeat_times = 1
        
        ##### for varying budget
        #BUDGET = bud/10.0
        #load_group_costs() # reload cost for normalization
        
        ##### for varying slots
        # init_slots(s)

        logger.info('=========== new run ==========')

        print 'num of groups:', TEST_GROUP_NUM
        print 'num of items:', TEST_ITEM_NUM
        print 'alpha:', alpha
        print 'budget:', BUDGET
        print 'slots:', SLOT_NUM
        print 'utility:', UTILITY_FUNCTION

        csd_greedy_utilities = {'KP':[], 'AP':[], 'BU':[]}
        as_greedy_utilities = {'KP':[], 'AP':[], 'BU':[]}
        nv_greedy_utilities = {'KP':[], 'AP':[], 'BU':[]}
        tp_greedy_utilities = {'KP':[], 'AP':[], 'BU':[]}
        random_utilities = []
        our_utilities = []

        # for group_item_pair in candidate_group_items:
        for i in range(repeat_times):
            # each time re-select the items and groups
            ITEMS = load_items()
            GROUPS = load_groups()

            # GROUPS = group_item_pair[0]
            # ITEMS = group_item_pair[1]

            print 'GROUPS', GROUPS
            print 'ITEMS', ITEMS

            ###### CSD greedy ###################
            csd_greedy_utilities['KP'].append(evaluate(CSD_KP, "CSD-KP"))
            csd_greedy_utilities['AP'].append(evaluate(CSD_AP, "CSD-AP"))
            csd_greedy_utilities['BU'].append(evaluate(CSD_BU, "CSD-BU"))

            for iii in range(10):
                ###### user number greedy ############
                as_greedy_utilities['KP'].append(evaluate(AS_KP, "AS-KP"))
                as_greedy_utilities['AP'].append(evaluate(AS_AP, "AS-AP"))
                as_greedy_utilities['BU'].append(evaluate(AS_BU, "AS-BU"))

            for iii in range(10):
                ##### network value greedy ########
                nv_greedy_utilities['KP'].append(evaluate(NV_KP, 'NV-KP'))
                nv_greedy_utilities['AP'].append(evaluate(NV_AP, 'NV-AP'))
                nv_greedy_utilities['BU'].append(evaluate(NV_BU, 'NV-BU'))

            ##### tp greedy ########
            tp_greedy_utilities['KP'].append(evaluate(TP_KP, 'TP-KP'))
            tp_greedy_utilities['AP'].append(evaluate(TP_AP, 'TP-AP'))
            tp_greedy_utilities['BU'].append(evaluate(TP_BU, 'TP-BU'))

            for iii in range(10):
                ####### random greedy ###############
                random_utilities.append(evaluate(random_greedy, 'RAN'))

            ###### our method ###################
            our_utilities.append(evaluate(near_opt_group_rec, 'CEIL'))

        csd_kp_result = get_result_str(csd_greedy_utilities['KP'])
        csd_ap_result = get_result_str(csd_greedy_utilities['AP'])
        csd_bu_result = get_result_str(csd_greedy_utilities['BU'])
        as_kp_result = get_result_str(as_greedy_utilities['KP'])
        as_ap_result = get_result_str(as_greedy_utilities['AP'])
        as_bu_result = get_result_str(as_greedy_utilities['BU'])
        nv_kp_result = get_result_str(nv_greedy_utilities['KP'])
        nv_ap_result = get_result_str(nv_greedy_utilities['AP'])
        nv_bu_result = get_result_str(nv_greedy_utilities['BU'])
        tp_kp_result = get_result_str(tp_greedy_utilities['KP'])
        tp_ap_result = get_result_str(tp_greedy_utilities['AP'])
        tp_bu_result = get_result_str(tp_greedy_utilities['BU'])
        rand_result = get_result_str(random_utilities)
        our_result = get_result_str(our_utilities)

        as_result_str = 'AS: {} {} {}'.format(as_kp_result, as_ap_result, as_bu_result)
        logger.info(as_result_str)
        nv_result_str = 'NV: {} {} {}'.format(nv_kp_result, nv_ap_result, nv_bu_result)
        logger.info(nv_result_str)
        tp_result_str = 'TP: {} {} {}'.format(tp_kp_result, tp_ap_result, tp_bu_result)
        logger.info(tp_result_str)
        csd_result_str = 'CSD: {} {} {}'.format(csd_kp_result, csd_ap_result, csd_bu_result)
        logger.info(csd_result_str)
        our_result_str = 'CEIL: {} RAN: {}'.format(our_result, rand_result)
        logger.info(our_result_str)
        parameter_setting = 'groups: {}, items: {}, budget {}, alpha: {}, slots: {}, cost: {}, utility: {}'.format(TEST_GROUP_NUM, TEST_ITEM_NUM, BUDGET, alpha, SLOT_NUM, COST_TYPE, UTILITY_FUNCTION)
        logger.info(parameter_setting)
