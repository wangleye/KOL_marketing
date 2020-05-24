import random
import numpy as np
import pymysql
import time
import math
import pp
import simulate_hit_pp as sh
import logging
import sys
# from joblib import Parallel, delayed


conn = pymysql.connect(host='127.0.0.1',
    user='root',
    passwd='123456',
    db='all0504')


SIM = {} # dictionary / numpy matrix to store the item similarity matrix
GROUP_USERS = {}
GROUP_CSD = {}
GROUP_TP_RANK = {}
USER_NUM_COSTS = {}
USER_FRIENDS = {}
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

epsilon = 0.1

DATA_DIR = './facebook'

CACHE_UTILITY = {}
CACHE_HIT_USERS = {}

def init_slots(k):
    global SLOT_NUM
    SLOT_NUM = k
    for group in GROUP_USERS:
        SLOTS[group] = k

def load_all_simulated_hits(suffix=''):
    groups = GROUP_USERS.keys()
    items = ALL_ITEMS
    print 'group len:', len(groups), 'item len:', len(items)

    for idx, group in enumerate(groups):
        print idx, group
        for item in items:
            load_simulated_hits(group, item, suffix)

def load_simulated_hits(group_id, item_id, suffix):
    # suffix: '' --> network-based similarity adoption
    # suffix: _only_item --> adoption with item-similairty
    # suffix: _only_friend --> adoption with friendship
    query_str = "select group_id, item_id, hit_users from `simulate_group_rec_{}_{}{}` where group_id = '{}' and item_id = '{}' limit 1000" \
                .format(SCENARIO, "%.2f"%alpha, suffix, group_id, item_id)
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
    with open("{}/{}_list".format(DATA_DIR, SCENARIO)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                item, item_count = line.split()
                ALL_ITEMS.append(item)

    start = 0
    if SCENARIO == 'book':
        start = 50
    ALL_ITEMS = ALL_ITEMS[start:start+TOTAL_ITEM_NUM] # select top N items

def load_item_revenues():
    """
    load revenues of items
    """
    global ITEM_REVENUES
    with open("{}/{}_price".format(DATA_DIR, SCENARIO)) as inputfile:
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

def load_user_item_similarity(suffix):
    """
    read an item similarity file
    """
    global SIM
    global ITEM_FANS
    with open("{}/user_{}_aff_score_100{}".format(DATA_DIR, SCENARIO, suffix)) as inputfile:
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

def load_group_item_costs():
    """
    return group costs from file
    """
    global COSTS
    max_cost = 0
    with open("{}/KOL_number_cost".format(DATA_DIR)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                words = line.strip().split(';')
                group_id = words[0]
                #normalized_cost = float(words[1])/BUDGET
                group_cost = float(words[1])
                USER_NUM_COSTS[group_id] = group_cost
                COSTS[group_id] = {}
                for item_id in ALL_ITEMS:
                    if SCENARIO == 'book':
                        cur_cost = group_cost * ITEM_REVENUES[item_id] if item_id in ITEM_REVENUES else 9.99 # media book price is 9.99
                    else:
                        cur_cost = group_cost
                    COSTS[group_id][item_id] = cur_cost
                    if cur_cost > max_cost:
                        max_cost = cur_cost
    for group_id in COSTS:
        for item_id in COSTS[group_id]:
            COSTS[group_id][item_id] = COSTS[group_id][item_id]/(max_cost*BUDGET) # normalization

    with open("{}/KOL_net_cost".format(DATA_DIR)) as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                words = line.strip().split(';')
                group_id = words[0]
                normalized_cost = float(words[1])/BUDGET
                NET_COSTS[group_id] = normalized_cost

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
            item_r = 8.97
        revenue += len(hit_users[item]) * item_r
    return revenue

def utility_monte_carlo(rec_pairs, use_cache_utility=True, K=0):
    if K == 0:
        K = SIMULATION_TIMES # simulation times
    if len(rec_pairs) == 0:
        return 0

    # if in the utility cache, directly obtain it
    if use_cache_utility and set2key(rec_pairs) in CACHE_UTILITY:
        return CACHE_UTILITY[set2key(rec_pairs)]

    utility_sum = 0

    for i in range(K):
        hit_users = {}
        for (item, group) in rec_pairs:
            hit_users[item] = set() # initialize hit users for any item in recommendations

        for (item, group) in rec_pairs:
            sim_hit_users = random.choice(CACHE_HIT_USERS[set2key((item, group))])
            hit_users[item].update(sim_hit_users)


        # use the HIT_USERS to calculate utility
        utility_sum += UTILITY_FUNCTION(hit_users)
    utility = utility_sum * 1.0 / K

    # save to cache
    if use_cache_utility:
        CACHE_UTILITY[set2key(rec_pairs)] = utility

    return utility

def find_max_utility(groups, items, costs, rho=0):
    max_utility = 0
    for g in groups:
        for i in items:
            if costs[g][i] > 1:
                continue
            recommend_pairs = {(i,g)} # initialize a set containing only one recommendation
            utility_ig = utility_monte_carlo(recommend_pairs)
            #print '({},{}): {}'.format(i,g,utility_ig)
            if utility_ig > max_utility and (rho == 0 or utility_ig*1.0/costs[g][i] >= rho):
                max_utility = utility_ig
    return max_utility

def sum_cost(costs, selected_pairs):
    sum_c = 0
    for (i, g) in selected_pairs:
        sum_c += costs[g][i]
    return sum_c

def find_max_utility_increase(groups, items, costs, slots, selected_pairs):
    remain_cost = 1 - sum_cost(costs, selected_pairs)
    max_utility_increase = 0
    max_pair = None
    max_utility = 0
    for g in groups:
        if not still_has_slot(selected_pairs, slots, g):
            continue
        for i in items:
            if costs[g][i] > remain_cost:
                continue
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
        if not still_has_slot(selected_pairs, slots, g):
            continue
        for i in items:
            if costs[g][i] > remain_cost:
                continue
            recommend_pair = {(i,g)}
            utility_ig = utility_monte_carlo(recommend_pair.union(selected_pairs)) - utility_monte_carlo(selected_pairs)
            utility_ig_per_cost = utility_ig * 1.0 / costs[g][i]
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
        sum_cost += normalized_costs[group][item]
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

def near_opt_group_rec_no_speedup(groups, items, normalized_costs, slots):
    return near_opt_group_rec(groups, items, normalized_costs, slots, speed_up = False)

def near_opt_group_rec_no_local_greedy(groups, items, normalized_costs, slots):
    return near_opt_group_rec(groups, items, normalized_costs, slots, local_greedy = False)

def near_opt_group_rec(groups, items, normalized_costs, slots, speed_up = True, local_greedy = True):
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
        S = set()
        last_utility_increase = {}
        for g in groups:
            for i in items:
                last_utility_increase[(i,g)] = 100000 # set a large utility as the initial value
        while tau  >= epsilon*1.0 / n * M_rho and not is_over_budget(S, normalized_costs):
            # use shuffle to add randomness into the algorithm
            candidates = []
            for g in groups:
                for i in items:
                    candidates.append((i,g))
            # random.shuffle(candidates)
            
            for i, g in candidates:
                # lazy evaluation (speed-up)
                utility_increase_max = last_utility_increase[(i,g)]
                if (i, g) in S or utility_increase_max < tau or utility_increase_max / normalized_costs[g][i] < rho or normalized_costs[g][i] > 1:
                    continue

                S_prime = S.union({(i,g)})
                utility_increase = utility_monte_carlo(S_prime) - utility_monte_carlo(S)
                if speed_up:
                    last_utility_increase[(i,g)] = utility_increase
                if not is_over_slot_constraint(S_prime, slots, g) and \
                utility_increase >= tau and \
                utility_increase / normalized_costs[g][i] >= rho:
                    if is_over_budget(S_prime, normalized_costs):
                        if local_greedy:
                            # local improvement - using S as base, simple/cost greedy
                            S_sg, u_sg = simple_greedy(groups, items, normalized_costs, slots, S)
                            S_cg, u_cg = cost_greedy(groups, items, normalized_costs, slots, S)
                            if u_sg >= u_cg:
                                R[str(rho)] = (S_sg, u_sg)
                            else:
                                R[str(rho)] = (S_cg, u_cg)

                            S_neg_sg, u_neg_sg = simple_greedy(groups, items, normalized_costs, slots, {(i,g)})
                            S_neg_cg, u_neg_cg = cost_greedy(groups, items, normalized_costs, slots, {(i,g)})
                            if u_neg_sg >= u_neg_cg:
                                R[str(-rho)] = (S_neg_sg, u_neg_sg)
                            else:
                                R[str(-rho)] = (S_neg_cg, u_neg_cg)
                        else:
                            R[str(rho)] = (S, utility_monte_carlo(S))
                            R[str(-rho)] = ({(i,g)}, utility_monte_carlo({(i,g)}))
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
            if local_greedy:
                S_sg, u_sg = simple_greedy(groups, items, normalized_costs, slots, S)
                S_cg, u_cg = cost_greedy(groups, items, normalized_costs, slots, S)
                if u_sg >= u_cg:
                    R[str(rho)] = (S_sg, u_sg)
                else:
                    R[str(rho)] = (S_cg, u_cg)
            else:
                R[str(rho)] = (S, utility_monte_carlo(S))
                R[str(-rho)] = (set(), 0)

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

def simple_greedy_baseline(groups, items, normalized_costs, slots):
    return simple_greedy(groups, items, normalized_costs, slots, set())

def simple_greedy(groups, items, normalized_costs, slots, S):
    max_utility = utility_monte_carlo(S)
    new_pair, utility_increase, utility = find_max_utility_increase(groups, items, normalized_costs, slots, S)
    while new_pair is not None:
        S = S.union({new_pair})
        max_utility = utility
        new_pair, utility_increase, utility = find_max_utility_increase(groups, items, normalized_costs, slots, S)
    return S, max_utility

def cost_greedy_baseline(groups, items, normalized_costs, slots):
    return cost_greedy(groups, items, normalized_costs, slots, set())

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

def most_popular_item(input_groups, items, normalized_costs, slots, sort_reference, is_reverse=True):
    S = set()
    groups = input_groups[:]
    groups = sorted(groups, key=lambda x: sort_reference[x], reverse=is_reverse)
    for group in groups: # group id is KOL
        num_fans = 0
        best_item = random.choice(items)
        for item in items: # select the most popular time for the KOL
            if len(ITEM_FANS[item]) > num_fans:
                best_item = item
                num_fans = len(ITEM_FANS[item])
        new_pair, utility_increase, utility = find_max_utility_increase([group,], [best_item,], normalized_costs, slots, S)
        if new_pair != None:
            S = S.union({new_pair})
            max_utility = utility
    return S, max_utility

def most_value_item(input_groups, items, normalized_costs, slots, sort_reference, is_reverse=True):
    S = set()
    groups = input_groups[:]
    groups = sorted(groups, key=lambda x: sort_reference[x], reverse=is_reverse)
    for group in groups: # group id is KOL
        value = 0
        best_item = random.choice(items)
        for item in items: # select the most popular time for the KOL
            tmp_value = len(ITEM_FANS[item]) * (ITEM_REVENUES[item] if item in ITEM_REVENUES else 1) 
            if tmp_value > value:
                best_item = item
                value = tmp_value
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
    if item_select_method == 'POP': # most popular
        return most_popular_item(input_groups, items, normalized_costs, slots, sort_reference, is_reverse)
    if item_select_method == 'VAL': # most value
        return most_value_item(input_groups, items, normalized_costs, slots, sort_reference, is_reverse)

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

def AS_POP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, USER_NUM_COSTS, True, 'POP')

def AS_VAL(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, USER_NUM_COSTS, True, 'VAL')

def NV_KP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, NET_COSTS, True, 'KP')

def NV_AP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, NET_COSTS, True, 'AP')

def NV_BU(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, NET_COSTS, True, 'BU')

def NV_POP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, NET_COSTS, True, 'POP')

def NV_VAL(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, NET_COSTS, True, 'VAL')

def TP_KP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, GROUP_TP_RANK, False, 'KP')

def TP_AP(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, GROUP_TP_RANK, False, 'AP')

def TP_BU(input_groups, items, normalized_costs, slots):
    return benchmark(input_groups, items, normalized_costs, slots, GROUP_TP_RANK, False, 'BU')

# for simulating final results
def simulate_final_utility(rec_pairs, simulation_times=1000):
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
    real_result = 0
    simulation_result = 0
    count = 0.0
    for result in result_array:
        real_result += result[0]
        simulation_result += result[1]
        count += 1.0
    result_str = str(real_result/count) + '(' + str(simulation_result/count) + ')'
    # result_str = str(np.mean(np.asarray(result_array)))
    return result_str

def evaluate(test_method, method_name):
    started = time.clock()
    results = test_method(GROUPS, ITEMS, COSTS, SLOTS)
    simulation_result = utility_monte_carlo(results[0], K=1000, use_cache_utility=False) # simulation enough times again to get result
    # simulation_result = float(results[1])
    real_result = simulate_final_utility(results[0], simulation_times=1000)
    print method_name, ":", results, real_result
    # logger.info('CSD greedy: {}'.format(results))
    finished = time.clock()
    print finished - started, ' seconds'
    return real_result, simulation_result


def test_utility_difference(group_sizes, suffix):
    uitlity_est_vector = []
    utility_act_vector = []

    if 1 in group_sizes:
        group_sizes.remove(1)
        for item in ITEMS:
            print item
            for group in GROUPS:
                utility_est = utility_monte_carlo([(item, group),])
                utility_act = simulate_final_utility([(item, group),])
                uitlity_est_vector.append(utility_est)
                utility_act_vector.append(utility_act)

    for each_size in group_sizes:
        for i in range(10000): # random sample times
            if i%100 == 0:
                print each_size, i
            selected_pairs = []
            for _ in range(each_size):
                item = random.choice(ITEMS)
                group = random.choice(GROUPS)
                selected_pairs.append((item, group))
            utility_est = utility_monte_carlo(selected_pairs)
            utility_act = simulate_final_utility(selected_pairs)
            uitlity_est_vector.append(utility_est)
            utility_act_vector.append(utility_act)

    uitlity_est_vector = np.array(uitlity_est_vector)
    utility_act_vector = np.array(utility_act_vector)

    # print 'mean utility_est_vector', np.mean(uitlity_est_vector)
    # print 'mean utility_act_vector', np.mean(utility_act_vector)
    # print 'mean difference', np.mean(np.abs(uitlity_est_vector-utility_act_vector))

    np.savetxt('estimate_utility_{}_{}{}'.format(SCENARIO, alpha, suffix), uitlity_est_vector)
    np.savetxt('actual_utility_{}_{}{}'.format(SCENARIO, alpha, suffix), utility_act_vector)


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
    
    TOTAL_GROUP_NUM = 100
    TOTAL_ITEM_NUM = 50
    SIMULATION_TIMES = 500

    UTILITY_FUNCTION = utility_unit_revenue # utility_user_count, utility_unit_revenue
    SCENARIO = 'book' # book or movie
    alpha = 0.02
    p = 1
    need_simulation = False
    TEST_ITEM_NUM = TOTAL_ITEM_NUM
    TEST_GROUP_NUM = int(TOTAL_GROUP_NUM * p)
    suffix = '' # adoption model
    # suffix: '_both' --> network-based similarity adoption (both item and friend)
    # suffix: _only_item --> adoption with item-similairty
    # suffix: _only_friend --> adoption with friendship


    # initialize logger file
    logger = logging.getLogger("evaluation_facebook")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('evaluation_results_{}_alpha.log'.format(SCENARIO))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    start = time.clock()
    load_group_users_and_csd()
    load_all_items()
    if SCENARIO == 'book':
        load_item_revenues()
    load_group_item_costs()
    load_user_item_similarity(suffix)
    

    init_slots(SLOT_NUM)
    initialize_finished = time.clock()
    print 'initialization finished: ', initialize_finished - start, ' seconds'

    if need_simulation:
        print 'start simulation in parallel'
        ppservers = ()
        n_worker = 4
        job_server = pp.Server(n_worker, ppservers=ppservers) # create multiple processes
        k_worker = int(SIMULATION_TIMES/n_worker)
        # n_worker * k_worker is the number of simulations for each (item, group) pair
        dependent_funcs = (sh.sim_hit_users, sh.similarity, sh.friends, sh.sim_to_hit_prob, sh.save_hit_users_to_db)
        jobs = [job_server.submit(sh.simulate_hit_users_monte_carlo,(ALL_ITEMS, GROUP_USERS, SCENARIO, alpha, SIM, k_worker, i, suffix),\
                dependent_funcs, ("math","random","time","pymysql","logging")) for i in range(n_worker)]
        # load cache hit users (may delete)
        for job in jobs:
            hit_users = job()

        simulation_finished = time.clock()
        print 'simulation ended', simulation_finished - initialize_finished, 'seconds'
        sys.exit()
    
  
    load_all_simulated_hits(suffix)
    
    # # test utility difference
    # print 'test utility difference...'
    # load_items()
    # load_groups()
    # test_utility_difference([1,2,3,4], suffix)
    # print 'test utility difference done'
    # sys.exit()

    candidate_group_items = []
    repeat_times = 5
    random.seed(1537) # ensure reproductivity for the loaded groups and items
    for i in range(repeat_times):
        candidate_group_items.append((load_groups(), load_items()))

    for s in [1, ]:
    # for bud in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    # for item_num, group_num in [(TOTAL_ITEM_NUM, 20), (TOTAL_ITEM_NUM, 40), (TOTAL_ITEM_NUM, 60), (TOTAL_ITEM_NUM, 80), (TOTAL_ITEM_NUM, 100)]:
        #### for varying item and group numbers
        # TEST_GROUP_NUM = group_num
        # TEST_ITEM_NUM = item_num
        
        #### for varying budget
        # BUDGET = bud
        # load_group_costs() # reload cost for normalization
        
        ##### for varying slots
        init_slots(s)

        logger.info('=========== new run ==========')

        print 'num of groups:', TEST_GROUP_NUM
        print 'num of items:', TEST_ITEM_NUM
        print 'alpha:', alpha
        print 'budget:', BUDGET
        print 'slots:', SLOT_NUM
        print 'utility:', UTILITY_FUNCTION

        csd_greedy_utilities = {'KP':[], 'AP':[], 'BU':[]}
        as_greedy_utilities = {'KP':[], 'AP':[], 'BU':[], 'POP':[], 'VAL':[]}
        nv_greedy_utilities = {'KP':[], 'AP':[], 'BU':[], 'POP':[], 'VAL':[]}
        tp_greedy_utilities = {'KP':[], 'AP':[], 'BU':[]}
        random_utilities = []
        s_greedy_utilities = []
        c_greedy_utilities = []
        our_no_local_greedy_utilities = []
        our_utilities = []

        for group_item_pair in candidate_group_items:
        #for i in range(repeat_times):
            # each time re-select the items and groups
            # ITEMS = load_items()
            # GROUPS = load_groups()

            GROUPS = group_item_pair[0]
            ITEMS = group_item_pair[1]

            print 'GROUPS', GROUPS
            print 'ITEMS', ITEMS

            ###### CSD greedy ###################
            for iii in range(10):
                csd_greedy_utilities['KP'].append(evaluate(CSD_KP, "CSD-KP"))
            csd_greedy_utilities['AP'].append(evaluate(CSD_AP, "CSD-AP"))
            csd_greedy_utilities['BU'].append(evaluate(CSD_BU, "CSD-BU"))

            ###### user number greedy ############
            for iii in range(10):
                as_greedy_utilities['KP'].append(evaluate(AS_KP, "AS-KP"))
            as_greedy_utilities['AP'].append(evaluate(AS_AP, "AS-AP"))
            as_greedy_utilities['BU'].append(evaluate(AS_BU, "AS-BU"))
            as_greedy_utilities['POP'].append(evaluate(AS_POP, "AS-POP"))
            as_greedy_utilities['VAL'].append(evaluate(AS_VAL, "AS-VAL"))

            ##### network value greedy ########
            for iii in range(10):
                nv_greedy_utilities['KP'].append(evaluate(NV_KP, 'NV-KP'))
            nv_greedy_utilities['AP'].append(evaluate(NV_AP, 'NV-AP'))
            nv_greedy_utilities['BU'].append(evaluate(NV_BU, 'NV-BU'))
            nv_greedy_utilities['POP'].append(evaluate(NV_POP, 'NV-POP'))
            nv_greedy_utilities['VAL'].append(evaluate(NV_VAL, 'NV-VAL'))

            ##### tp greedy ########
            for iii in range(10):
                tp_greedy_utilities['KP'].append(evaluate(TP_KP, 'TP-KP'))
            tp_greedy_utilities['AP'].append(evaluate(TP_AP, 'TP-AP'))
            tp_greedy_utilities['BU'].append(evaluate(TP_BU, 'TP-BU'))

            # for iii in range(10):
                ####### random greedy ###############
                # random_utilities.append(evaluate(random_greedy, 'RAN'))

            ###### simple & cost greedy ####
            s_greedy_utilities.append(evaluate(simple_greedy_baseline, 'S-Greedy'))
            c_greedy_utilities.append(evaluate(cost_greedy_baseline, 'C-Greedy'))

            ###### our method ###################
            # CACHE_UTILITY = {}
            # evaluate(near_opt_group_rec_no_speedup, 'CEIL no lazy evaluation') # just for time test experiment
            # CACHE_UTILITY = {}
            our_no_local_greedy_utilities.append(evaluate(near_opt_group_rec_no_local_greedy, 'CEIL (No Local Greedy)'))
            # CACHE_UTILITY = {}
            our_utilities.append(evaluate(near_opt_group_rec, 'CEIL'))

        csd_kp_result = get_result_str(csd_greedy_utilities['KP'])
        csd_ap_result = get_result_str(csd_greedy_utilities['AP'])
        csd_bu_result = get_result_str(csd_greedy_utilities['BU'])
        as_kp_result = get_result_str(as_greedy_utilities['KP'])
        as_ap_result = get_result_str(as_greedy_utilities['AP'])
        as_bu_result = get_result_str(as_greedy_utilities['BU'])
        as_pop_result = get_result_str(as_greedy_utilities['POP'])
        as_val_result = get_result_str(as_greedy_utilities['VAL'])
        nv_kp_result = get_result_str(nv_greedy_utilities['KP'])
        nv_ap_result = get_result_str(nv_greedy_utilities['AP'])
        nv_bu_result = get_result_str(nv_greedy_utilities['BU'])
        nv_pop_result = get_result_str(nv_greedy_utilities['POP'])
        nv_val_result = get_result_str(nv_greedy_utilities['VAL'])
        tp_kp_result = get_result_str(tp_greedy_utilities['KP'])
        tp_ap_result = get_result_str(tp_greedy_utilities['AP'])
        tp_bu_result = get_result_str(tp_greedy_utilities['BU'])
        s_greedy_result = get_result_str(s_greedy_utilities)
        c_greedy_result = get_result_str(c_greedy_utilities)
        # rand_result = get_result_str(random_utilities)
        our_no_local_greedy_result = get_result_str(our_no_local_greedy_utilities)
        our_result = get_result_str(our_utilities)

        as_result_str = 'AS: {} {} {} {} {}'.format(as_kp_result, as_ap_result, as_bu_result, as_pop_result, as_val_result)
        logger.info(as_result_str)
        nv_result_str = 'NV: {} {} {} {} {}'.format(nv_kp_result, nv_ap_result, nv_bu_result, nv_pop_result, nv_val_result)
        logger.info(nv_result_str)
        tp_result_str = 'TP: {} {} {}'.format(tp_kp_result, tp_ap_result, tp_bu_result)
        logger.info(tp_result_str)
        csd_result_str = 'CSD: {} {} {}'.format(csd_kp_result, csd_ap_result, csd_bu_result)
        logger.info(csd_result_str)
        # random_result_str = 'RAN: {}'.format(rand_result)
        # logger.info(random_result_str)
        greedy_result_str = 'S-Greedy: {} C-Greedy: {}'.format(s_greedy_result, c_greedy_result)
        logger.info(greedy_result_str)
        our_result_str = 'CEIL: {} CEIL (No Local Greedy): {}'.format(our_result, our_no_local_greedy_result)
        logger.info(our_result_str)
        parameter_setting = 'groups: {}, items: {}, budget {}, alpha: {}, slots: {}, utility: {}, simulation: {}, adoption_suffix: {}'\
            .format(TEST_GROUP_NUM, TEST_ITEM_NUM, BUDGET, alpha, SLOT_NUM, UTILITY_FUNCTION.__name__, SIMULATION_TIMES, suffix)
        logger.info(parameter_setting)
