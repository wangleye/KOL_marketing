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

TEST_ITEM_NUM = 100 # the number of items used in the test
TEST_GROUP_NUM = 100 # the number of groups used in the test

SIM = {} # dictionary / numpy matrix to store the item similarity matrix
GROUP_USERS = {}
GROUP_CSD = {}
USER_NUM_COSTS = {}
NET_COSTS = {}
COSTS = {}
GROUPS = []
ITEMS = []
ITEMS_COUNT = {}
ITEM_FANS = {}
USER_FRIENDS = {}
SLOTS = {}
SLOT_NUM = 1
BUDGET = 1
COST_TYPE = 'number' # 'net' or 'number'

alpha = 0.04
epsilon = 0.1

DATA_DIR = './facebook'

CACHE_UTILITY = {}
CACHE_HIT_USERS = {}

def init_slots(k=SLOT_NUM):
	for group in GROUP_USERS:
		SLOTS[group] = k

def load_all_simulated_hits():
	groups = GROUP_USERS.keys()
	items = ITEMS
	print 'group len:', len(groups), 'item len:', len(items)
	for idx, group in enumerate(groups):
		print idx, group
		for item in items:
			load_simulated_hits(group, item)


	# limit_lowerbound = 0
	# step = 100000
	# while True:
	# 	query_str = "select group_id, item_id, hit_users from simulate_group_rec where alpha between {} and {} limit {}, {}".format(alpha-0.001, alpha+0.001, limit_lowerbound, step)
	# 	print(query_str)
	# 	limit_lowerbound += step
	# 	x = conn.cursor()
	# 	x.execute(query_str)
	# 	results = x.fetchall()
	# 	count = 0
	# 	for each_result in results:
	# 		item = str(each_result[1])
	# 		group = str(each_result[0])
	# 		hit_users = set(str(each_result[2]).split(','))
	# 		key_value = set2key((item, group))
	# 		if key_value not in CACHE_HIT_USERS:
	# 			CACHE_HIT_USERS[key_value] = []
	# 		if len(CACHE_HIT_USERS[key_value]) < K: # at most load K simulation results
	# 			CACHE_HIT_USERS[key_value].append(hit_users)
	# 	if len(results) < step:
	# 		print("finish loading!")
	# 		break

def load_simulated_hits(group_id, item_id):
	if set2key((item_id, group_id)) in CACHE_HIT_USERS:
		return CACHE_HIT_USERS[set2key((item_id, group_id))]
	query_str = "select group_id, item_id, hit_users from simulate_group_rec where alpha between {} and {} and group_id = '{}' and item_id = '{}' limit 1000".format(alpha-0.001, alpha+0.001, group_id, item_id)
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
	return hit_users_list

def load_groups():
	global GROUPS
	GROUPS = random.sample(GROUP_USERS.keys(), TEST_GROUP_NUM)

def load_items():
	"""
	read item list from file
	"""
	global ITEMS
	with open("{}/item_list".format(DATA_DIR)) as inputfile:
		for line in inputfile:
			if len(line.strip()) > 0:
				item, item_count = line.split()
				ITEMS.append(item)
				ITEMS_COUNT[item] = int(item_count)
	ITEMS_TOP_100 = ITEMS[0:100]
	ITEMS = random.sample(ITEMS_TOP_100, TEST_ITEM_NUM)

def load_user_item_similarity():
	"""
	read an item similarity file
	"""
	read_user_item_similarity_file_line_by_line()


def read_user_item_similarity_file_line_by_line():
	global SIM
	global ITEM_FANS
	with open("{}/user_item_aff_score_100_item_only_KOL_complete".format(DATA_DIR)) as inputfile:
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
	with open("{}/KOL_audience".format(DATA_DIR)) as inputfile:
		for line in inputfile:
			if len(line.strip()) > 0:
				words = line.strip().split(';')
				group_id = words[0]
				users = words[1].split()
				GROUP_USERS[group_id] = users

	with open("{}/KOL_CSD".format(DATA_DIR)) as inputfile:
		for line in inputfile:
			if len(line.strip()) > 0:
				words = line.strip().split(';')
				group_id = words[0]
				csd = float(words[1])
				GROUP_CSD[group_id] = csd

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
		revenue += len(hit_users[item])
	return revenue

UTILITY_FUNCTION = utility_user_count # utility_user_count, utility_unit_revenue

def utility_monte_carlo(rec_pairs):
	K = 10000 # simulation times
	if len(rec_pairs) == 0:
		return 0

	# if in the utility cache, directly obtain it
	if set2key(rec_pairs) in CACHE_UTILITY:
		return CACHE_UTILITY[set2key(rec_pairs)]

	utility_sum = 0
	cache_hit_users = {}
	for (item, group) in rec_pairs:
		cache_hit_users[set2key((item, group))] = load_simulated_hits(group, item)

	for i in range(K):
		hit_users = {}
		for (item, group) in rec_pairs:
			hit_users[item] = set() # initialize hit users for any item in recommendations

		for (item, group) in rec_pairs:
			sim_hit_users = random.choice(cache_hit_users[set2key((item, group))])
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

def random_greedy(input_groups, items, normalized_costs, slots, S):
	groups = input_groups[:]
	max_utility = utility_monte_carlo(S)
	rand_group = random.choice(groups)
	groups.remove(rand_group)
	new_pair, utility_increase, utility = find_max_utility_increase([rand_group,], items, normalized_costs, slots, S)
	while len(groups) > 0: # test all the groups until no one can be added
		if new_pair != None: # sometimes rand_group's cost is too high, then new_pair will be None, but we still can find more groups
			S = S.union({new_pair})
			max_utility = utility
		rand_group = random.choice(groups)
		groups.remove(rand_group)
		new_pair, utility_increase, utility = find_max_utility_increase([rand_group,], items, normalized_costs, slots, S)
	return S, max_utility

def CSD_greedy(input_groups, items, normalized_costs, slots, S):
	return baseline_greedy(input_groups, items, normalized_costs, slots, S, GROUP_CSD)

def baseline_greedy(input_groups, items, normalized_costs, slots, S, sort_reference):
	groups = input_groups[:]
	groups = sorted(groups, key=lambda x:sort_reference[x], reverse=True)
	max_utility = utility_monte_carlo(S)
	for group in groups:
		new_pair, utility_increase, utility = find_max_utility_increase([group,], items, normalized_costs, slots, S)
		if new_pair != None: 
			S = S.union({new_pair})
			max_utility = utility
	return S, max_utility

def user_num_greedy(input_groups, items, normalized_costs, slots, S):
	return baseline_greedy(input_groups, items, normalized_costs, slots, S, USER_NUM_COSTS)

def network_value_greedy(input_groups, items, normalized_costs, slots, S):
	return baseline_greedy(input_groups, items, normalized_costs, slots, S, NET_COSTS)

# for simulating final results
def simulate_final_utility(rec_pairs, simulation_times = 10000):
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
					if (random.random()<=alpha):
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

if __name__ == '__main__':

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
	load_items()
	load_groups()
	init_slots()
	initialize_finished = time.clock()
	print 'initialization finished: ', initialize_finished - start, ' seconds'

	# print 'start simulation in parallel'
	# ppservers = ()
	# n_worker = 4
	# job_server = pp.Server(n_worker, ppservers=ppservers) # create 8 processes
	# k_worker = 250
	# # n_worker * k_worker is the number of simulations for each (item, group) pair
	# dependent_funcs = (sh.sim_hit_users, sh.similarity, sh.friends, sh.sim_to_hit_prob, sh.save_hit_users_to_db)
	# jobs = [job_server.submit(sh.simulate_hit_users_monte_carlo,(ITEMS, GROUP_USERS, alpha, SIM, k_worker, i), dependent_funcs, ("math","random","time","pymysql","logging")) for i in range(8)]

	## load cache hit users (may delete)
	# for i in range(n_worker*k_worker):
	# 	CACHE_HIT_USERS[i] = {}
	# i = 0
	# for job in jobs:
	# 	hit_users = job()
	# 	for rec_pair in hit_users:
	# 		for j in range(i*k_worker, (i+1)*k_worker):
	# 			CACHE_HIT_USERS[j][rec_pair] = hit_users[rec_pair][j-i*k_worker]
	# 	i += 1

	load_all_simulated_hits()
	simulation_finished = time.clock()
	print 'simulation ended', simulation_finished - initialize_finished, 'seconds'

	# simulate groups and items
	repeat_times = 10
	# candidates = []
	# for i in range(repeat_times):
	# 	load_items()
	# 	load_groups()
	# 	candidates.append((list(ITEMS), list(GROUPS)))

	# for bud in [1.5, 2.0, 2.5]:
	# for s in [1, ]:
	for item_num, group_num in [(100, 100)]:

		#### for varying item and group numbers
		TEST_GROUP_NUM = group_num
		TEST_ITEM_NUM = item_num
		
		if TEST_ITEM_NUM == 100 and TEST_GROUP_NUM == TOTAL_GROUP_NUM:
		 	repeat_times = 1
		
		##### for varying budget
		# BUDGET = bud
		# load_group_costs() # reload cost for normalization
		
		##### for varying slots
		# init_slots(s)

		logger.info('=========== new run ==========')

		print 'num of groups:', TEST_GROUP_NUM
		print 'num of items:', TEST_ITEM_NUM
		print 'alpha:', alpha
		print 'budget:', BUDGET
		print 'slots:', SLOT_NUM
		print 'utility:', UTILITY_FUNCTION

		csd_greedy_utilities = []
		user_number_greedy_utilities = []
		network_value_greedy_utilities = []
		random_utilities = []
		simple_greedy_utilities = []
		cost_greedy_utilities = []
		our_utilities = []
		
		# for it, gr in candidates:
		# 	GROUPS = gr
		# 	ITEMS = it

		for i in range(repeat_times):
			# each time re-select the items and groups
			load_items()
			load_groups()

			print 'GROUPS', GROUPS
			print 'ITEMS', ITEMS

			###### CSD greedy ###################
			csd_greedy_results = CSD_greedy(GROUPS, ITEMS, COSTS, SLOTS, set())
			real_result = simulate_final_utility(csd_greedy_results[0])
			print 'CSD greedy:', csd_greedy_results, real_result
			# logger.info('CSD greedy: {}'.format(csd_greedy_results))
			csd_finished = time.clock()
			print csd_finished - simulation_finished, ' seconds'
			csd_greedy_utilities.append(real_result)

			###### user number greedy ############
			user_num_greedy_results = user_num_greedy(GROUPS, ITEMS, COSTS, SLOTS, set())
			real_result = simulate_final_utility(user_num_greedy_results[0])
			print 'user number greedy:', user_num_greedy_results, real_result
			# logger.info('user number greedy: {}'.format(user_num_greedy_results))
			ung_finished = time.clock()
			print ung_finished - csd_finished, ' seconds'
			user_number_greedy_utilities.append(real_result)

			##### network value greedy ########
			network_value_greedy_results = network_value_greedy(GROUPS, ITEMS, COSTS, SLOTS, set())
			real_result = simulate_final_utility(network_value_greedy_results[0])
			print 'network value greedy:', network_value_greedy_results, real_result
			# logger.info('network value greedy: {}'.format(network_value_greedy_results))
			nvg_finished = time.clock()
			print nvg_finished - ung_finished, ' seconds'
			network_value_greedy_utilities.append(real_result)			

			####### random greedy ###############
			for iii in range(5):
				random_results = random_greedy(GROUPS, ITEMS, COSTS, SLOTS, set())
				real_result = simulate_final_utility(random_results[0])
				print 'random greedy:', random_results, real_result
				# logger.info('random greedy: {}'.format(random_results))
				rg_finished = time.clock()
				print rg_finished - nvg_finished, ' seconds'
				random_utilities.append(real_result)

			###### simple greedy ###################
			simple_greedy_results = simple_greedy(GROUPS, ITEMS, COSTS, SLOTS, set())
			real_result = simulate_final_utility(simple_greedy_results[0])
			print 'simple greedy:', simple_greedy_results, real_result
			# logger.info('simple greedy: {}'.format(simple_greedy_results))
			sg_finished = time.clock()
			print sg_finished - rg_finished, ' seconds'
			simple_greedy_utilities.append(real_result)

			###### cost greedy ###################
			cost_greedy_results = cost_greedy(GROUPS, ITEMS, COSTS, SLOTS, set())
			real_result = simulate_final_utility(cost_greedy_results[0])
			print 'cost greedy:', cost_greedy_results, real_result
			# logger.info('cost greedy: {}'.format(cost_greedy_results))
			cg_finished = time.clock()
			print cg_finished - sg_finished, ' seconds'
			cost_greedy_utilities.append(real_result)

			###### our method ###################
			our_method_results = near_opt_group_rec(GROUPS, ITEMS, COSTS, SLOTS)
			real_result = simulate_final_utility(our_method_results[0])
			print 'our method:', our_method_results, real_result
			# logger.info('our method: {}'.format(our_method_results))
			our_finished = time.clock()
			print our_finished - cg_finished, ' seconds'
			our_utilities.append(real_result)

		csd_result = get_result_str(csd_greedy_utilities)
		ung_result = get_result_str(user_number_greedy_utilities)
		nvg_result = get_result_str(network_value_greedy_utilities)
		rand_result = get_result_str(random_utilities)
		sg_result = get_result_str(simple_greedy_utilities)
		cg_result = get_result_str(cost_greedy_utilities)
		our_result = get_result_str(our_utilities)

		result_str = 'random: {}, user number: {}, net value: {}, csd: {}, simple: {}, cost: {}, our: {}'.format(rand_result, ung_result, nvg_result, csd_result, sg_result, cg_result, our_result)
		# result_str = 'random: {}, csd: {}'.format(rand_result, csd_result)
		print result_str
		parameter_setting = 'groups: {}, items: {}, budget {}, alpha: {}, slots: {}, cost: {}, utility: {}'.format(TEST_GROUP_NUM, TEST_ITEM_NUM, BUDGET, alpha, SLOT_NUM, COST_TYPE, UTILITY_FUNCTION)
		logger.info(parameter_setting)
		logger.info(result_str)
		logger.info("{} {} {} {} {}".format(rand_result, ung_result, nvg_result, csd_result, our_result))