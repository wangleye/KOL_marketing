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

TOTAL_GROUP_NUM = 50

TEST_ITEM_NUM = 100 # the number of items used in the test
TEST_GROUP_NUM = 50 # the number of groups used in the test

SIM = {} # dictionary / numpy matrix to store the item similarity matrix
GROUP_USERS = {}
GROUP_CSD = {}
USER_NUM_COSTS = {}
NET_COSTS = {}
COSTS = {}
GROUPS = []
ITEMS = []
ITEMS_COUNT = {}
USER_ITEMS = {}
USER_FRIENDS = {}
SLOTS = {}
BUDGET = 3
COST_TYPE = 'number' # 'net' or 'number'

alpha = 0.02
epsilon = 0.1

DATA_DIR = './facebook'

CACHE_UTILITY = {}
CACHE_HIT_USERS = {}

def init_slots(k=1):
	for group in GROUP_USERS:
		SLOTS[group] = k

def load_simulated_hits():
	query_str = "select group_id, item_id, hit_users from simulate_group_rec where alpha between {} and {} order by group_id, item_id".format(alpha-0.001, alpha+0.001)
	x = conn.cursor()
	x.execute(query_str)
	results = x.fetchall()
	K = 1000  # current MC simulation is conducted for 1000 times
	for i in range(K):
		CACHE_HIT_USERS[i] = {}
	last_rec_pair = (-1,-1)
	for each_result in results:
		item = str(each_result[1])
		group = int(each_result[0])
		hit_users = set(str(each_result[2]).split(','))
		rec_pair = (item,group)
		if rec_pair != last_rec_pair:
			j = 0
			last_rec_pair = rec_pair
		CACHE_HIT_USERS[j%K][rec_pair] = hit_users # j%K to prevent some pairs with more than K simulations
		j += 1

def load_groups():
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
	with open("{}/user_item_aff_score_100_item_50000_relation_KOL".format(DATA_DIR)) as inputfile:
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

				if user not in SIM:
					SIM[user] = {}
				SIM[user][item] = similarity

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
	K = len(CACHE_HIT_USERS)
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
			hit_users[item].update(CACHE_HIT_USERS[i][(item, group)])

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

def still_has_slot(selected_recs, slots, i):
	"""
	judge whether the i-th group still has slot
	"""
	max_slot = slots[i]
	current_slot = 0
	for (item, group) in selected_recs:
		if group == i:
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
		if group == i:
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

	print 'start simulation in parallel'
	ppservers = ()
	n_worker = 8
	job_server = pp.Server(n_worker, ppservers=ppservers) # create 8 processes
	k_worker = 125
	# n_worker * k_worker is the number of simulations for each (item, group) pair
	dependent_funcs = (sh.sim_hit_users, sh.similarity, sh.friends, sh.sim_to_hit_prob, sh.save_hit_users_to_db)
	jobs = [job_server.submit(sh.simulate_hit_users_monte_carlo,(ITEMS, GROUP_USERS, alpha, SIM, k_worker, i), dependent_funcs, ("math","random","time","pymysql","logging")) for i in range(8)]
	for i in range(n_worker*k_worker):
		CACHE_HIT_USERS[i] = {}
	i = 0
	for job in jobs:
		hit_users = job()
		for rec_pair in hit_users:
			for j in range(i*k_worker, (i+1)*k_worker):
				CACHE_HIT_USERS[j][rec_pair] = hit_users[rec_pair][j-i*k_worker]
		i += 1

	print 'load simulation user hits'
	load_simulated_hits()
	simulation_finished = time.clock()
	print 'simulation ended', simulation_finished - initialize_finished, 'seconds'

	# simulate groups and items
	repeat_times = 5
	# candidates = []
	# for i in range(repeat_times):
	# 	load_items()
	# 	load_groups()
	# 	candidates.append((list(ITEMS), list(GROUPS)))

	# for bud in [1.5, 2.0, 2.5]:
	# for s in [1, ]:
	for item_num, group_num in [(20,10), (40, 20), (60, 30), (80, 40), (100, 50)]:

		#### for varying item and group numbers
		TEST_GROUP_NUM = group_num
		TEST_ITEM_NUM = item_num
		if item_num == 100 and group_num == TOTAL_GROUP_NUM:
			repeat_times = 1
		
		##### for varying budget
		# BUDGET = bud
		# load_group_costs() # reload cost for normalization
		
		##### for varying slots
		# SLOTS = np.ones(TOTAL_GROUP_NUM) * s

		logger.info('=========== new run ==========')

		print 'num of groups:', TEST_GROUP_NUM
		print 'num of items:', TEST_ITEM_NUM
		print 'alpha:', alpha
		print 'budget:', BUDGET
		print 'slots:', SLOTS[0]
		print 'utility:', UTILITY_FUNCTION

		csd_greedy_utilities = []
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
			print 'CSD greedy:', csd_greedy_results
			logger.info('CSD greedy: {}'.format(csd_greedy_results))
			csd_finished = time.clock()
			print csd_finished - simulation_finished, ' seconds'
			csd_greedy_utilities.append(csd_greedy_results[1])

			####### random greedy ###############
			for iii in range(10):
				random_results = random_greedy(GROUPS, ITEMS, COSTS, SLOTS, set())
				print 'random greedy:', random_results
				logger.info('random greedy: {}'.format(random_results))
				rg_finished = time.clock()
				print rg_finished - csd_finished, ' seconds'
				random_utilities.append(random_results[1])

			###### simple greedy ###################
			simple_greedy_results = simple_greedy(GROUPS, ITEMS, COSTS, SLOTS, set())
			print 'simple greedy:', simple_greedy_results
			logger.info('simple greedy: {}'.format(simple_greedy_results))
			sg_finished = time.clock()
			print sg_finished - rg_finished, ' seconds'
			simple_greedy_utilities.append(simple_greedy_results[1])

			###### cost greedy ###################
			cost_greedy_results = cost_greedy(GROUPS, ITEMS, COSTS, SLOTS, set())
			print 'cost greedy:', cost_greedy_results
			logger.info('cost greedy: {}'.format(cost_greedy_results))
			cg_finished = time.clock()
			print cg_finished - sg_finished, ' seconds'
			cost_greedy_utilities.append(cost_greedy_results[1])

			###### our method ###################
			our_method_results = near_opt_group_rec(GROUPS, ITEMS, COSTS, SLOTS)
			print 'our method:', our_method_results
			logger.info('our method: {}'.format(our_method_results))
			our_finished = time.clock()
			print our_finished - cg_finished, ' seconds'
			our_utilities.append(our_method_results[1])

		csd_result = np.mean(np.asarray(csd_greedy_utilities))
		rand_result = np.mean(np.asarray(random_utilities))
		sg_result = np.mean(np.asarray(simple_greedy_utilities))
		cg_result = np.mean(np.asarray(cost_greedy_utilities))
		our_result = np.mean(np.asarray(our_utilities))
		result_str = 'random: {}, csd: {}, simple: {}, cost: {}, our: {}'.format(rand_result, csd_result, sg_result, cg_result, our_result)
		# result_str = 'random: {}, csd: {}'.format(rand_result, csd_result)
		print result_str
		parameter_setting = 'groups: {}, items: {}, budget {}, alpha: {}, slots: {}, cost: {}, utility: {}'.format(TEST_GROUP_NUM, TEST_ITEM_NUM, BUDGET, alpha, SLOTS[0], COST_TYPE, UTILITY_FUNCTION)
		logger.info(parameter_setting)
		logger.info(result_str)
