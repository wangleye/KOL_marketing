import random
import pymysql
import time
import math
import logging

def save_hit_users_to_db(item, group, hit_users, alpha):
	insert_statement = "INSERT INTO simulate_group_rec (group_id, item_id, hit_users, alpha) VALUES ('{}','{}','{}','{}')".format(group, item, ','.join(hit_users), alpha)
	x = conn.cursor()
	x.execute(insert_statement)

def simulate_hit_users_monte_carlo(items, group_users, sim_dict, K, proc_id):
	global conn, USER_FRIENDS
	conn = pymysql.connect(host='127.0.0.1',
		user='root',
		passwd='123456',
		db='all0504')

	USER_FRIENDS = {}

	rec_pairs = []
	for item in items:
		for group in range(len(group_users)):
			rec_pairs.append((item, group))

	cache_hit_users = {}

	# initialize logger file
	logger = logging.getLogger("simulate_hit_proc{}".format(proc_id))
	logger.setLevel(logging.DEBUG)
	fh = logging.FileHandler('monte_carlo_simulation_parallel.log')
	fh.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	logger.info('staring a new simulation, proc_id: {}, simulation_times: {}, group_num: {}, item_num: {}'.format(proc_id, K, len(group_users), len(items)))

	count_finish_pair = 0
	alpha = 0.02
	for (item, group) in rec_pairs:
		count_finish_pair += 1
		cache_hit_users[(item,group)] = []
		start = time.clock()
		for k in range(K):
			this_hit_users = sim_hit_users(item, group_users[group], sim_dict, alpha)
			cache_hit_users[(item,group)].append(this_hit_users)
			save_hit_users_to_db(item, group, this_hit_users, alpha)
		# commit to db just K times of simulation ends
		try:
			conn.commit()
		except:
			conn.rollback()
		end = time.clock()
		logger.info("({},{}), {} seconds, {} pairs finished".format(item, group, end-start, count_finish_pair))

	return cache_hit_users

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

def sim_to_hit_prob(sim):
	# learned logistic / isonotic function
	return 1.0/(1.0+math.exp(-(-3.559+1.084*sim)))

# store the hit users calculated before
def sim_hit_users(item, users_in_group, sim_dict, alpha):
	"""
	simulation to get hit users for calculating utility
	"""
	hit_users = set()
	share_users = set()
	# hit users in group
	for u in users_in_group:
		sim = similarity(item, u, sim_dict)
		hit_u_p = sim_to_hit_prob(sim)
		if (random.random()<=hit_u_p):
			hit_users.add(u)
			if (random.random()<=alpha):
				share_users.add(u)

	# hit users through social influcence
	while len(share_users) > 0:
		new_share_users = set()
		for u in share_users:
			for f in friends(u):
				sim = similarity(item, f, sim_dict)
				hit_f_p = sim_to_hit_prob(sim)
				if (random.random()<=hit_f_p):
					hit_users.add(f)
					if (random.random()<=alpha):
						new_share_users.add(f)
		share_users = new_share_users
	return hit_users

def similarity(item, user, sim_dict):
	"""
	similarity between an item and a user (a set of items)
	"""
	if user not in sim_dict or item not in sim_dict[user]:
		return 0
	else:
		return sim_dict[user][item]

if __name__ == '__main__':
	items = ["74089565764", "10498014129"]
	group_users = {0:["100000147616782","100000155110661"], 1:["1592423125", "1626196906"]}
	sim_dict = dict()
	simulate_hit_users_monte_carlo(items, group_users, sim_dict, 1000, 99)


