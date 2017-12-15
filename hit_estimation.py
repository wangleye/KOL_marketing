import random
import pymysql

conn = pymysql.connect(host='127.0.0.1',
	user='root',
	passwd='123456',
	db='all0504')

DATA_DIR = './facebook'

ITEM_NUM = 50

ITEMS = []
ITEMS_COUNT = {}
USERS = set()
GROUP_USERS = []
USER_ITEMS = {}
SIM = {}

def load_group_users():
	"""
	return groups
	"""
	global GROUP_USERS
	with open("{}/group_users".format(DATA_DIR)) as inputfile:
		for line in inputfile:
			if len(line.strip()) > 0:
				users = line.split()
				GROUP_USERS.append(users)
				USERS.update(users)

def load_items():
	"""
	read item list from file
	"""
	with open("{}/item_list".format(DATA_DIR)) as inputfile:
		for line in inputfile:
			if len(line.strip()) > 0:
				item = line.split()[0]
				item_count = line.split()[1]
				ITEMS.append(item)
				print item_count
				ITEMS_COUNT[item] = int(item_count)
	global ITEMS
	ITEMS = ITEMS[0:ITEM_NUM]


def load_item_similarity():
	"""
	read an item similarity file
	"""
	# global SIM
	# SIM = np.loadtxt('{}/item_similarity'.format(DATA_DIR)) # read item similarity from files
	read_item_similarity_file_line_by_line()

def read_item_similarity_file_line_by_line():
	global SIM
	with open("{}/item_similarity_con_v2_norm".format(DATA_DIR)) as inputfile:
		for line in inputfile:
			line = line.strip()
			if len(line) > 0:
				words = line.split()
				item1 = words[0]
				item2 = words[1]
				similarity = float(words[2])

				if item1 not in SIM:
					SIM[item1] = {}
				SIM[item1][item2] = similarity

def get_similarity(item1, item2):
	if item1 not in SIM or item2 not in SIM[item1]:
		return 0
	return SIM[item1][item2]

def liked_items(user_id):
	if user_id in USER_ITEMS:
		return USER_ITEMS[user_id]
	# read from DB
	query_statement = "SELECT moviestr FROM user WHERE iduser = '{}'".format(user_id)
	x = conn.cursor()
	x.execute(query_statement)
	results = x.fetchall()
	if len(results) == 0:
		# print 'user {} not found in DB!!!'.format(user_id)
		return set()
	else:
		moviestr = results[0][0]
		if len(moviestr.strip())==0:
			USER_ITEMS[user_id] = []
			return list()
		else:
			moviestr_items = moviestr.split(';')
			USER_ITEMS[user_id] = moviestr_items
			return moviestr_items

def get_weight(item):
	"""
	popular items with less weight
	"""
	if item in ITEMS_COUNT:
		return 1.0/(ITEMS_COUNT[item]**0.5) # according to item-based top n recommendation
	else:
		return 0


def similarity(item, user):
	"""
	similarity between an item and a user (a set of items)
	"""
	s = 0.0
	s_w = 0.0
	max_s = 0.0
	prob = 1.0
	user_items = liked_items(user)
	for item2 in user_items:
		s_pair = get_similarity(item, item2) 
		# w = get_weight(item2)
		# s += w * s_pair 
		s += s_pair
		#prob *= (1-s_pair)
		#s_w += w
		# if s_pair > max_s:
		# 	max_s = s_pair
	if s == 0:
		return 0
	else:
		#return s / s_w #get weighted sum
		return s
		#return max_s
		#return 1-prob

def create_training_datset():
	i = 0
	with open('hit_regression/hit_estimation_training_data_sum_con_v2_norm', 'w') as outputfile:
		outputfile.write("y x l\n")
		for rand_item in ITEMS:
			for rand_user in USERS:
				if rand_item in liked_items(rand_user):
					y = 1
				else:
					y = 0
				x = similarity(rand_item, rand_user)
				if x > 0:
					outputfile.write("{} {} {}\n".format(y, x, len(USER_ITEMS[rand_user])))
				i += 1
				if i%1000 == 0:
					print i

if __name__ == '__main__':
	load_items()
	load_item_similarity()
	load_group_users()
	create_training_datset()