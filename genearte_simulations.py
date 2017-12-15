import random
import numpy as np

GROUP_NUM = 100
USER_NUM = 1000
ITEM_NUM = 100

USER_GROUP_MIN = 10
USER_GROUP_MAX = 50

ITEM_USER_MIN = 2
ITEM_USER_MAX = 10

COST_GROUP_MIN = 0.1
COST_GROUP_MAX = 0.5

def simulate_item_similarity():
	item_sim_matrix = np.zeros((ITEM_NUM, ITEM_NUM))
	for i in range(ITEM_NUM):
		for j in range(ITEM_NUM):
			if i >= j:
				continue
			else:
				item_sim_matrix[i,j] = random.random()
				item_sim_matrix[j,i] = item_sim_matrix[i,j] # simulate a symmetric similarity matrix
	np.savetxt('./simulation/item_similarity', item_sim_matrix)

def simulate_group_users():
	with open ('./simulation/group_users', 'w') as output:
		for i in range(GROUP_NUM):
			group_user_num = random.randint(USER_GROUP_MIN, USER_GROUP_MAX)
			group_users = random.sample(range(USER_NUM), group_user_num)
			group_users.sort()
			output.write("{}\n".format(' '.join(str(x) for x in group_users)))

def simulate_user_items():
	with open ('./simulation/user_items', 'w') as output:
		for i in range(USER_NUM):
			user_item_num = random.randint(ITEM_USER_MIN, ITEM_USER_MAX)
			user_items = random.sample(range(ITEM_NUM), user_item_num)
			user_items.sort()
			output.write("{}\n".format(' '.join(str(x) for x in user_items)))

def simulate_group_costs():
	with open ('./simulation/group_costs', 'w') as output:
		for i in range(GROUP_NUM):
			cost = random.uniform(COST_GROUP_MIN, COST_GROUP_MAX)
			output.write("{}\n".format(cost))

if __name__ == '__main__':
	simulate_item_similarity()
	simulate_user_items()
	simulate_group_users()
	simulate_group_costs()
