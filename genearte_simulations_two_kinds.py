import random
import numpy as np

GROUP_1_NUM = 6
GROUP_2_NUM = 45
USER_NUM = 100
ITEM_NUM = 100

USER_GROUP_1 = 10
USER_GROUP_2 = 20

ITEM_USER_MIN = 2
ITEM_USER_MAX = 5

COST_GROUP_1 = 0.0999
COST_GROUP_2 = 0.4999

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
		for i in range(GROUP_1_NUM):
			group_users = random.sample(range(USER_NUM), USER_GROUP_1)
			group_users.sort()
			output.write("{}\n".format(' '.join(str(x) for x in group_users)))
		for i in range(GROUP_2_NUM):
			group_users = random.sample(range(USER_NUM), USER_GROUP_2)
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
		for i in range(GROUP_1_NUM):
			output.write("{}\n".format(COST_GROUP_1))
		for i in range(GROUP_2_NUM):
			output.write("{}\n".format(COST_GROUP_2))

if __name__ == '__main__':
	simulate_item_similarity()
	simulate_user_items()
	simulate_group_users()
	simulate_group_costs()
