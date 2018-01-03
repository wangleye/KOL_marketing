import pymysql
import numpy as np

conn = pymysql.connect(host='127.0.0.1',
    user='root',
    passwd='123456',
    db='all0504')

USERS_BY_UID = {}
NET = {}

class User:
    def __init__(self, uid, friends, items):
        self.uid = uid
        self.friends = friends
        self.items = items

def get_KOLs_with_most_friends(k=1000):
    users = []
    x = conn.cursor()
    x.execute("SELECT iduser, friendstr, musicstr FROM user limit 100000")
    results = x.fetchall()
    for result in results:
        user_id = result[0]
        friends = set(result[1].split(';'))
        items = result[2].split(';')
        user = User(user_id, friends, items)
        users.append(user)
        USERS_BY_UID[user.uid] = user
    
    count = 0
    for result in results:
        count += 1
        if count%100 == 0:
            print count
        user_id = result[0]
        USERS_BY_UID[user_id].friends = USERS_BY_UID[user_id].friends & set(USERS_BY_UID.keys()) # delete users who are not in database

    users.sort(key=lambda x : len(x.friends), reverse=True)
    return users[0:k]

def output_KOL_audience_to_file(KOLs):
    with open('KOL_audience', 'w') as outputfile:
        for kol in KOLs:
            outputfile.write('{};{}\n'.format(kol.uid, ' '.join(kol.friends)))
            print(kol.uid, len(kol.friends))

def liked_items(user_id):
    """
    return the liked movies of a user
    """
    if user_id in USERS_BY_UID:
        return USERS_BY_UID[user_id].items
    else:
        return []

def KOL_CSD(kol):
    distinct_items = set()
    item_popularity = 0
    num_users = len(kol.friends)
    for u in kol.friends:
        distinct_items.update(liked_items(u))
        item_popularity += len(liked_items(u))
    return (item_popularity*1.0/len(distinct_items)-1)/(num_users-1)

def output_KOL_CSD(KOLs):
    with open ('KOL_CSD', 'w') as outputfile:
        for kol in KOLs:
            print(kol.uid)
            outputfile.write('{};{}\n'.format(kol.uid, KOL_CSD(kol)))

def friends(user_id):
    """
    return the friends' ids of a user
    """
    if user_id in USERS_BY_UID:
        return USERS_BY_UID[user_id].friends
    else:
        return []

def net_value(user_id):
    if user_id in NET:
        return NET[user_id]

    f = friends(user_id)
    net_val = len(f)
    alpha = 0.02
    ff_set = set()

    for each_f in f:
        ff = friends(each_f)
        ff_set.update(set(ff))

    for each_f in f:
        if each_f in ff_set:
            ff_set.remove(each_f)
    
    net_val += alpha * len(ff_set)

    NET[user_id] = net_val
    return net_val

def KOL_net_value(kol):
    value = 0
    for friend in kol.friends:
        value += net_value(friend)
    return value

def output_KOL_net_cost(KOLs):
    KOL_net_values = {}
    max_value = 0
    for kol in KOLs:
        print(kol.uid)
        KOL_net_values[kol.uid] = KOL_net_value(kol)
        if KOL_net_values[kol.uid] > max_value:
            max_value = KOL_net_values[kol.uid]

    with open('KOL_net_cost', 'w') as outputfile:
        for kol in KOLs:
            outputfile.write('{};{}\n'.format(kol.uid, KOL_net_values[kol.uid]*1.0/max_value))

def output_KOL_number_cost(KOLs):
    max_value = 0
    for kol in KOLs:
        if len(kol.friends) > max_value:
            max_value = len(kol.friends)
    with open('KOL_number_cost', 'w') as outputfile:
        for kol in KOLs:
            outputfile.write('{};{}\n'.format(kol.uid, len(kol.friends)*1.0/max_value))

if __name__ == '__main__':
    print('selecting KOLs...')
    KOLs = get_KOLs_with_most_friends()
    KOLs_100 = np.random.choice(KOLs, 100, replace=False)
    print('outputing KOL audience to files...')
    output_KOL_audience_to_file(KOLs_100)
    print('outputing KOL CSDs...')
    output_KOL_CSD(KOLs_100)
    print('outputing KOL net costs...')
    output_KOL_net_cost(KOLs_100)
    print('outputing KOL number costs...')
    output_KOL_number_cost(KOLs_100)