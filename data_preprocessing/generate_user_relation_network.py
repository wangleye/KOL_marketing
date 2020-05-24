import pymysql

conn = pymysql.connect(host='127.0.0.1',
    user='root',
    passwd='123456',
    db='all0504')

def get_user_set():
    user_set = set()
    with open('../facebook/KOL_audience') as input_user_file:
        for line in input_user_file:
            if line.strip() == '':
                continue
            words = line.strip().split(';')
            user_ids = set(words[1].split(' '))
            user_set = user_set.union(user_ids)

    return user_set

def get_item_set(num=100):
    item_set = set()
    count = 0
    with open('../facebook/{}_list'.format(SCENARIO)) as input_item_file:
        for line in input_item_file:
            if line.strip() == '':
                continue
            count += 1
            item_info = line.strip().split()
            item_set.add(item_info[0])
            if count == num:
                break
    return item_set

def read_user_item_preference(user_set):
    user_item_likes = {}
    x = conn.cursor()
    x.execute("SELECT iduser, {}str FROM user".format(SCENARIO))
    results = x.fetchall()
    for result in results:
        user_id = result[0]
        moviestr_items = result[1].split(';')
        if user_id in user_set or len(user_set) == 0:
            user_item_likes[user_id] = set()
            for movie_i in moviestr_items:
                user_item_likes[user_id].add(movie_i)
    return user_item_likes, set(user_item_likes.keys())

def read_user_relationship(user_set):
    user_relationship = {}
    for user in user_set:
        user_relationship[user] = set()
    x = conn.cursor()
    x.execute("SELECT iduser, friendstr FROM user")
    results = x.fetchall()
    for result in results:
        user_id = result[0]
        if user_id not in user_set:
            continue
        friends = result[1].split(';')
        for friend in friends:
            user_relationship[user_id].add(friend)
    return user_relationship

def read_item_similarity_from_file():
    SIM = {}
    with open("../facebook/{}_similarity".format(SCENARIO)) as inputfile:
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
    return SIM

def calculate_user_similairity(user1, user2):
    if user1 not in USER_PREF or user2 not in USER_PREF:
        return 0
    num_co_liked_item = len(USER_PREF[user1] & USER_PREF[user2])
    if num_co_liked_item == 0:
        user_sim = 0
    else:
        user_sim = num_co_liked_item*1.0 / (len(USER_PREF[user2])+1)
    return user_sim

def get_item_similarity(item1, item2):
    if item1 not in ITEM_SIM or item2 not in ITEM_SIM[item1]:
        return 0
    return ITEM_SIM[item1][item2]

def user_item_affinity(user_id, target_item, consider_item=True, consider_friend=True, indirect_friend=False, inindirect_friend=False): #indirect_friends: whether consider indirect friends
    score = 0
    for item in ITEM_SET:
        if item in USER_PREF[user_id]:
            score += get_item_similarity(target_item, item)

    if score == 0:
        return 0 # early stop the users whith no item similarity (if continue, too slow for the algorithm)

    if not consider_item:
        score = 0.0001
    considered_f = set()
    if consider_friend:
        for friend in USER_RELATION[user_id]:
            if friend in USER_PREF and target_item in USER_PREF[friend]:
                score += calculate_user_similairity(user_id, friend)
                considered_f.add(friend)

            # if counting indirect friends
            if indirect_friend and (friend in USER_RELATION):
                friends_of_f = USER_RELATION[friend]
                for friend_of_f in friends_of_f:
                    if (friend_of_f in USER_PREF) and (friend_of_f not in considered_f) and target_item in USER_PREF[friend_of_f]:
                        # score += calculate_user_similairity(user_id, friend) * calculate_user_similairity(friend, friend_of_f)
                        score += calculate_user_similairity(user_id, friend_of_f)
                        considered_f.add(friend_of_f)
                    
                    # if counting inindirect friends
                    if inindirect_friend and (friend_of_f in USER_RELATION):
                        for fff in USER_RELATION[friend_of_f]:
                            if (fff in USER_PREF) and (fff not in considered_f) and target_item in USER_PREF[fff]:
                                # score += calculate_user_similairity(user_id, friend) * calculate_user_similairity(friend, friend_of_f) * calculate_user_similairity(friend_of_f, fff)
                                score += calculate_user_similairity(user_id, fff)
                                considered_f.add(fff)
    return score

def output_user_item_aff():
    with open("user_{}_aff_score_100_both".format(SCENARIO), "w") as outputfile:
        outputfile.write('user {} score truth\n'.format(SCENARIO))
        for user in USER_SET:
            if user not in USER_PREF:
                continue
            for item in ITEM_SET:
                score = user_item_affinity(user, item, consider_item=True, consider_friend=True)
                isTrue = 1 if item in USER_PREF[user] else 0
                if score > 0:
                    outputfile.write('{} {} {} {}\n'.format(user, item, score, isTrue))

def output_user_item_aff_only_item():
    with open("user_{}_aff_score_100_only_item".format(SCENARIO), "w") as outputfile:
        outputfile.write('user {} score truth\n'.format(SCENARIO))
        for user in USER_SET:
            if user not in USER_PREF:
                continue
            for item in ITEM_SET:
                score = user_item_affinity(user, item, consider_item=True, consider_friend=False)
                isTrue = 1 if item in USER_PREF[user] else 0
                if score > 0:
                    outputfile.write('{} {} {} {}\n'.format(user, item, score, isTrue))

def output_user_item_aff_only_friend():
    with open("user_{}_aff_score_100_only_friend".format(SCENARIO), "w") as outputfile:
        outputfile.write('user {} score truth\n'.format(SCENARIO))
        for user in USER_SET:
            if user not in USER_PREF:
                continue
            for item in ITEM_SET:
                score = user_item_affinity(user, item, consider_item=False, consider_friend=True)
                isTrue = 1 if item in USER_PREF[user] else 0
                if score > 0:
                    outputfile.write('{} {} {} {}\n'.format(user, item, score, isTrue))

if __name__ == '__main__':
    SCENARIO = 'book'
    print('reading user set...')
    USER_SET = get_user_set()
    print('reading item set...')
    ITEM_SET = get_item_set()
    print('reading user preference...')
    USER_PREF, USER_SET = read_user_item_preference(USER_SET)
    print(len(USER_PREF.keys()))
    print('reading user relationship...')
    USER_RELATION = read_user_relationship(USER_SET)
    print('reading item similarity...')
    ITEM_SIM = read_item_similarity_from_file()
    print('outputing to file...')
    output_user_item_aff()
    #output_user_item_aff_only_item()
    #output_user_item_aff_only_friend()
