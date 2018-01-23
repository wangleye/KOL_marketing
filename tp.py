#generate KOL ranking list according to Top Persuader algorithm
import numpy as np
import pymysql

conn = pymysql.connect(host='127.0.0.1',
    user='root',
    passwd='123456',
    db='all0504')

KOLs = []
user_item_likes = {}

def read_user_item_preference():
    x = conn.cursor()
    x.execute("SELECT iduser, moviestr FROM user")
    results = x.fetchall()
    for result in results:
        user_id = str(result[0])
        moviestr_items = set(result[1].split(';'))
        if '' in moviestr_items:
            moviestr_items.remove('')
        user_item_likes[user_id] = moviestr_items

def read_KOLs():
    # read kol file
    with open("facebook/KOL_audience") as inputfile:
        for line in inputfile:
            if len(line.strip()) > 0:
                words = line.strip().split(';')
                group_id = words[0]
                KOLs.append(group_id)

def calculate_influence_probability():
    # generate the probability matrix indicating kol i influencing kol j
    n = len(KOLs)
    P = np.asmatrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            if i != j:
                if len(user_item_likes[KOLs[i]]) == 0:
                    P[i,j] = 0
                else:
                    P[i,j] = len(user_item_likes[KOLs[i]] & user_item_likes[KOLs[j]]) * 1.0 / len(user_item_likes[KOLs[i]])
                if P[i,j] == 0:
                    P[i,j] = 0.001
            else:
                P[i,j] = 0
    return P

def iterative_calculate_tp_score(P):
    n = len(KOLs)
    C = np.asmatrix(np.random.rand(n, 1))
    for i in range(10000):
        C = P * C
        C = C / C.max()

    print(C.shape, C)
    # rank according to the persuader score
    ranked_idx = sorted(range(n), key=lambda k: C[k,0], reverse=True)
    print(ranked_idx)

    ranked_kol_list = []
    for i in ranked_idx:
        ranked_kol_list.append(KOLs[i])
    return ranked_kol_list

if __name__ == '__main__':
    print("read user liked movies...")
    read_user_item_preference()
    print("read KOLs...")
    read_KOLs()
    print("calculate influence probability...")
    P = calculate_influence_probability()
    print("get ranked KOL list by tp")
    kol_list = iterative_calculate_tp_score(P)

    with open('./facebook/KOL_tp_rankedlist', 'w') as outputfile:
        for kol in kol_list:
            outputfile.write('{}\n'.format(kol))
    print(kol_list)