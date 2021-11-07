"""
Metrics

"""
import numpy as np

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return (flags.sum() > 0) * 1

def hit_rate_at_k(recommended_list, bought_list, k=5):
    return hit_rate(recommended_list[:k], bought_list)

def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(recommended_list)

def precision_at_k(recommended_list, bought_list, k=5):
    return precision(recommended_list[:k], bought_list)

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_recommended.sum()

def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list)
    return flags.sum() / len(bought_list)


def recall_at_k(recommended_list, bought_list, k=5):
    return recall(recommended_list[:k], bought_list)


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_recommended = np.array(prices_recommended)[:k]
    prices_bought = np.array(prices_bought)
    flags = np.isin(recommended_list, bought_list)
    return np.dot(flags, prices_recommended).sum() / prices_bought.sum()


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]
    if len(relevant_indexes) == 0:
        return 0
    amount_relevant = len(relevant_indexes)

    relevant_indexes = relevant_indexes[relevant_indexes <= k]

    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
    return sum_ / amount_relevant

def map_k(recommended_list, bought_list, k=5):
    
    # сделать дома
    
    result = 0
    N = len(bought_list_3_users)
    for i in range(N):
        print(f"ap@k({i})={ap_k(recommended_list[i], bought_list[i], k=5)}")
        result += ap_k(recommended_list[i], bought_list[i], k=5)
        
    return result/N


# Normalized discounted cumulative gain ( NDCG@k)

def idial_dcg_at_k(recommended_list, bought_list, k):
    bought = np.array(bought_list)
    recommended = np.array(recommended_list[:k])
    discount = []
    
    for i in range(k):
        if i < 2:
            discount.append( 1/(i+1) )
        else:
            discount.append( 1/math.log2(i+1))
    
    print(discount)
    return  sum(discount)/k


def dcg_at_k(recommended_list, bought_list, k):
    bought = np.array(bought_list)
    recommended = np.array(recommended_list[:k])
    discount = []
    
    for i in range(k):
        if recommended_list[i] in bought_list:
            if i < 2:
                discount.append( 1/(i+1) )
            else:
                discount.append( 1/math.log2(i+1) )
        else:
            discount.append(0)
    
    print(discount)
    return  sum(discount)/k


def ndcg_at_k(recommended_list, bought_list, k):
    dcg_atk= dcg_at_k(recommended_list, bought_list, k)
    idial_dcg_atk = idial_dcg_at_k(recommended_list, bought_list, k)
    
    return  dcg_atk/(k*idial_dcg_atk)



# Mean Reciprocal Rank ( MRR@k )

def inverse_rank(recommended_list, bought_list, k=1):
    # Находим 1/ранг первого релевантного предсказания для пользователя
    rank=[( 1/(i+1) ) for i in range(len(recommended_list[:k])) for j in range(len(bought_list)) if recommended_list[i] == bought_list[j]]
    
    if len(rank) > 0:
        return rank[0]
    else:
        return 0
    

def reciprocal_rank(recommended_list_users, bought_list_users, k=1):
    # Находим список рангов
    
    rec_rank = []
    for i in range(len(bought_list_users)):
        rec_rank.append(inverse_rank(recommended_list_users[i], bought_list_users[i], k))
    rec_rank
    return rec_rank


def mean_reciprocal_rank(recommended_list_users, bought_list_users, k=1):
    # Mean Reciprocal Rank
    
    rec_rank = reciprocal_rank(recommended_list_users, bought_list_users, k)

    mrr_at_k = np.sum(rec_rank)/len(bought_list_users)
    
    return mrr_at_k