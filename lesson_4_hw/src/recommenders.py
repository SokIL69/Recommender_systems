import pandas as pd
import numpy as np


import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        # См. лекцию 2
        # data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999_999
        
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id',
                                  columns='item_id', 
                                  values='quantity',
                                  aggfunc='count', 
                                  fill_value=0
                                 )
        
        user_item_matrix[user_item_matrix > 0] = 1 # так как в итоге хотим предсказать 
        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        # переведем в формат saprse matrix
        sparse_user_item = csr_matrix(user_item_matrix)

        print(user_item_matrix.head(3))
        
        return user_item_matrix
    
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
    
    
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
        # См. лекцию 2
        
        own_recommender = ItemItemRecommender(K=1, num_threads=4)  # K - кол-во ближайших соседей
        # own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr(),  # На вход item-user matrix
                             show_progress=True)
        
        return own_recommender
    
    
    @staticmethod
    def fit(user_item_matrix, factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        # См. лекции 3, 4
        
        model = AlternatingLeastSquares(factors=factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model
    
    
    def __init__(self, data, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
    
    def get_similar_items_recommendation(self, user, N = 5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # См. лекцию 3
        # your_code
        # Практически полностью реализовали на прошлом вебинаре
        
        # Получаем список из N item_id приобретённых пользователем user
        user_to_id = self.userid_to_id[user]
        user_items = self.user_item_matrix.getrow(user_to_id).todense()
        user_items = user_items.tolist()

        user_item_list = []  # список из N item_id приобретённых пользователем user
        items = []
        for row in user_items:
            i = 0
            k = 0
            for element in row:
                if element != 0:
                    user_item_list.append(i)
                    items.append(self.id_to_itemid[i])
                    # print(i, id_to_itemid[i], element)
                    k += 1
                i += 1
                if k == N:
                    break
                    
          
        print(f'Список {N} товаров приобретённых пользователем {user}: \n {items}')
#         print(type(self.user_item_matrix))
#         user_items = self.user_item_matrix.reset_index()
#         user_items[user_items.user_id == user]
#         user_item_list = []  # список из N item_id приобретённых пользователем user
#         k=0
#         for col in list(item_0.columns[1:]):
#             if item_0[col][0] != 0.0:
#                 user_item_list.append(col)
#                 k += 1
#                 if k == 5:
#                     break
            
        # Находим товары похожие на товары приобретённые пользователем
        
        closest_items = []  # Список рекомендуемых товаров
        
        for item_row_id in user_item_list:
            closest_items.append([self.id_to_itemid[row_id] for row_id, score in self.model.similar_items(item_row_id, N=1)])
        
        res = closest_items
       
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
  
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        # # См. лекцию 3
        # your_code

        user_list = self.model.similar_users(self.userid_to_id[user], N)  # список пар: похожий пользователь, купленный им товар
        
        # Создаём список из N похожих пользователей
        users=[]  # список похожих юзеров
        for row in user_list:
            users.append(row[0])
        
        print(f'Список пользователей похожих на пользователя {user}: \n {users}')
        
        # Создаём список N item_id приобретёнными похожими пользователями (по одному товару на пользователя)
        user_item_list = []  # список из N item_id приобретённых пользователем user
        for userid in users:
            user_to_id = self.userid_to_id[userid]
            user_items = self.user_item_matrix.getrow(user_to_id).todense()
            user_items = user_items.tolist()
            for row in user_items:
                i = 0
                k = 0
                for element in row:
                    if element != 0:
                        user_item_list.append(i)
                        k += 1
                    i += 1
                    if k == 1:
                        break
        
        closest_items = []  # Список рекомендуемых товаров
        for item_row_id in user_item_list:
            closest_items.append([self.id_to_itemid[row_id] for row_id, score in self.model.similar_items(item_row_id, N=1)])
        
        res = closest_items
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    
#     def MainRecommender(data, take_n_popular=5000, item_features=None):
#         pass