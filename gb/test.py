import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight

class MainRecommender:
    def __init__(self, data, weighting=True):
        self.user_item_matrix = self._prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)

    def _prepare_matrix(self, data):
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                         )
        user_item_matrix = user_item_matrix.astype(float)
        return user_item_matrix

    def _prepare_dicts(self, user_item_matrix):
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values
        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))
        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))
        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    def fit(self, user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        return model

    def get_als_recommendations(self, user, N=5):
        user_id = self.userid_to_id[user]
        recommendations = self.model.recommend(user_id, csr_matrix(self.user_item_matrix).tocsr(), N=N)
        item_ids = [self.id_to_itemid[rec[0]] for rec in recommendations]
        return item_ids

    def get_precision_at_5(self, user, actual_purchases):
        recommendations = self.get_als_recommendations(user, N=5)
        relevant_recs = [rec for rec in recommendations if rec in actual_purchases]
        precision = len(relevant_recs) / 5
        return precision

    def generate_recommendations_csv(self, user_ids, file_path):
        recommendations = []
        for user in user_ids:
            recommendations.append((user, self.get_als_recommendations(user)))
        df = pd.DataFrame(recommendations, columns=['user_id', 'recommendations'])
        df.to_csv(file_path, index=False)





# Загрузка данных
data = pd.read_csv('retail_train.csv')

# Создание экземпляра класса MainRecommender
recommender = MainRecommender(data)

# Получение рекомендаций для пользователя с id=1
user_id = 1
recommendations = recommender.get_als_recommendations(user_id)
print(recommendations)

# Расчет метрики precision@5 для пользователя с id=1
actual_purchases = [123, 456, 789]  # Фактические покупки пользователя
precision = recommender.get_precision_at_5(user_id, actual_purchases)
print(precision)

# Генерация файла recommendations.csv для заданных пользователей
user_ids = [1, 2, 3]
recommender.generate_recommendations_csv(user_ids, 'recommendations.csv')
