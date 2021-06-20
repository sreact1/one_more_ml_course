import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # допишите ваш код здесь
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        
        # scale train data
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(
            inp_feat_array=X_train, inp_query_ids=self.query_ids_train))
#         self.ys_train = torch.FloatTensor(y_train).t()
        self.ys_train = torch.FloatTensor(y_train).reshape(-1,1)
        
        # scale test data
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(
            inp_feat_array=X_test, inp_query_ids=self.query_ids_test))
#         self.ys_test = torch.FloatTensor(y_test).t()
        self.ys_test = torch.FloatTensor(y_test).reshape(-1,1)
        pass

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        # get unique ids
        inp_query_ids_uniq = np.unique(inp_query_ids)
        
        # scale each group by id
        for id_i in inp_query_ids_uniq:
            scaler = StandardScaler()
            inp_feat_array[inp_query_ids == id_i, :] = \
                scaler.fit_transform(inp_feat_array[inp_query_ids == id_i, :])

        return inp_feat_array

    
    
    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        # допишите ваш код здесь
        random.seed(cur_tree_idx)
        
        ## compute lambdas with train_preds
        ids_uniq = np.unique(self.query_ids_train)
        
#         print('self.train_preds')
#         print(self.train_preds)
#         print('np.shape(self.train_preds)')
#         print(np.shape(self.train_preds))
        if all(np.array(self.train_preds) == 0):
            lambdas_upd = self.ys_train
        else:
            lambdas_list = []

            for n_id in ids_uniq:

                X_train_id = self.X_train[self.query_ids_train == n_id]
                ys_train_id = self.ys_train[self.query_ids_train == n_id]
                train_preds_id = train_preds[self.query_ids_train == n_id]

                lambdas_list.append(\
                    self._compute_lambdas(y_true=ys_train_id, y_pred=train_preds_id))

#             lambdas_upd = np.concatenate(lambdas_list)
            lambdas_upd = np.concatenate(lambdas_list) * -1
#             lambdas_upd = np.concatenate(lambdas_list) * -1 * self.lr
        
        ## set train data
        N, M = np.shape(self.X_train)
        
        random.seed(cur_tree_idx)
        sample_ids = random.sample(range(N), int(N * self.subsample))
        random.seed(cur_tree_idx)
        colsample_ids = np.array(random.sample(range(M), int(M * self.colsample_bytree)))
        
#         print('np.shape(self.X_train)')
#         print(np.shape(self.X_train))
#         print('np.shape(sample_ids)')
#         print(np.shape(sample_ids))
#         print('np.shape(colsample_ids)')
#         print(np.shape(colsample_ids))
        X_train = self.X_train[sample_ids, :][:, colsample_ids]
#         ys_train = self.ys_train[sample_ids, ]
#         ys_train = train_preds[sample_ids]
        ys_train = lambdas_upd[sample_ids]
        
#         print('ys_train_train_one_tree')
#         print(ys_train)
        
        # fit one tree & predict
        single_tree = DecisionTreeRegressor(max_depth=self.max_depth,\
                                            min_samples_leaf=self.min_samples_leaf,\
                                            random_state=cur_tree_idx)
        single_tree.fit(X=X_train, y=ys_train)
        
        # return tree and features ids
        return (single_tree, colsample_ids)

    
#     def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
#         # допишите ваш код здесь
#         pass
    
    def _compute_lambdas(self, y_true, y_pred, ndcg_scheme='exp2'):
        # рассчитаем нормировку, IdealDCG
#         print('y_true')
#         print(y_true)
#         print('y_pred')
#         print(y_pred)
        if all(np.array(y_true) == 0):
            ideal_dcg = 1
        else:
            ideal_dcg = self._compute_ideal_dcg(y_true.reshape(1,-1)[0],\
                                            ndcg_scheme=ndcg_scheme)
        
#         print('ideal_dcg')
#         print(ideal_dcg)
        N = 1 / ideal_dcg

        # рассчитаем порядок документов согласно оценкам релевантности
        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1

        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
#             pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.reshape(-1,1)))
#             print('pos_pairs_score_diff')
#             print(pos_pairs_score_diff)
            
            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = self._compute_labels_in_batch(y_true)
#             print('Sij')
#             print(Sij)
            # посчитаем изменение gain из-за перестановок
            gain_diff = self._compute_gain_diff(y_true, ndcg_scheme)
#             print('gain_diff')
#             print(gain_diff)
            
            # посчитаем изменение знаменателей-дискаунтеров
#             decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - \
                (1.0 / torch.log2(rank_order.reshape(1,-1) + 1.0))
#             print('decay_diff')
#             print(decay_diff)
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
#             print('lambda_update')
#             print(lambda_update)
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

#             return Sij, gain_diff, decay_diff, delta_ndcg, lambda_update
        return lambda_update
    
    def _compute_labels_in_batch(self, y_true):

        # разница релевантностей каждого с каждым объектом
#         print('y_true_in_compute_labels_in_batch')
#         print(y_true)
#         rel_diff = y_true - y_true.t()
        rel_diff = y_true - y_true.reshape(1,-1)
#         print('rel_diff')
#         print(rel_diff)
        
        # 1 в этой матрице - объект более релевантен
        pos_pairs = (rel_diff > 0).type(torch.float32)

        # 1 тут - объект менее релевантен
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        return Sij

    def _compute_gain_diff(self, y_true, gain_scheme):
        if gain_scheme == "exp2":
#             gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.reshape(1,-1))
        elif gain_scheme == "diff":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff
    
    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        
        ndcgs = []
        
        # допишите ваш код здесь
        with torch.no_grad():

#             ids_test_uniq = np.unique(self.query_ids_test)

#             for id_test in ids_test_uniq:
#                 cur_X_test = self.X_test[self.query_ids_test == id_test, :]
#                 valid_pred = self.predict(cur_X_test)
#                 cur_ndcg = self._ndcg_k(\
#                     ys_true=self.ys_test.reshape(1,-1)[0][self.query_ids_test == id_test],\
#                     ys_pred=valid_pred,\
#                     ndcg_top_k=10)
                
            ids_test_uniq = np.unique(queries_list)

            for id_test in ids_test_uniq:
                cur_ndcg = self._ndcg_k(\
                    ys_true=true_labels[queries_list == id_test],\
                    ys_pred=preds[queries_list == id_test],\
                    ndcg_top_k=self.ndcg_top_k)
                
                ndcgs.append(cur_ndcg)
                
        return np.mean(ndcgs)

    def fit(self):
        np.random.seed(0)
        # допишите ваш код здесь
        self.train_preds = torch.zeros(np.shape(self.ys_train)[0])
        self.features_ids = []
        self.trees = []
        self.best_ndcg = 0
        self.best_n_trees = 1
        self.ndcg_list = []
        
        
        for i in range(self.n_estimators):
            single_tree, features_ids = self._train_one_tree(\
                cur_tree_idx=i, train_preds=self.train_preds)
            self.trees.append(single_tree)
            self.features_ids.append(features_ids)
            
            ## calculate new tree prediction & update train_preds
            new_ys_preds = single_tree.predict(self.X_train[:, features_ids])
#             self.train_preds += new_ys_preds
#             self.train_preds += new_ys_preds * self.lr
            
#             print('new_ys_preds')
#             print(new_ys_preds)
#             print('self.train_preds')
#             print(self.train_preds)
            
            ## calc new ndcg and update best & best_n_tree with _calc_data_ndcg
            
            # apply _calc_data_ndcg
            ndcg_cur_train = self._calc_data_ndcg(\
                queries_list=self.query_ids_train, true_labels=self.ys_train,\
                preds=self.predict(self.X_train))
            
            ndcg_cur_test = self._calc_data_ndcg(\
                queries_list=self.query_ids_test, true_labels=self.ys_test,\
                preds=self.predict(self.X_test))
            
            ndcg_cur = ndcg_cur_test
            self.ndcg_list.append(ndcg_cur)
            self.best_ndcg = max(self.ndcg_list)
            self.best_n_trees = np.argmax(self.ndcg_list) + 1
            
# #             print('i_in_fit_fun')
#             print(i)
# #             print('ndcg_cur_in_fit_fun')
#             print(ndcg_cur)
            print("n_tree: {}, train_ndcg_k: {}, test_ndcg_k: {}".format(\
                i, ndcg_cur_train, ndcg_cur_test))
        
        self.trees = self.trees[:self.best_n_trees]
        self.features_ids = self.features_ids[:self.best_n_trees]
        self.n_estimators = self.best_n_trees
        
#         self.trees = trees_list

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        preds = torch.zeros(np.shape(data)[0])
#         for i in range(self.n_estimators):
        for i in range(len(self.trees)):
#             print('len(self.trees)')
#             print(len(self.trees))
            ## calculate new tree prediction & update train_preds
#             print("n_estimator_i")
#             print(i)
            new_ys_preds = self.trees[i].predict(data[:, self.features_ids[i]])
#             self.train_preds += new_ys_preds * self.lr
            preds += new_ys_preds * self.lr
            
        return preds
            
#     def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
#         # допишите ваш код здесь
#         pass
    
    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        # допишите ваш код здесь
        ys_true = torch.reshape(ys_true, (-1,))
        ys_pred = torch.reshape(ys_pred, (-1,))

        ys_pred_sorted = torch.sort(ys_pred, descending=True)
        ys_true_sorted = ys_true[ys_pred_sorted[1]]
        ys_true_sorted_sep = torch.sort(ys_true, descending=True)[0]

        dcg_act = self._dcg(ys_true=ys_true_sorted[:self.ndcg_top_k],\
                            ys_pred=ys_pred_sorted[0][:self.ndcg_top_k],\
                            gain_scheme='exp2')

        dcg_max = self._dcg(ys_true=ys_true_sorted_sep[:self.ndcg_top_k],\
                            ys_pred=ys_true_sorted_sep[:self.ndcg_top_k],\
                            gain_scheme='exp2')

#         try:
#             ndcg_k = dcg_act / dcg_max
#         except:
#             ndcg_k = 0
#         if np.isnan(ndcg_k):
#             ndcg_k = 0
        
#         if (dcg_act == 0):
#             ndcg_k = 0
        if (dcg_max == 0):
            ndcg_k = 1
        else:
            ndcg_k = dcg_act / dcg_max
        
        return ndcg_k

    def _ndcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str = 'const') -> float:
#         try:
        dcg_val = self._dcg(ys_true=ys_true, ys_pred=ys_pred, gain_scheme=gain_scheme)
        ideal_dcg_val = self._dcg(ys_true=ys_true, ys_pred=ys_true, gain_scheme=gain_scheme)
        
#         print('dcg_val')
#         print(dcg_val)
#         print('ideal_dcg_val')
#         print(ideal_dcg_val)
        
#         if (dcg_val == 0):
#             return dcg_val
        if (ideal_dcg_val == 0):
            return 1
        
        else:
            return dcg_val / ideal_dcg_val
    
    def _compute_gain(self, y_value: float, gain_scheme: str) -> float:
#         return 2 ** y_value - 1.
        if gain_scheme == 'exp2':
            return 2 ** y_value - 1.
        else:
            return y_value + 0.

    def _dcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str) -> float:
        ys_pred_sorted = torch.sort(ys_pred, descending=True)
#         print('ys_pred_sorted')
#         print(ys_pred_sorted)
        
        log2_list = [math.log2(x) for x in range(2, len(ys_pred) + 2)]

        dcg_val = 0.
        for i in range(len(log2_list)):
            dcg_val += self._compute_gain(ys_true[ys_pred_sorted[1]][i].item(), \
                                          gain_scheme=gain_scheme) / log2_list[i]
            
#         print('dcg_val')
#         print(dcg_val)
        
#         if (dcg_val == 0):
# #             dcg_val = -1
#             dcg_val = 0.00001
        return dcg_val
    
    def _compute_ideal_dcg(self, y_true, ndcg_scheme='exp2'):
#         print(y_true)
#         try:
        return self._dcg(ys_true=y_true, ys_pred=y_true, gain_scheme=ndcg_scheme)
#         except:
#             return 1
    
    
    def save_model(self, path: str):
        # допишите ваш код здесь
        ## save model
        state = {
            'trees': self.trees, 
            'features_ids': self.features_ids,
            'ndcg_top_k': self.ndcg_top_k,
#             'best_n_trees': self.best_n_trees,
            'n_estimators': self.n_estimators,
            'lr': self.lr}
        
        f = open(path, 'wb')
        pickle.dump(state, f)
        
        pass

    def load_model(self, path: str):
        # допишите ваш код здесь
        f = open(path, 'rb')
        state = pickle.load(f)
        
        self.trees = state['trees']
        self.features_ids = state['features_ids']
        self.ndcg_top_k = state['ndcg_top_k']
#         self.best_n_trees = state['best_n_trees']
        self.lr = state['lr']
        
#         return state
        pass
    
#     def save_obj(obj, name):
#         with open(name + '.pkl', 'wb') as f:
#             pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#     def load_obj(name):
#         with open(name + '.pkl', 'rb') as f:
#             return pickle.load(f)