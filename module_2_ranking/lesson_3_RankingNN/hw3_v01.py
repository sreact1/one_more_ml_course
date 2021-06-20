## template
import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # укажите архитектуру простой модели здесь
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        self.out_activation = torch.nn.Sigmoid()

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits
    
class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        
        self.X_train = torch.FloatTensor(self._scale_features_in_query_groups(
            inp_feat_array=X_train, inp_query_ids=self.query_ids_train))
        self.ys_train = torch.FloatTensor(y_train)
        
        
        self.X_test = torch.FloatTensor(self._scale_features_in_query_groups(
            inp_feat_array=X_test, inp_query_ids=self.query_ids_test))
        self.ys_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                    inp_query_ids: np.ndarray) -> np.ndarray:
    
        inp_query_ids_uniq = np.unique(inp_query_ids)

        for id_i in inp_query_ids_uniq:
            scaler = StandardScaler()
            inp_feat_array[inp_query_ids == id_i, :] = \
                scaler.fit_transform(inp_feat_array[inp_query_ids == id_i, :])

        return inp_feat_array

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        # допишите ваш код здесь
        net = ListNet(num_input_features=listnet_num_input_features,\
                      hidden_dim=listnet_hidden_dim)
        return net

    
    def fit(self) -> List[float]:
        # допишите ваш код здесь
#         self.model.train()
        ndcg_k = []
        for epoche in range(5):
            self._train_one_epoch()
            ndcg_k.append(self._eval_test_set())
#             print('epoche {} finished'.format(epoche + 1))
#             print('mean test nDCG top 10 score {}'.format(self._eval_test_set()))
        return ndcg_k
    
    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        #         def listnet_ce_loss(y_i, z_i):
        #     """
        #     y_i: (n_i, 1) GT
        #     z_i: (n_i, 1) preds
        #     """

        P_y_i = torch.softmax(batch_ys, dim=0)
        P_z_i = torch.softmax(batch_pred, dim=0)
        
#         return -torch.sum(P_y_i * torch.log(P_z_i))
        return -torch.sum(P_y_i * torch.log(P_z_i/P_y_i))

    def _train_one_epoch(self) -> None:
        # допишите ваш код здесь
        self.model.train()
        ids_uniq = np.unique(self.query_ids_train)
#         ids_uniq_rand = ids_uniq
        ids_uniq_rand = ids_uniq[torch.randperm(len(ids_uniq))]
        
        for n_id in ids_uniq_rand:
        
            X_train_id = self.X_train[self.query_ids_train == n_id]
            ys_train_id = self.ys_train[self.query_ids_train == n_id]
            
#             N_train = np.shape(X_train_id)[0]
#             idx_rand = torch.randperm(N_train)
#             X_train_id = X_train_id[idx_rand]
#             ys_train_id = ys_train_id[idx_rand]
            
#             batch_X = X_train_id
#             batch_ys = ys_train_id

#             self.optimizer.zero_grad()
#             # self.optimizer.no_grad()
#             # torch.no_grad()
            
# #             if len(batch_X) > 0:
#             batch_pred = self.model(batch_X).reshape(-1,)
# #                 print('batch_pred')
# #                 print(batch_pred)
#             batch_loss = self._calc_loss(batch_ys=batch_ys, batch_pred=batch_pred)
#             batch_loss.backward(retain_graph=False)
# #             batch_loss.backward(retain_graph=True)
#             self.optimizer.step()
            
            batch_size = 25
            cur_batch = 0
            N_train = np.shape(X_train_id)[0]

            for n_batch in range(N_train // batch_size):
                batch_X = X_train_id[cur_batch: cur_batch + batch_size]
                batch_ys = ys_train_id[cur_batch: cur_batch + batch_size]
                cur_batch += batch_size

                self.optimizer.zero_grad()
#                 if len(batch_X) > 0:
                batch_pred = self.model(batch_X).reshape(-1,)
                batch_loss = self._calc_loss(batch_ys=batch_ys, batch_pred=batch_pred)
                batch_loss.backward(retain_graph=False)
                self.optimizer.step()

                
    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            # допишите ваш код здесь
            with torch.no_grad():
                
                ids_test_uniq = np.unique(self.query_ids_test)
                
                for id_test in ids_test_uniq:
                    cur_X_test = self.X_test[self.query_ids_test == id_test]
                    valid_pred = self.model(cur_X_test)
#                     print('self.ys_test[self.query_ids_test == id_test]')
#                     print(self.ys_test[self.query_ids_test == id_test])
                    cur_ndcg_k = self._ndcg_k(\
                        ys_true=self.ys_test[self.query_ids_test == id_test],\
                        ys_pred=valid_pred,\
                        ndcg_top_k=10)
        
                    ndcgs.append(cur_ndcg_k)
                
                #  ndcgs_avg = sum(ndcgs) / len(ndcgs)
                #  print('mean test nDCG top 10 score {}'.format(ndcgs_avg))
            
            return np.mean(ndcgs)
        
        
    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        # допишите ваш код здесь
        ys_true = torch.reshape(ys_true, (-1,))
        ys_pred = torch.reshape(ys_pred, (-1,))
        
        ys_pred_sorted = torch.sort(ys_pred, descending=True)
        ys_true_sorted = ys_true[ys_pred_sorted[1]]
        ys_true_sorted_sep = torch.sort(ys_true, descending=True)[0]
        
#         print('ys_true')
#         print(np.shape(ys_true_sorted[:ndcg_top_k]))
#         print(ys_true_sorted[:ndcg_top_k])
#         print('ys_pred')
#         print(np.shape(ys_pred_sorted[0][:ndcg_top_k]))
#         print(ys_pred_sorted[0][:ndcg_top_k])
        
#         ndcg_k = self._ndcg(ys_true=ys_true_sorted[:self.ndcg_top_k],\
#                             ys_pred=ys_pred_sorted[0][:self.ndcg_top_k],\
#                             gain_scheme='exp2')
        dcg_act = self._dcg(ys_true=ys_true_sorted[:self.ndcg_top_k],\
                            ys_pred=ys_pred_sorted[0][:self.ndcg_top_k],\
                            gain_scheme='exp2')
        
        dcg_max = self._dcg(ys_true=ys_true_sorted_sep[:self.ndcg_top_k],\
                            ys_pred=ys_true_sorted_sep[:self.ndcg_top_k],\
                            gain_scheme='exp2')
        
        try:
            ndcg_k = dcg_act / dcg_max
        except:
            ndcg_k = 0
            
        if np.isnan(ndcg_k):
            ndcg_k = 0
        
        return ndcg_k


    # def _ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str = 'const') -> float:
    def _ndcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str) -> float:
        try:
            ndcg_val = self._dcg(ys_true=ys_true, ys_pred=ys_pred, gain_scheme=gain_scheme) / \
                self._dcg(ys_true=ys_true, ys_pred=ys_true, gain_scheme=gain_scheme)
        except:
            ndcg_val = 0.
        
        return ndcg_val
    
    def _compute_gain(self, y_value: float, gain_scheme: str) -> float:
#         if gain_scheme == 'exp2':
#             return 2 ** y_value - 1.
#         else:
#             return y_value + 0.
        return 2 ** y_value - 1.

    def _dcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str) -> float:
        ys_pred_sorted = torch.sort(ys_pred, descending=True)
        log2_list = [math.log2(x) for x in range(2, len(ys_pred) + 2)]

        dcg_val = 0.
        for i in range(len(log2_list)):
            dcg_val += self._compute_gain(ys_true[ys_pred_sorted[1]][i].item(), \
                                          gain_scheme=gain_scheme) / log2_list[i]

        return dcg_val
     
        