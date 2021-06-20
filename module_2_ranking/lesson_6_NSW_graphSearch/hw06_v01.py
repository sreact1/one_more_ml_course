from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    # допишите ваш код здесь 
    return np.sqrt(np.sum((documents - pointA) ** 2, axis=1)).reshape(-1,1)


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:
    # допишите ваш код здесь 
    dict_points = dict()
    
    for i in range(np.shape(data)[0]):
        # random sampling for large datasets
        if use_sampling:
            N_points = int(sampling_share * np.shape(data)[0])
            N_points_idx = np.random.choice(np.shape(data)[0], N_points, replace=False)
            # distance from current point
            dist_i = distance(data[i, ], data[N_points_idx, ])
        else:
            dist_i = distance(data[i, ], data)
        
        # get indices of sorted values
        dist_i_argsort = np.argsort(dist_i, axis=0)
        
        # choose long edges (points)
        num_edges_long = np.min((num_candidates_for_choice_long, num_edges_long))
        long_rand_idx = np.random.choice(num_candidates_for_choice_long, num_edges_long,\
                                         replace=False)
        long_idx = dist_i_argsort[-num_candidates_for_choice_long:][long_rand_idx]
        
        
        # choose short edges (points)
        num_edges_short = np.min((num_candidates_for_choice_short, num_edges_short))
        short_rand_idx = np.random.choice(num_candidates_for_choice_short, num_edges_short,\
                                     replace=False)
        short_idx = dist_i_argsort[1:num_candidates_for_choice_short+1][short_rand_idx]
        
        # collect points' indices to dict
#         dict_points[i] = np.unique(np.concatenate(\
#             (long_idx, short_idx), axis=0).reshape(1,-1)[0]).tolist()
        dict_points[i] = np.concatenate(\
            (long_idx, short_idx), axis=0).reshape(1,-1)[0].tolist()

    
    return dict_points

def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    # допишите ваш код здесь 
    ii = 0
    jj = 0
    visited_points_dict = OrderedDict()
    seen_points_dict = OrderedDict()
    
    N_docs = np.shape(all_documents)[0]
    
    num_point_i = 0
    while (num_start_points > num_point_i) or (len(seen_points_dict) < search_k):
        jj += 1
        
#         new_start_point = np.random.choice(range(N_docs), 1)[0] ## EXTREMELY SLOW
        new_start_point = np.random.randint(0, N_docs)
        if new_start_point in visited_points_dict.keys():
#             num_point_i += 1
            continue
            
        if new_start_point in seen_points_dict.keys():
            cur_dist = seen_points_dict[new_start_point]
        else:
            cur_dist = dist_f(pointA=query_point,\
                documents=all_documents[[new_start_point], ])[0][0]
        
        visited_points_dict[new_start_point] = cur_dist
        is_new_point_added = False
        
        while not is_new_point_added:
            ii += 1
            
            new_edges_num = graph_edges[new_start_point]
            prev_new_start_point = new_start_point
            
            for edge_num in new_edges_num:
                if edge_num not in seen_points_dict.keys():
                    new_distance = dist_f(pointA=query_point,\
                        documents=all_documents[[edge_num], ])[0][0]
                    seen_points_dict[edge_num] = new_distance
                
                if cur_dist > seen_points_dict[edge_num]:
                    new_start_point = edge_num
                    cur_dist = seen_points_dict[edge_num]
            
            if prev_new_start_point == new_start_point:
                num_point_i += 1
                break
            
            if new_start_point in visited_points_dict.keys():
                num_point_i += 1
                break
            else:
                visited_points_dict[new_start_point] = cur_dist
    
    print(ii)
    print(jj)  
    top_k_points_i = np.argsort(list(seen_points_dict.values()))[:search_k]
    top_k_points_nums = np.array(list(seen_points_dict.keys()))[top_k_points_i]
    return top_k_points_nums
