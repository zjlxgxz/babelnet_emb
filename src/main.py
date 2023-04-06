import argparse
import os
from os.path import join as os_join
import time
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm as tqdm
from ppe_utils import PPREstimator, get_hash_LUT, get_hash_embed
from scipy.spatial import distance
from glob import glob


def load_graph(graph_path, max_node_num):
    # read csv file line by line and convert to sparse matrix
    triplet_path = os_join(graph_path, "triples.csv")

    rows = []
    cols = []
    data = []

    with open(triplet_path) as f:
        for line in tqdm(f):
            # split line by comma
            line = line.split(",")
            # remove whitespace
            line = [x.strip() for x in line]
            u, r, r_w, v = (
                int(line[0]),
                int(line[1]),
                float(line[2]),
                int(line[3]),
            )

            if r_w == 0.0:
                # assign small value to zero weight edges
                r_w = 1e-5

            if r_w == 0.0:
                # drop the zero weight edges
                continue

            if u == v:
                # drop the self loop
                continue

            rows.append(u)
            cols.append(v)
            data.append(r_w)

            # convert to undir
            rows.append(v)
            cols.append(u)
            data.append(r_w)

            if len(data) == 500_000:
                break
            # print(u, r, r_w, v)
            # exit()

    return sp.sparse.csr_matrix(
        (data, (rows, cols)), shape=(max_node_num, max_node_num)
    )


def load_query_node_pairs(query_node_path):
    pairs = []
    with open(query_node_path, "r") as f:
        for line in f:
            line = line.split(",")
            line = [x.strip() for x in line]
            u, v = line[0], line[1]

            if u == "" or v == "":
                continue
            pairs.append(
                [int(u), int(v)],
            )
    pairs = np.array(pairs)

    return {
        "pairs": pairs,
        "unique_nodes": np.unique(pairs.flatten()),
    }


def get_node_embedding(
    ppr_estimator,
    ppe_hashcache_id,
    ppe_hash_cache_sign,
    out_dim,
):
    # gather embeddings and convert to sparse
    # return dictionary of node embeddings and node ids.
    ppv_list = []
    ppv_track_node_list = []
    for track_node_id in ppr_estimator.track_nodes_set:
        ppv_track_node_list.append(track_node_id)
        ppv_list.append(ppr_estimator.dict_p_arr[track_node_id])
    ppv_track_node_list = np.array(ppv_track_node_list)
    ppv_mat_dense = np.array(ppv_list, dtype=np.float32)
    ppv_mat_sparse = sp.sparse.csr_matrix(ppv_mat_dense)

    indices = ppv_mat_sparse.indices
    indptr = ppv_mat_sparse.indptr
    data = ppv_mat_sparse.data

    ppv_dim = ppv_mat_dense.shape[-1]

    hash_emb: np.ndarray = get_hash_embed(
        ppe_hashcache_id,
        ppe_hash_cache_sign,
        out_dim,
        ppv_track_node_list,
        ppv_dim,
        indices,
        indptr,
        data,
    )

    return {
        "hash_emb": hash_emb,
        "ppv_track_node_list": ppv_track_node_list,
    }


def get_distane_measure(
    all_pairs, hash_emb, emb_node_id_to_emb_idx, proc_time
):
    """get pair distance measures"""
    records = []
    for i in range(all_pairs.shape[0]):
        pair = all_pairs[i]
        u, v = pair[0], pair[1]
        emb_idx_u = emb_node_id_to_emb_idx[u]
        emb_idx_v = emb_node_id_to_emb_idx[v]
        emb_u = hash_emb[emb_idx_u, :]
        emb_v = hash_emb[emb_idx_v, :]
        record = {
            "u": u,
            "v": v,
            "emb_u": emb_u,
            "emb_v": emb_v,
            "proc_time": proc_time,
            "l2-dist": distance.euclidean(emb_u, emb_v),
            "cosine-dist": distance.cosine(emb_u, emb_v),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def main(args):
    all_query_file_paths = glob(args.query_node_path)
    print(f"Number of all query files: {len(all_query_file_paths)}")

    # load graph
    max_node_num = 13_801_844  # hardcoded babelnet size
    graph_csr = load_graph(args.graph_path, max_node_num)  # slow

    # preproess hash embedding mapping
    ppe_node_id_2_dim_id, ppe_node_id_2_sign = get_hash_LUT(
        max_node_num,
        args.out_dim,
        rnd_seed=args.rs,
    )

    for query_file_path in tqdm(all_query_file_paths):
        # load query node
        pairs_dict = load_query_node_pairs(query_file_path)
        track_nodes = pairs_dict["unique_nodes"]
        all_pairs = pairs_dict["pairs"]

        print(f"all query nodes:{track_nodes.shape[0]}")
        print(f"all query pairs:{all_pairs.shape}")

        # init ppr calculation
        ppr_estimator = PPREstimator(
            max_node_num=max_node_num,
            track_nodes=track_nodes,
            alpha=args.ppr_alpha,
            ppr_algo=args.ppr_algo,
        )

        # ppr calculation hyper-parameter
        ppr_algo_param_dict = {
            "init_epsilon": np.float64(args.ppr_epsilon),
            "ista_max_iter": np.uint64(5000),
        }

        # start calculate embeddings
        t_start = time.time()
        ppr_updates_metric = ppr_estimator.update_ppr(
            graph_csr.indptr,
            graph_csr.indices,
            graph_csr.data,
            np.squeeze(np.asarray(graph_csr.sum(axis=1))),  # out degree
            alpha=args.ppr_alpha,
            **ppr_algo_param_dict,
        )
        print(f"PPR update time: {(time.time() - t_start):.4f}")
        # print(ppr_updates_metric)

        # get node embeddings (2d arr) and corresponding node ids
        node_emb_result_dict = get_node_embedding(
            ppr_estimator,
            ppe_node_id_2_dim_id,
            ppe_node_id_2_sign,
            out_dim=args.out_dim,
        )

        hash_emb = node_emb_result_dict["hash_emb"]
        emb_node_id = node_emb_result_dict["ppv_track_node_list"]
        emb_node_id_to_emb_idx = {}
        for i in range(emb_node_id.shape[0]):
            node_id = emb_node_id[i]
            emb_node_id_to_emb_idx[node_id] = i

        # print("results: ")
        # print(node_emb_result_dict.keys())
        # print(hash_emb.shape)
        # print(emb_node_id.shape)

        # get distance for each pair
        proc_time = time.time() - t_start
        df_pair_distance = get_distane_measure(
            all_pairs,
            hash_emb,
            emb_node_id_to_emb_idx,
            proc_time,
        )
        print(f"All elapsed time: {proc_time:.4f}")

        # save results to json
        query_node_folder = os.path.dirname(query_file_path)
        query_node_filename = os.path.basename(query_file_path).replace(
            ".csv", ".pkl"
        )
        output_res_path = os_join(query_node_folder, query_node_filename)
        print(
            f"output result file to: {output_res_path}. Please use"
            " pandas.read_pickle() to read"
        )
        df_pair_distance.to_pickle(output_res_path)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--graph_path",
        type=str,
        default="/home/xingzguo/projects_data/babelnet_emb",
    )
    arg.add_argument(
        "--query_node_path",
        type=str,
        default=(
            "/home/xingzguo/projects_data/babelnet_emb/graph_query/*/*.csv"
        ),
    )
    arg.add_argument("--ppr_algo", type=str, default="ista")
    arg.add_argument("--ppr_alpha", type=float, default=0.5)  # alpha in [0, 1]
    arg.add_argument(
        "--ppr_epsilon", type=float, default=1e-3
    )  # epsilon in 1e-1 ~ 1e-10
    arg.add_argument("--out_dim", type=int, default=1024)  # embedding dim
    arg.add_argument("--rs", type=int, default=621)
    arg.add_argument("--use_verbose", action="store_true")
    args = arg.parse_args()
    main(args)

    # command example [high-precision, slow ]
    # python main.py --ppr_epsilon 1e-10  --ppr_alpha 0.2

    # command example [low-precision, fast ]
    # python main.py --ppr_epsilon 1e-2  --ppr_alpha 0.2

    # command example [low-precision, faster ]
    # python main.py --ppr_epsilon 1e-2  --ppr_alpha 0.8
