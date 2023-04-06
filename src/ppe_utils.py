import numpy as np
from numba.typed import Dict as nb_dict
from numba.core import types
import numba as nb
import scipy as sp
import sys
from dataclasses import dataclass
from typing import Dict, List, Any

from sklearn.utils import murmurhash3_32 as murmurhash

MAX_EPSILON_PRECISION = np.float64(1e-7)
MIN_EPSILON_PRECISION = np.float64(1e-15)


@dataclass
class CSRGraph:
    indptr = np.ndarray
    indices = np.ndarray
    data = np.ndarray


class PPREstimator:
    """Incremental PPR Calulation"""

    def __init__(
        self,
        max_node_num: int,
        track_nodes: np.ndarray,  # uint32
        alpha: float = 0.15,
        ppr_algo: str = None,
    ):
        self.max_node_num = max_node_num
        self.track_nodes = np.unique(track_nodes)
        self.track_nodes_set = set(list(self.track_nodes.flatten()))

        self.dict_p_arr = nb_dict.empty(
            key_type=types.uint64, value_type=types.float64[:]
        )
        self.dict_r_arr = nb_dict.empty(
            key_type=types.uint64, value_type=types.float64[:]
        )
        self.ppr_algo = ppr_algo
        self.alpha = alpha

        self._init_dict_p(self.track_nodes)
        self._init_dict_r(self.track_nodes)

    def _init_dict_p(
        self,
        init_node_ids: np.ndarray,
    ):
        """init typed dict of p, keyed by tracked node id"""
        for _u in init_node_ids:
            self.dict_p_arr[_u] = np.zeros(
                self.max_node_num,
                dtype=np.float64,
            )

        return None

    def _init_dict_r(
        self,
        init_node_ids: np.ndarray,
    ):
        """init typed dict of r, keyed by tracked node id.
        r[i] = 1.0 if i==u, else 0.0.

        """
        if self.ppr_algo == "ista":
            for _u in init_node_ids:
                self.dict_r_arr[_u] = np.zeros(
                    self.max_node_num,
                    dtype=np.float64,
                )
                self.dict_r_arr[_u][_u] = 1.0
        else:
            for _u in init_node_ids:
                self.dict_r_arr[_u] = np.zeros(
                    self.max_node_num,
                    dtype=np.float64,
                )
                self.dict_r_arr[_u][_u] = 1.0
        return None

    def __str__(self):
        if self.ppr_algo == "forward_push":
            printable_str = (
                f"PPR Algorithm:\t{self.ppr_algo}\n"
                f"tracked nodes:\t{len(self.dict_p_arr)}\n"
                f"tracked pprs:\t{self.dict_p_arr}"
                f"incremental update:\t{self.incrmt_ppr}"
            )
        elif self.ppr_algo == "power_iteration":
            printable_str = (
                f"PPR Algorithm:\t{self.ppr_algo}\n"
                f"tracked nodes:\t{len(self.dict_p_arr)}\n"
                f"tracked pprs:\t{self.dict_p_arr}"
                f"incremental update:\t{self.incrmt_ppr}"
            )
        elif self.ppr_algo == "ista":
            printable_str = (
                f"PPR Algorithm:\t{self.ppr_algo}\n"
                f"tracked nodes:\t{len(self.dict_p_arr)}\n"
                f"tracked pprs:\t{self.dict_p_arr}"
                f"incremental update:\t{self.incrmt_ppr}"
            )

        else:
            raise NotImplementedError
        return printable_str

    def add_nodes_to_ppr_track(
        self,
        new_nodes: np.ndarray,
    ):
        """dynamically add tracked nodes to the ppr estimator"""

        if new_nodes.shape[0] == 0:
            return None

        new_nodes_uni = np.unique(new_nodes.astype(np.uint64))
        should_add_node_ids = []
        for node_id in new_nodes_uni:
            node_id = np.uint64(node_id)
            if node_id not in self.track_nodes_set:
                should_add_node_ids.append(node_id)
                self.track_nodes_set.add(node_id)
        new_tracked_node_ids = np.array(should_add_node_ids).astype(np.uint64)
        # print(
        #     f"add new {new_tracked_node_ids.shape[0]} nodes to"
        #     f" tracking{new_tracked_node_ids}"
        # )
        self._init_dict_p(new_tracked_node_ids)
        self._init_dict_r(new_tracked_node_ids)

        self.track_nodes = np.hstack((self.track_nodes, new_tracked_node_ids))

    def update_ppr(
        self,
        csr_indptr: np.ndarray,
        csr_indices: np.ndarray,
        csr_data: np.ndarray,
        out_degree: np.ndarray,
        alpha: float,
        *args,
        **kwargs,
    ):
        """update ppr given current sparse graph
        using method, which in-place modifies dict_p_arr):
        - Local push
        - Power Iteration
        - L1 regularization solvers (ista, others are not implemented)

        Args:
            csr_indptr (np.ndarray): csr matrix indptr array
            csr_indices (np.ndarray): csr matrix indices array
            csr_data (np.ndarray): csr matrix data array
            out_degree (np.ndarray): grpah out-degree array
            in_degree (np.ndarray): graph in-degree array
            alpha (float): teleport probability in PPR

        Raises:
            NotImplementedError
        """
        # ops
        ppr_updates_metric = {
            "power_iter_ops": np.uint64(0),
            "local_push_pos_push": np.uint64(0),
            "local_push_neg_push": np.uint64(0),
            "ista_ops": np.uint64(0),
        }

        if self.ppr_algo == "forward_push":
            # push specific params:
            beta = 0.0 if "beta" not in kwargs else kwargs["beta"]
            assert "init_epsilon" in kwargs, "init_epsilon?"
            init_epsilon = kwargs["init_epsilon"]

            # init_epsilon = (
            #     1e-6
            #     if "init_epsilon" not in kwargs
            #     else kwargs["init_epsilon"]
            # )

            # print(self.dict_p_arr.keys())
            # print(self.dict_r_arr.keys())
            (
                _,
                local_push_pos_push,
                local_push_neg_push,
                _,
            ) = forward_push_routine(
                self.max_node_num,
                self.track_nodes,
                csr_indices,
                csr_indptr,
                csr_data,
                out_degree,
                self.dict_p_arr,
                self.dict_r_arr,
                alpha,
                beta,
                init_epsilon,
            )
            ppr_updates_metric["local_push_pos_push"] = local_push_pos_push
            ppr_updates_metric["local_push_neg_push"] = local_push_neg_push

        elif self.ppr_algo == "power_iteration":
            beta = 0.0 if "beta" not in kwargs else kwargs["beta"]
            assert "init_epsilon" in kwargs, "init_epsilon?"
            assert "power_iteration_max_iter" in kwargs, "max_iter?"
            init_epsilon = kwargs["init_epsilon"]
            max_iter = kwargs["power_iteration_max_iter"]
            # init_epsilon = (
            #     1e-6
            #     if "init_epsilon" not in kwargs
            #     else kwargs["init_epsilon"]
            # )
            # max_iter: int = (
            #     5000
            #     if "power_iteration_max_iter" not in kwargs
            #     else kwargs["power_iteration_max_iter"]
            # )
            (power_iter_ops, _, _, _,) = power_iteration_routine(
                self.max_node_num,
                self.track_nodes,
                csr_indices,
                csr_indptr,
                csr_data,
                out_degree,
                self.dict_p_arr,
                self.dict_r_arr,
                alpha,
                beta,
                init_epsilon,
                max_iter=max_iter,
            )
            ppr_updates_metric["power_iter_ops"] = power_iter_ops

        elif self.ppr_algo == "ista":
            # assert "ista_rho" in kwargs, "ISTA: ista_rho is not assigned!"
            # assert "ista_max_iter" in kwargs, "ISTA: ista_max_iter miss!"
            # assert "ista_erly_brk_tol" in kwargs, "ISTA: early exit-tol miss!"
            init_epsilon = kwargs["init_epsilon"]

            (_, _, _, ista_ops) = ista_ppr_routine(
                self.max_node_num,
                self.track_nodes,
                csr_indices,
                csr_indptr,
                csr_data,
                out_degree,
                self.dict_p_arr,
                self.dict_r_arr,  # no r-vec anymore
                alpha_norm=alpha,  # alpha of no-lazy random walk
                max_iter=kwargs["ista_max_iter"],
                init_epsilon=init_epsilon,
            )
            ppr_updates_metric["ista_ops"] = ista_ops

        else:
            raise NotImplementedError

        return ppr_updates_metric


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def forward_push_routine(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha: float,
    beta: float,
    init_epsilon: float,
):
    """Calculate PPR based on Andersen's local push algorithm with
    Numba multi-thread acceleration.


    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha (float): Teleport probability in PPR vector
        beta (float): = 0,
        init_epsilon (float): for the first push.
    """
    # print("1. start push")
    # eps_prime = np.float64(init_epsilon / node_degree.sum())
    epsilon = np.float64(init_epsilon / node_degree.sum())

    total_pos_ops = np.zeros(query_list.shape[0], dtype=np.uint64)
    total_neg_ops = np.zeros(query_list.shape[0], dtype=np.uint64)

    for i in nb.prange(query_list.shape[0]):
        # Multi-thread if using numba's nonpython mode. No GIL
        # Adaptive push according to p_s
        s = query_list[i]

        p_s: np.ndarray = dict_p_arr[s]
        r_s: np.ndarray = dict_r_arr[s]

        # 1: Positive FIFO Queue
        # r_s[v] > epsilon*node_degree[v]
        q_pos = nb.typed.List()
        q_pos_ptr = np.uint64(0)
        # NOTE: Numba Bug: element in set() never returns!!
        # SEE: https://github.com/numba/numba/issues/6543
        q_pos_marker = nb.typed.Dict.empty(
            key_type=nb.types.uint64,
            value_type=nb.types.boolean,
        )

        # scan all residual in r_s, select init for pushing
        # Or only maintain the top p_s[v] nodes for pushing? since it's
        # likely affect top-ppr

        for v in nb.prange(r_s.shape[0]):
            # v = v
            if r_s[v] > epsilon * node_degree[v]:
                q_pos.append(v)
                q_pos_marker[v] = True

        # Positive: pushing pushing!
        while np.uint64(len(q_pos)) > q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr += np.uint64(1)
            q_pos_marker.pop(u)
            deg_u = node_degree[u]
            r_s_u = r_s[u]

            if r_s_u > epsilon * deg_u:  # for positive
                p_s[u] += alpha * r_s_u
                push_residual = np.float64((1 - alpha) * r_s_u / deg_u)
                _v = indices[indptr[u] : indptr[u + 1]]
                _w = data[indptr[u] : indptr[u + 1]]
                for _ in range(_v.shape[0]):
                    total_pos_ops[i] += np.uint64(1)
                    v = _v[_]
                    w_u_v = np.float64(_w[_])
                    # should multply edge weights.
                    r_s[v] += np.float64(push_residual * w_u_v)
                    if v not in q_pos_marker:
                        q_pos.append(v)
                        q_pos_marker[v] = True
                # r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float64(0.0)

        # Add dummy +=0 trick to avoid numba bug when convert while into for.
        # SEE: https://github.com/numba/numba/issues/5156
        q_pos_ptr += np.uint64(0)

        # 2: Negative FIFO Queue
        # r_s[v] < -epsilon*node_degree[v]
        q_pos = nb.typed.List()
        q_pos_ptr = np.uint64(0)
        q_pos_marker = nb.typed.Dict.empty(
            key_type=nb.types.uint64,
            value_type=nb.types.boolean,
        )

        # scan all residual in r_s, select init for pushing
        for v in nb.prange(r_s.shape[0]):
            # v = v
            if r_s[v] < -epsilon * node_degree[v]:  # for negative
                q_pos.append(v)
                q_pos_marker[v] = True

        # Negative: pushing pushing!
        while np.uint64(len(q_pos)) > q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr += np.uint64(1)
            q_pos_marker.pop(u)
            deg_u = node_degree[u]
            r_s_u = r_s[u]
            if r_s_u < -epsilon * deg_u:  # for negative
                p_s[u] += alpha * r_s_u
                push_residual = np.float64((1 - alpha) * r_s_u / deg_u)
                _v = indices[indptr[u] : indptr[u + 1]]
                _w = data[indptr[u] : indptr[u + 1]]
                for _ in range(_v.shape[0]):
                    total_neg_ops[i] += np.uint64(1)
                    v = _v[_]
                    w_u_v = np.float64(_w[_])
                    # should multply edge weights.
                    r_s[v] += np.float64(push_residual * w_u_v)
                    if v not in q_pos_marker:
                        q_pos.append(v)
                        q_pos_marker[v] = True
                # r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float64(0.0)
        # Add dummy +=0 trick to avoid numba bug when convert while into for.
        # SEE: https://github.com/numba/numba/issues/5156
        q_pos_ptr += np.uint64(0)

    return [
        np.uint64(0),
        np.sum(total_pos_ops),
        np.sum(total_neg_ops),
        np.uint64(0),
    ]


@nb.njit(cache=True, parallel=False, fastmath=True, nogil=True)
def vsmul_csr(x, A, iA, jA):
    power_iter_ops = np.uint64(0)
    res = np.zeros(x.shape[0], dtype=np.float64)
    for row in nb.prange(len(iA) - 1):
        for i in nb.prange(iA[row], iA[row + 1]):
            data = A[i]
            row_i = row
            col_j = jA[i]
            res[col_j] += np.float64(data * x[row_i])
            power_iter_ops += 1
    return res.astype(np.float64), power_iter_ops


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def __power_iter_numba(
    N,
    query_list,
    indices,
    indptr,
    data,
    dict_p_arr,
    alpha,
    max_iter,
    tol,
):
    num_nodes = np.uint64(query_list.shape[0])
    power_iter_ops = np.uint64(0)
    for i in nb.prange(num_nodes):
        s = query_list[i]
        power_iter_ops_local = np.uint64(0)
        # initial vector
        x: np.ndarray = dict_p_arr[s]
        if x.sum() != 0.0:
            x /= x.sum()
        # Personalization vector
        p = np.zeros(np.int64(N), dtype=np.float64)
        p[s] = np.float64(1.0)
        # power iteration: make up to max_iter iterations
        for _i in range(max_iter):
            xlast = x
            res, __power_iter_ops = vsmul_csr(x, data, indptr, indices)
            power_iter_ops_local += __power_iter_ops
            x = ((1.0 - alpha) * res + alpha * p).astype(np.float64)
            # check convergence, l1 norm
            err = np.absolute(x - xlast).sum()
            # print(err)
            if err < tol:
                dict_p_arr[s] = x.astype(np.float64)
                power_iter_ops += power_iter_ops_local
                break
    return np.uint64(power_iter_ops)


def power_iteration_routine(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha: float,
    beta: float,
    init_epsilon: float,
    max_iter: int,
    *args,
    **kwargs,
):
    """Calculate PPR based on power iteration algorithm


    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha (float): Teleport probability in PPR vector
        beta (float): = 0,
        init_epsilon (float): early exit
    """

    tol: int = init_epsilon
    # nodelist = np.arange(N, dtype=np.uint64)
    # dangling = None

    A = sp.sparse.csr_matrix((data, indices, indptr), shape=(N, N))
    S = np.array(A.sum(axis=1), dtype=np.float64).flatten()
    S[S != 0.0] = 1.0 / S[S != 0.0]  # 1 over degree
    Q = sp.sparse.spdiags(S.T, 0, *A.shape, format="csr")
    A = Q * A
    # ensure no dangling nodes.
    if S[S == 0.0].shape[0] == 0:
        print(
            "warning: the graph has dangling node "
            "If the graph is undirected, it should be fine "
            "since the dangling node is isolated."
        )
    # assert S[S == 0.0].shape[0] == 0, "graph has dangling nodes"

    power_iter_ops = __power_iter_numba(
        np.uint64(N),
        np.uint64(query_list),
        A.indices,
        A.indptr,
        A.data,
        dict_p_arr,
        np.float64(alpha),
        np.uint64(max_iter),
        np.float64(tol),
    )

    return power_iter_ops, np.uint64(0), np.uint64(0), np.uint64(0)


# global gradient
@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def ista_ppr_routine(
    N: int,
    query_list: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    node_degree: np.ndarray,
    dict_p_arr: nb_dict,
    dict_r_arr: nb_dict,
    alpha_norm: np.float64,
    max_iter: int,
    init_epsilon: np.float64 = np.float64(1e-7),
):
    """calculate PPR vector using ISTA in the lens of L-1 regularized
    optimization.

    Args:
        N (int): total number of nodes |V|
        query_list (np.ndarray): the queried nodes for ppr
        indices (np.ndarray): indices arr for csr graph
        indptr (np.ndarray): indptr arr for csr graph
        data (np.ndarray): edge_w arr for csr graph
        node_degree (np.ndarray): node out-degree arr
        dict_p_arr (nb_dict): estimated PPR vector indexed by node-id
        dict_r_arr (nb_dict): estimated Residual vector ind by node-id
        alpha_norm (float): teleport proba of random walk (not lazy walk)
        rho (float): the l1 regularization param
        max_iter (int): the max iteration
        early_break_tol (float): the early exit tolerance of ppv
            (per node).
    """
    # the corresponding alpha of lazy random walk coverted from normal
    # random walk .
    total_ista_ops = np.zeros(len(query_list), dtype=np.uint64)

    alpha = alpha_norm
    eta = np.float64(1.0 / (2.0 - alpha))
    # eta = np.float64(1.0)
    # D^{-1/2}
    sqrt_d_out = np.sqrt(node_degree).astype(np.float64)
    nsqrt_d_out = np.zeros_like(sqrt_d_out).astype(np.float64)
    for i in nb.prange(N):
        _sqrt_d = sqrt_d_out[i]
        if _sqrt_d > 0.0:
            nsqrt_d_out[i] = np.float64(1.0) / _sqrt_d

    # optimality condition
    # init_epsilon = init_epsilon * 1e-3
    opt_cond_val = np.float64(init_epsilon / node_degree.sum()) * sqrt_d_out

    for s_i in nb.prange(len(query_list)):
        s_id = query_list[s_i]
        ista_ops = np.uint64(0)
        e_s = np.zeros(N, dtype=np.float64)
        e_s[s_id] = 1.0
        prev_ppv = np.zeros(N, dtype=np.float64)

        p = dict_p_arr[s_id]  # x = np.zeros(N, dtype=np.float64)
        x = np.multiply(nsqrt_d_out, p)
        # x = np.zeros(N, dtype=np.float64)

        # r = dict_r_arr[s_id]  # grad_f_q = np.zeros(N, dtype=np.float64)
        # grad_f = np.multiply(nsqrt_d_out, r)  # working
        grad_f = np.zeros(N, dtype=np.float64)
        z = np.zeros(N, dtype=np.float64)

        for iter_num in range(max_iter):
            # eval gradient now Wx+b
            for i in range(N):
                dAd = np.float64(0.0)
                for ptr in range(indptr[i], indptr[i + 1]):
                    j = indices[ptr]
                    dAd += nsqrt_d_out[i] * data[ptr] * nsqrt_d_out[j] * x[j]
                grad_f[i] = (
                    x[i] - (1 - alpha) * dAd - alpha * nsqrt_d_out[i] * e_s[i]
                )
                z[i] = x[i] - eta * grad_f[i]

            # solve proximal
            for i in range(N):
                if z[i] > opt_cond_val[i]:
                    ista_ops += np.uint64(1)
                    x[i] = z[i] - opt_cond_val[i]
                elif np.abs(z[i]) < opt_cond_val[i]:
                    x[i] = np.float64(0)
                elif z[i] < -opt_cond_val[i]:
                    ista_ops += np.uint64(1)
                    x[i] = z[i] + opt_cond_val[i]
                else:
                    pass

            ppv = np.multiply(sqrt_d_out, x)
            if (
                np.absolute(ppv - prev_ppv).sum() < 0.01 * init_epsilon
                # np.multiply(grad_f, sqrt_d_out).sum() < init_epsilon
                or iter_num == max_iter - 1
            ):
                dict_p_arr[s_id] = ppv
                dict_r_arr[s_id] = np.multiply(grad_f, sqrt_d_out)
                total_ista_ops[s_i] += ista_ops
                break
            prev_ppv = ppv

    return (np.uint64(0), np.uint64(0), np.uint64(0), np.sum(total_ista_ops))


def get_hash_LUT(n: int, dim: int = 512, rnd_seed: int = 0):
    """get the cache of dim-sign, dim-map from hash function

    Args:
        n (int): the total number of input dim size (ppr vector length)
        dim (int, optional): the out dimension. Defaults to 512.
        rnd_seed (int, optional): the hash random seed. Defaults to 0.

    Returns:
        np.ndarray: the dimension mapping of ppr-vector to out-vector
        np.ndarray: the sign (+/-) mapping of ppr-vector to out-vector
    """

    node_id_2_dim_id: np.ndarray = np.zeros(n, dtype=np.int32)
    node_id_2_sign: np.ndarray = np.zeros(n, dtype=np.int8)
    for _ in range(n):
        dim_id = murmurhash(_, seed=rnd_seed, positive=True) % dim
        sign = murmurhash(_, seed=rnd_seed, positive=True) % 2
        node_id_2_dim_id[_] = dim_id
        node_id_2_sign[_] = 1 if sign == 1 else -1
    return node_id_2_dim_id, node_id_2_sign


@nb.njit(cache=True, parallel=True, fastmath=True, nogil=True)
def get_hash_embed(
    node_id_2_dim_id: np.ndarray,
    node_id_2_sign: np.ndarray,
    out_dim: int,
    q_nodes: np.ndarray,
    ppv_dim: int,
    indices: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
):
    """get hash embeddings from ppvs and pre-cached dimension/sign
    mapping.
    The input indices/indptr/data is from the row-sliced csr_mat.

    Args:
        node_id_2_dim_id (np.ndarray): the dimension mapping of ppr
            vector to out-vector
        node_id_2_sign (np.ndarray): the sign (+/-) mapping of ppr
            vector to out-vector
        out_dim (int): the dimension of output embedding vectors.
        q_nodes (np.ndarray): the queried node ids.
        ppv_dim (int): the dimension of original ppr vector
        indices (np.ndarray): the csr.indices of ppr matrix
        indptr (np.ndarray): the csr.indptr of ppr matrix
        data (np.ndarray): the csr.data of ppr matrix

    Returns:
        np.ndarray: the queried hash embeddings from ppr.
    """
    emb_mat: np.ndarray = np.zeros(
        (q_nodes.shape[0], out_dim),
        dtype=np.float64,
    )
    for i in nb.prange(q_nodes.shape[0]):  # for all nodes.
        js = indices[indptr[i] : indptr[i + 1]]
        vals = data[indptr[i] : indptr[i + 1]]
        # emb_vec = emb_mat[i, :]
        for j, val in zip(js, vals):
            _map_out_dim = node_id_2_dim_id[j]
            # emb_vec[_map_out_dim]
            emb_mat[i, _map_out_dim] += node_id_2_sign[j] * np.maximum(
                np.float64(0.0),
                np.float64(np.log(val * ppv_dim)),
            )
    return emb_mat
