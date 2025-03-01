import argparse
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)


def find_non_overlap_idx(adj_mats: np.ndarray, half_overlap_width: int = 10):
    """
    Find the non-overlapping indexes for different quantiles of the correlation matrix difference.
    Output the most indexes which are not overlap to each other by give overlap width.
    """
    adj_mats_diff_list = np.abs((adj_mats[1:] - adj_mats[0:-1]).sum(axis=1).sum(axis=1))
    non_overlap_idxs_quan = {"quan": None, "non_overlap_idxs": None}
    max_num_non_overlap_idxs = 0
    for quan in np.linspace(0.5, 1, 51):
        quan = np.round(quan, 2)
        over_threshold_idx_list = np.where(adj_mats_diff_list > np.quantile(adj_mats_diff_list, quan))[0]
        idx_pairs = list(combinations(over_threshold_idx_list, 2))
        overlap_idxs = set()
        # check if there exists any pair that satisfies the condition
        for idx, idx2 in idx_pairs:
            is_within_range = abs(idx - idx2) <= half_overlap_width
            if is_within_range:
                overlap_idxs.update([idx, idx2])

        non_overlap_idxs = set(over_threshold_idx_list) - overlap_idxs
        if len(non_overlap_idxs) > max_num_non_overlap_idxs:
            max_num_non_overlap_idxs = len(non_overlap_idxs)
            non_overlap_idxs_quan["quan"] = quan
            non_overlap_idxs_quan["non_overlap_idxs"] = non_overlap_idxs

        if quan > 0.94:
            logger.debug(f"At quantile threshold: {quan}, number of over threshold index: {len(over_threshold_idx_list)}, number of overlap indexes: {len(overlap_idxs)}, number of non-overlap indexes: {len(non_overlap_idxs)}")
            logger.debug(f"over_threshold_idx_list:\n{over_threshold_idx_list}")
            logger.debug(f"non overlap indexes:\n{non_overlap_idxs}")
            logger.debug("-"*50)

    non_overlap_idxs_quan["non_overlap_idxs"] = np.array(sorted(non_overlap_idxs_quan["non_overlap_idxs"])) + 1
    logger.info(f"non_overlap_idxs_quan:{non_overlap_idxs_quan}")

    return non_overlap_idxs_quan


def find_top_diff_idx(adj_mats: np.ndarray, quan: float = 0.9):
    """
    TODO: FINISH Description
    """
    adj_mats_diff_list = np.abs((adj_mats[1:] - adj_mats[0:-1]).sum(axis=1).sum(axis=1))
    over_threshold_idx_list = np.where(adj_mats_diff_list > np.quantile(adj_mats_diff_list, quan))[0]

    return np.sort(over_threshold_idx_list) + 1


def gen_cord_cpd_data(adj_mats: np.ndarray, nodes_mats: np.ndarray, idx_list: np.ndarray, idx_mode: str, data_sp_mode: str, retrieve_t_width: int = 10):
    edges_mats_list = []
    nodes_mats_list = []
    labels_list = []

    for idx in idx_list:
        start_pos = np.random.randint(2, retrieve_t_width, size=1)[0]
        begin_idx = min(max(0, (idx - start_pos)), adj_mats.shape[0]-retrieve_t_width)
        end_idx = min((begin_idx+retrieve_t_width), adj_mats.shape[0])
        label_idx = idx - begin_idx
        if idx_mode == "non_overlap":
            assert ((idx_list[1:] - idx_list[:-1]) > retrieve_t_width).all(), "Containing indexes is overlap, please change the `idx_mode` or double check the elements of `idx_list`."
            #proc_adj_mats = adj_mats
            #proc_nodes_mats = nodes_mats
            max_diff_idx = label_idx
        elif idx_mode == "overlap":
            #trans_idx_list = idx_list - idx
            #overlap_mask = np.logical_and(np.logical_and(-retrieve_t_width < trans_idx_list, trans_idx_list < retrieve_t_width), trans_idx_list != 0)
            #overlap_idxs = idx_list[overlap_mask]
            #non_overlap_mask = np.ones(len(adj_mats), dtype=bool)
            #non_overlap_mask[overlap_idxs] = False
            #proc_adj_mats = adj_mats[non_overlap_mask]
            #proc_nodes_mats = nodes_mats[non_overlap_mask]
            #proc_adj_mats = adj_mats
            #proc_nodes_mats = nodes_mats
            #max_diff_idx = np.argmax(np.abs(np.diff(proc_adj_mats[begin_idx:end_idx], axis=0).sum(axis=1).sum(axis=1))) + 1
            max_diff_idx = np.argmax(np.abs(np.diff(adj_mats[begin_idx:end_idx], axis=0).sum(axis=1).sum(axis=1))) + 1

        if label_idx == max_diff_idx and end_idx <= adj_mats.shape[0]:
            edges_mats_list.append(adj_mats[begin_idx:end_idx])
            nodes_mats_list.append(nodes_mats[begin_idx:end_idx])
            labels_list.append(label_idx)
            logger.debug(f"original idx: {idx}, retrieve indexes range: {begin_idx}~{end_idx}, label: {label_idx}, max_diff_idx: {max_diff_idx}")
            logger.debug(f"adj_mats.shape:{adj_mats.shape}")
            #logger.debug(f"adj_mats.shape:{adj_mats.shape}, proc_adj_mats.shape:{proc_adj_mats.shape}")
            #logger.debug(f"overlap idxs:\n{overlap_idxs if 'overlap_idxs' in locals() else None}")
            #logger.debug(f"adj_mats[32:36, :2, :5]:\n{adj_mats[32:36, :2, :5]}\nproc_adj_mats[32:36, :2, :5]:\n{proc_adj_mats[32:36, :2, :5]}")
            logger.debug(f"np.diff(adj_mats[begin_idx:end_idx]):\n{np.diff(adj_mats[begin_idx:end_idx], axis=0).sum(axis=1).sum(axis=1)}")
            #logger.debug(f"np.diff(proc_adj_mats[begin_idx:end_idx]):\n{np.diff(proc_adj_mats[begin_idx:end_idx], axis=0).sum(axis=1).sum(axis=1)}")

    labels_arr = np.array(labels_list)
    edges_mats = np.stack(edges_mats_list)
    feature_mats = np.stack(nodes_mats_list)
    logger.info(f"labels_arr.shape :{labels_arr.shape}, edges_mats.shape:{edges_mats.shape}, feature_mats.shape:{feature_mats.shape}")
    logger.info(f"edges_mats[0, 0, 0, :10]:{edges_mats[0, 0, 0, :10]}")
    logger.info(f"feature_mats[0, 0, 0, :10]:{feature_mats[0, 0, 0, :10]}")
    logger.info("-"*100)
    current_dir = Path(__file__).parent
    save_dir = current_dir / "ywt_cp_change"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"ywt_cp_change_point_{data_sp_mode}", labels_arr)
    np.save(save_dir / f"ywt_cp_edges_{data_sp_mode}", edges_mats)
    np.save(save_dir / f"ywt_cp_feature_{data_sp_mode}", feature_mats)


def gen_mts_corr_ad_data():
    cord_ywt_train_fea_ar = np.load("./cord_cpd_ywt_data/ori_cord_cpd_data/cp_feature_train.npy")
    cord_ywt_train_edges_ar = np.load("./cord_cpd_ywt_data/ori_cord_cpd_data/cp_edges_train.npy")
    cord_ywt_train_cp_ar = np.load("./cord_cpd_ywt_data/ori_cord_cpd_data/cp_change_point_train.npy")
    num_batchs = cord_ywt_train_fea_ar.shape[0]
    time_len = cord_ywt_train_fea_ar.shape[2]
    num_nodes = cord_ywt_train_fea_ar.shape[1]
    cord_ywt_train_edges_ar = cord_ywt_train_edges_ar.reshape(num_batchs*time_len, num_nodes, num_nodes)
    np.save("./cord_cpd_ywt_data/artif_particle_adj_mat", cord_ywt_train_edges_ar)
    logger.info(f"Converted artifical particle edges array shape: {cord_ywt_train_edges_ar.shape}")
    logger.info(f"Time point of change point of no.1001 data: {cord_ywt_train_cp_ar[1001]}")
    logger.info(f"Edges array during time point [48 ~ 52] of no. 1001 Converted artifical particle data: {cord_ywt_train_edges_ar[100148:100152]}")


if __name__ == "__main__":
    convert_args_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    convert_args_parser.add_argument("--convert_ywt_data", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                     help="input --convert_ywt_data to convert ywt graph data to the format of input of CORD_CPD")
    convert_args_parser.add_argument("--do_filt", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                     help="input --do_filt to convert ywt graph data with filtering nan to 0, non-nan to 1")
    convert_args_parser.add_argument("--graph_nodes_v_mode", type=str, nargs='?', default=None,
                                     help="Decide mode of nodes' vaules of graph_nodes_matrix, look up the options by execute python ywt_library/data_module.py -h")
    convert_args_parser.add_argument("--t_idx_mode", type=str, nargs='?', default="non_overlap",
                                     help=f"Decide whether top different time indexes overlap within time length:"
                                          f"    - overlap"
                                          f"    - non_overlap")
    convert_args_parser.add_argument("--convert_artif_data", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                     help="input --convert_artif_data to convert artificial particle data to the format of input MTSCorrAD")
    args = convert_args_parser.parse_args()
    dataset_dir = "../../correlation-change-predict/pipeline_dataset/sp500_20082017_corr_ser_reg_corr_mat_hrchy_11_cluster_label_last-train_train"
    if args.convert_ywt_data:
        if args.do_filt:
            gra_edges_data_mats = np.load(f"{dataset_dir}/filtered_graph_adj_mat/keep_positive-quan05/corr_s1_w10_adj_mat.npy")
            # replace NaN with 0 and non-NaN with 1
            gra_edges_data_mats[np.logical_not(np.isnan(gra_edges_data_mats))] = 1
            gra_edges_data_mats[np.isnan(gra_edges_data_mats)] = 0
        else:
            gra_edges_data_mats = np.load(f"{dataset_dir}/graph_adj_mat/corr_s1_w10_adj_mat.npy")

        gra_nodes_data_mats = np.load(f"{dataset_dir}/graph_node_mat/{nodes_mode}_s1_w10_nodes_mat.npy") if (nodes_mode := args.graph_nodes_v_mode) is not None else np.ones((gra_edges_data_mats.shape[0], 1, gra_edges_data_mats.shape[2]))

        assert not np.isnan(gra_edges_data_mats).any(), "input adjacency matrices contains null values"
        train_adj_mats, train_nodes_mats = gra_edges_data_mats[:int((len(gra_edges_data_mats)-1)*0.9)], gra_nodes_data_mats[:int((len(gra_edges_data_mats)-1)*0.9)]
        valid_adj_mats, valid_nodes_mats = gra_edges_data_mats[int((len(gra_edges_data_mats)-1)*0.9):int((len(gra_edges_data_mats)-1)*0.95)], gra_nodes_data_mats[int((len(gra_edges_data_mats)-1)*0.9):int((len(gra_edges_data_mats)-1)*0.95)]
        test_adj_mats, test_nodes_mats = gra_edges_data_mats[int((len(gra_edges_data_mats)-1)*0.95):], gra_nodes_data_mats[int((len(gra_edges_data_mats)-1)*0.95):]
        if args.t_idx_mode == "non_overlap":
            TIME_LEN = 10
            OVERLAP_WIDTH = TIME_LEN * 2
            (_, tr_quan), (_, tr_idx_list) = find_non_overlap_idx(train_adj_mats, half_overlap_width=int(OVERLAP_WIDTH/2)).items()
            (_, val_quan), (_, val_idx_list) = find_non_overlap_idx(valid_adj_mats, half_overlap_width=int(OVERLAP_WIDTH/2)).items()
            (_, tt_quan), (_, tt_idx_list) = find_non_overlap_idx(test_adj_mats, half_overlap_width=int(OVERLAP_WIDTH/2)).items()
        elif args.t_idx_mode == "overlap":
            TIME_LEN = 100
            PERCENT = 0.2
            tr_idx_list = find_top_diff_idx(train_adj_mats, quan=PERCENT)
            val_idx_list = find_top_diff_idx(valid_adj_mats, quan=PERCENT)
            tt_idx_list = find_top_diff_idx(test_adj_mats, quan=PERCENT)

        gen_cord_cpd_data(train_adj_mats, train_nodes_mats, tr_idx_list, idx_mode=args.t_idx_mode, retrieve_t_width=TIME_LEN, data_sp_mode="train")
        gen_cord_cpd_data(valid_adj_mats, valid_nodes_mats, val_idx_list, idx_mode=args.t_idx_mode, retrieve_t_width=TIME_LEN, data_sp_mode="valid")
        gen_cord_cpd_data(test_adj_mats, test_nodes_mats, tt_idx_list, idx_mode=args.t_idx_mode, retrieve_t_width=TIME_LEN, data_sp_mode="test")

    if args.convert_artif_data:
        gen_mts_corr_ad_data()
