import numpy as np
from itertools import combinations
import logging
from pathlib import Path
import argparse


logger = logging.getLogger(__name__)
logger_console = logging.StreamHandler()
logger_formatter = logging.Formatter('%(levelname)-8s [%(filename)s] %(message)s')
logger_console.setFormatter(logger_formatter)
logger.addHandler(logger_console)
logger.setLevel(logging.INFO)


def find_non_overlap_idx(mats, half_overlap_width=50):
    mats_diff_list = np.abs((mats[1:] - mats[0:-1]).sum(axis=1).sum(axis=1))
    non_overlap_indices_quan = {"quan": None, "non_overlap_indcies": None}
    max_num_non_overlap_indices = 0
    for quan in np.linspace(0.5, 1, 51):
        quan = np.round(quan, 2)
        over_threshold_idx_list = np.where(mats_diff_list > np.quantile(mats_diff_list, quan))[0]
        idx_pairs = list(combinations(over_threshold_idx_list, 2))
        overlap_indices = set()
        # check if there exists any pair that satisfies the condition
        for idx, idx2 in idx_pairs:
            is_within_range = abs(idx - idx2) <= half_overlap_width
            if is_within_range:
                overlap_indices.update([idx, idx2])

        non_overlap_indices = set(over_threshold_idx_list) - overlap_indices
        if len(non_overlap_indices) > max_num_non_overlap_indices:
            max_num_non_overlap_indices = len(non_overlap_indices)
            non_overlap_indices_quan["quan"] = quan
            non_overlap_indices_quan["non_overlap_indcies"] = non_overlap_indices

        if quan > 0.94:
            logger.debug(quan)
            logger.debug(over_threshold_idx_list)
            logger.debug("-"*50)
        if len(non_overlap_indices) > 5 :
            logger.debug(f"At quantile threshold: {quan}, number of over threshold index: {len(over_threshold_idx_list)}, number of overlap indices: {len(overlap_indices)}, number of non-overlap indices: {len(non_overlap_indices)}")
            logger.debug(f"non overlap indices: {non_overlap_indices}")

    non_overlap_indices_quan["non_overlap_indcies"] = np.array(sorted(non_overlap_indices_quan["non_overlap_indcies"])) + 1

    return non_overlap_indices_quan


def gen_cord_cpd_data(mats, idx_list, data_mode, retrieve_t_width=10):
    retrieve_t_width = 10
    edges_mats_list = []
    labels_list = []
    for idx in idx_list:
        start_pos = np.random.randint(2, retrieve_t_width, size=1)[0]
        begin_idx = max(0, (idx - start_pos))
        end_idx = begin_idx + retrieve_t_width
        label_idx = idx - begin_idx
        logger.debug(f"original idx: {idx}, retrieve indices range: {begin_idx}~{end_idx}, label: {label_idx}")
        edges_mats_list.append(mats[begin_idx:end_idx])
        labels_list.append(label_idx)
    labels_arr = np.array(labels_list)
    edges_mats = np.stack(edges_mats_list)
    feature_mats = np.ones(edges_mats.shape[:3])
    feature_mats = feature_mats[:, :, np.newaxis, :]
    logger.info(f"labels_arr.shape :{labels_arr.shape}, edges_mats.shape:{edges_mats.shape}, feature_mats.shape:{feature_mats.shape}")
    current_dir = Path(__file__).parent
    save_dir = current_dir / "ywt_cp_change"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f"ywt_cp_change_point_{data_mode}", labels_arr)
    np.save(save_dir / f"ywt_cp_edges_{data_mode}", edges_mats)
    np.save(save_dir / f"ywt_cp_feature_{data_mode}", feature_mats)


if __name__ == "__main__":
    convert_args_parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    convert_args_parser.add_argument("--convert_ywt_data", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                     help="input --convert_ywt_data to convert ywt graph data to the format of input of CORD_CPD")
    convert_args_parser.add_argument("--convert_artif_data", type=bool, default=False, action=argparse.BooleanOptionalAction,  # setting of output files
                                     help="input --convert_artif_data to convert artificial particle data to the format of input MTSCorrAD")
    args = convert_args_parser.parse_args()
    if args.convert_ywt_data:
        arr = np.load("../../correlation-change-predict/pipeline_dataset/sp500_20082017_corr_ser_reg_corr_mat_hrchy_11_cluster-train_train-graph_data/corr_s1_w10_graph.npy")
        train_mats = arr[:int((len(arr)-1)*0.9)]
        valid_mats = arr[int((len(arr)-1)*0.9):int((len(arr)-1)*0.95)]
        test_mats = arr[int((len(arr)-1)*0.95):]
        overlap_width = 20
        time_len = 10
        (_, tr_quan), (_, tr_idx_list) = find_non_overlap_idx(train_mats, half_overlap_width=int(overlap_width/2)).items()
        (_, val_quan), (_, val_idx_list) = find_non_overlap_idx(valid_mats, half_overlap_width=int(overlap_width/2)).items()
        (_, tt_quan), (_, tt_idx_list) = find_non_overlap_idx(test_mats, half_overlap_width=int(overlap_width/2)).items()
        gen_cord_cpd_data(train_mats, tr_idx_list, retrieve_t_width=time_len, data_mode="train")
        gen_cord_cpd_data(valid_mats, val_idx_list, retrieve_t_width=time_len, data_mode="valid")
        gen_cord_cpd_data(test_mats, tt_idx_list, retrieve_t_width=time_len, data_mode="test")
    if args.convert_artif_data:
        cord_ywt_train_fea_ar = np.load("./cord_cpd_ywt_data/ori_cord_cpd_data/cp_feature_train.npy")
        cord_ywt_train_edges_ar = np.load("./cord_cpd_ywt_data/ori_cord_cpd_data/cp_edges_train.npy")
        cord_ywt_train_cp_ar = np.load("./cord_cpd_ywt_data/ori_cord_cpd_data/cp_change_point_train.npy")
        num_batchs = cord_ywt_train_fea_ar.shape[0]
        time_len = cord_ywt_train_fea_ar.shape[2]
        num_nodes = cord_ywt_train_fea_ar.shape[1]
        num_features = cord_ywt_train_fea_ar.shape[3]
        cord_ywt_train_edges_ar = cord_ywt_train_edges_ar.reshape(num_batchs*time_len, num_nodes, num_nodes)
        np.save("./cord_cpd_ywt_data/artif_particle_adj_mat", cord_ywt_train_edges_ar)
        logger.info(f"Converted artifical particle edges array shape: {cord_ywt_train_edges_ar.shape}")
        logger.info(f"Time point of change point of no.1001 data: {cord_ywt_train_cp_ar[1001]}")
        logger.info(f"Edges array during time point [48 ~ 52] of no. 1001 Converted artifical particle data: {cord_ywt_train_edges_ar[100148:100152]}")
