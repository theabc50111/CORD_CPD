#!/usr/bin/env bash
ywt_data=${1:-false}
GPU=${2:-0}
mode=${3:-train}
exp_name=${4:-}
graph_nodes_v_mode=${5:-None}
nodes_v_dim=${6:-1}
timestamp=$(date +%Y%m%d%H%M)

echo ${mode}
#if [ ${mode} == 'train' ]
#then
#    data_path=data/ywt_cp_change
#elif [ ${mode} == 'test' ]
#then
#    data_path=data
#fi

# Parameter for RNN TEL, GNN SEL
if [ $ywt_data == "false" ];
then
    CUDA_VISIBLE_DEVICES=${GPU} python main.py --mode $mode --data_path data/cp_change \
                                               --spatial-encoding-layer gnn --temporal-encoding-layer rnn \
                                               --exp_dir exp/artif/my_exp_${exp_name}_gnn_tra_mlp_${timestamp} \
                                               --batch-size 12 \
                                               --epochs 1000 --eval_epoch 5
else
    CUDA_VISIBLE_DEVICES=${GPU} python main.py --mode $mode --data_path data/ywt_cp_change \
                                               --spatial-encoding-layer gnn --temporal-encoding-layer rnn \
                                               --exp_dir exp/ywt_stock/graph_nodes_v_mode_${graph_nodes_v_mode}/my_exp_${exp_name}_gnn_tra_mlp_${timestamp} \
                                               --batch-size 12 \
                                               --epochs 1000 --eval_epoch 5 \
                                               --dims $nodes_v_dim  --num-atoms 66 --timesteps 10
fi

# Parameter for TRANS TEL, GNN SEL
#CUDA_VISIBLE_DEVICES=${GPU} python main.py --mode $mode --data_path $data_path \
#    --spatial-encoding-layer gnn --temporal-encoding-layer transformer \
#    --encoder-hidden 64 --decoder mlp --batch-size 32  \
#    --exp_dir exp/${exp_name}_trans_gnn


# Parameter for RNN TEL, TRANS SEL
#CUDA_VISIBLE_DEVICES=${GPU} python main.py --mode $mode --data_path $data_path \
#    --spatial-encoding-layer trans --temporal-encoding-layer rnn \
#    --encoder-hidden 64 --decoder mlp --batch-size 32  \
#    --exp_dir exp/${exp_name}_rnn_trans
