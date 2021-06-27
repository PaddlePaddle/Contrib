script_path=$(cd `dirname $0`; pwd)
cd $script_path

CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
                        -c ./configs/DLA/DLA60_best.yaml \
                        -o pretrained_model='output/DLA60/best_model/ppcls' \
                        -o load_static_weights=False