script_path=$(cd `dirname $0`; pwd)
cd $script_path

CUDA_VISIBLE_DEVICES=0 python tools/eval.py \
                        -c ./configs/HarDNet/HarDNet68_best_v2.yaml \
                        -o pretrained_model='output/HarDNet68_v2/HarDNet68/best_model/ppcls' \
                        -o load_static_weights=False