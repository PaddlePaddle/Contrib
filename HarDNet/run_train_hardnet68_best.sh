script_path=$(cd `dirname $0`; pwd)
cd $script_path

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch \
                            --gpus="0,1,2,3" tools/train.py -c ./configs/HarDNet/HarDNet68_best_v2.yaml