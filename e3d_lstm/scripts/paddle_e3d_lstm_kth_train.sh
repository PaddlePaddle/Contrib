python runp.py --num_save_samples 10 --train_data_paths /home/aistudio/work/kth_action --is_training True --dataset_name action --valid_data_paths /home/aistudio/work/kth_action \
--save_dir checkpoints/_kth_e3d_lstm --gen_frm_dir results/_kth_e3d_lstm --model_name e3d_lstm \
--pretrained_model /home/aistudio/work/e3d-lstm-paddle/checkpoints/_kth_e3d_lstm_5e-6_26.916687_6000/6000 \
--allow_gpu_growth True --img_channel 1 --img_width 128 --input_length 10 \
--total_length 30 --filter_size 5 --num_hidden 64,64,64,64 --patch_size 8 --layer_norm True --reverse_input False --sampling_stop_iter 0 --sampling_start_value 0 \
--lr 0.000002 --batch_size 4 --max_iterations 400000 --display_interval 1 --test_interval 500 --snapshot_interval 500
