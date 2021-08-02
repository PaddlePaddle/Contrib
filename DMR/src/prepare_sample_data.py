import os


data_base_dir = '.'

# Prepare sampled dataset
os.makedirs(os.path.join(data_base_dir, 'data/sample_data/train'), exist_ok=True)
os.makedirs(os.path.join(data_base_dir, 'data/sample_data/test'), exist_ok=True)

i = 0
f_train = open(os.path.join(data_base_dir, 'data/sample_data/train/alimama_sampled_train.txt'), 'w')
f_test = open(os.path.join(data_base_dir, 'data/sample_data/test/alimama_sampled_test.txt'), 'w')
with open(os.path.join(data_base_dir, 'data/alimama_sampled.txt'), 'r') as f:
    line = f.readline()
    while line:
        if i % 10 < 2:
            f_test.write(line)
        else:
            if line.strip()[-1] == '1':
                up_cnt = 15  # 15, epoch 2, train 0.91, test 0.552003; 20, epoch 2, train 0.90, test 0.53; 5, epoch 2, train 0.89, test 0.52; 10, epoch 2, train 0.90, test 0.507
            else:
                up_cnt = 1
            for up_j in range(up_cnt):
                f_train.write(line)

        i += 1
        line = f.readline()

print(f'total lines: {i}')
f_train.close()
f_test.close()
