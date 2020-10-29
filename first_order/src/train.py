import os
import sys
import yaml
import random
import logging
from argparse import ArgumentParser

import paddle
from paddle import fluid
import paddle.fluid.dygraph as dygraph
from tqdm import trange
import numpy as np

from frames_dataset import FramesDataset
from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
# from reconstruction import reconstruction
# from animate import animate

if paddle.version.full_version == '1.8.4':
    from paddle.fluid.dygraph.learning_rate_scheduler import MultiStepDecay
elif paddle.version.major == '2':
    from paddle.fluid.dygraph import MultiStepDecay
from modules.model import GeneratorFullModel, DiscriminatorFullModel

TEST_MODE = False
if TEST_MODE:
    logging.warning('TEST MODE: train.py')
    # fake_input可随意指定,此处的batchsize=2
    fake_input = np.transpose(np.tile(np.load('/home/aistudio/img.npy')[:1, ...], (2, 1, 1, 1)).astype(np.float32)/255, (0, 3, 1, 2))  #Shape:[2, 3, 256, 256]


def train(config, generator, discriminator, kp_detector, save_dir, dataset):
    train_params = config['train_params']
    
    # learning_rate_scheduler
    if paddle.version.full_version in ['1.8.4'] or paddle.version.major == '2':
        gen_lr = MultiStepDecay(learning_rate=train_params['lr_generator'],
                                milestones=train_params['epoch_milestones'], decay_rate=0.1)
        dis_lr = MultiStepDecay(learning_rate=train_params['lr_discriminator'],
                                milestones=train_params['epoch_milestones'], decay_rate=0.1)
        kp_lr = MultiStepDecay(learning_rate=train_params['lr_kp_detector'],
                               milestones=train_params['epoch_milestones'], decay_rate=0.1)
    else:
        gen_lr = train_params['lr_generator']
        dis_lr = train_params['lr_discriminator']
        kp_lr = train_params['lr_kp_detector']
    
    # optimer
    if TEST_MODE:
        logging.warning('TEST MODE: Optimer is SGD, lr is 0.001. train.py: L50')
        optimizer_generator = fluid.optimizer.SGDOptimizer(
            parameter_list=generator.parameters(),
            learning_rate=0.001
        )
        optimizer_discriminator = fluid.optimizer.SGDOptimizer(
            parameter_list=discriminator.parameters(),
            learning_rate=0.001
        )
        optimizer_kp_detector = fluid.optimizer.SGDOptimizer(
            parameter_list=kp_detector.parameters(),
            learning_rate=0.001
        )
    else:
        optimizer_generator = fluid.optimizer.AdamOptimizer(
            parameter_list=generator.parameters(),
            learning_rate=gen_lr
        )
        optimizer_discriminator = fluid.optimizer.AdamOptimizer(
            parameter_list=discriminator.parameters(),
            learning_rate=dis_lr
        )
        optimizer_kp_detector = fluid.optimizer.AdamOptimizer(
            parameter_list=kp_detector.parameters(),
            learning_rate=kp_lr
        )
    
    # load start_epoch
    if isinstance(config['ckpt_model']['start_epoch'], int):
        start_epoch = config['ckpt_model']['start_epoch']
    else:
        start_epoch = 0
    logging.info('Start Epoch is :%i' % start_epoch)
    
    # dataset pipeline
    def indexGenertaor():
        """随机生成索引序列
        """
        order = list(range(len(dataset)))
        order = order * train_params['num_repeats']
        random.shuffle(order)
        for i in order:
            yield i
    
    _dataset = fluid.io.xmap_readers(dataset.getSample, indexGenertaor, process_num=4, buffer_size=128, order=False)
    _dataset = fluid.io.batch(_dataset, batch_size=train_params['batch_size'], drop_last=True)
    dataloader = fluid.io.buffered(_dataset, 1)
    
    ###### Restore Part ######
    ckpt_config = config['ckpt_model']
    has_key = lambda key: key in ckpt_config.keys() and ckpt_config[key] is not None
    if has_key('generator'):
        if ckpt_config['generator'][-3:] == 'npz':
            G_param = np.load(ckpt_config['generator'], allow_pickle=True)['arr_0'].item()
            G_param_clean = [(i, G_param[i]) for i in G_param if 'num_batches_tracked' not in i]
            parameter_clean = generator.parameters()
            del (parameter_clean[65])  # The parameters in AntiAliasInterpolation2d is not in dict_set and should be ignore.
            for v, b in zip(parameter_clean, G_param_clean):
                v.set_value(b[1])
            logging.info('Generator is loaded from *.npz')
        else:
            param, optim = fluid.load_dygraph(ckpt_config['generator'])
            generator.set_dict(param)
            if optim is not None:
                optimizer_generator.set_dict(optim)
            else:
                logging.info('Optimizer of G is not loaded')
            logging.info('Generator is loaded from *.pdparams')
    if has_key('kp'):
        if ckpt_config['kp'][-3:] == 'npz':
            KD_param = np.load(ckpt_config['kp'], allow_pickle=True)['arr_0'].item()
            KD_param_clean = [(i, KD_param[i]) for i in KD_param if 'num_batches_tracked' not in i]
            parameter_cleans = kp_detector.parameters()
            for v, b in zip(parameter_cleans, KD_param_clean):
                v.set_value(b[1])
            logging.info('KP is loaded from *.npz')
        else:
            param, optim = fluid.load_dygraph(ckpt_config['kp'])
            kp_detector.set_dict(param)
            if optim is not None:
                optimizer_kp_detector.set_dict(optim)
            else:
                logging.info('Optimizer of KP is not loaded')
            logging.info('KP is loaded from *.pdparams')
    if has_key('discriminator'):
        if ckpt_config['discriminator'][-3:] == 'npz':
            D_param = np.load(ckpt_config['discriminator'], allow_pickle=True)['arr_0'].item()
            if 'NULL Place' in ckpt_config['discriminator']:
                # 针对未开启spectral_norm的Fashion数据集模型
                ## fashion数据集的默认设置中未启用spectral_norm，但其官方ckpt文件中存在spectral_norm特有的参数 需要重排顺序
                ## 已提相关issue，作者回应加了sn也没什么影响 https://github.com/AliaksandrSiarohin/first-order-model/issues/264
                ## 若在配置文件中开启sn则可通过else语句中的常规方法读取，故现已在配置中开启sn。
                D_param_clean = [(i, D_param[i]) for i in D_param if
                                 'num_batches_tracked' not in i and 'weight_v' not in i and 'weight_u' not in i]
                for idx in range(len(D_param_clean) // 2):
                    if 'conv.bias' in D_param_clean[idx * 2][0]:
                        D_param_clean[idx * 2], D_param_clean[idx * 2 + 1] = D_param_clean[idx * 2 + 1], D_param_clean[
                            idx * 2]
                parameter_clean = discriminator.parameters()
                for v, b in zip(parameter_clean, D_param_clean):
                    v.set_value(b[1])
            else:
                D_param_clean = list(D_param.items())
                parameter_clean = discriminator.parameters()
                assert len(D_param_clean) == len(parameter_clean)
                # 调换顺序
                ## PP中:        [conv.weight,   conv.bias,          weight_u, weight_v]
                ## pytorch中:   [conv.bias,     conv.weight_orig,   weight_u, weight_v]
                for idx in range(len(parameter_clean)):
                    if list(parameter_clean[idx].shape) == list(D_param_clean[idx][1].shape):
                        parameter_clean[idx].set_value(D_param_clean[idx][1])
                    elif parameter_clean[idx].name.split('.')[-1] == 'w_0' and D_param_clean[idx + 1][0].split('.')[
                        -1] == 'weight_orig':
                        parameter_clean[idx].set_value(D_param_clean[idx + 1][1])
                    elif parameter_clean[idx].name.split('.')[-1] == 'b_0' and D_param_clean[idx - 1][0].split('.')[
                        -1] == 'bias':
                        parameter_clean[idx].set_value(D_param_clean[idx - 1][1])
                    else:
                        print('Error', idx)
            logging.info('Discriminator is loaded from *.npz')
        else:
            param, optim = fluid.load_dygraph(ckpt_config['discriminator'])
            discriminator.set_dict(param)
            if optim is not None:
                optimizer_discriminator.set_dict(optim)
            else:
                logging.info('Optimizer of Discriminator is not loaded')
            logging.info('Discriminator is loaded from *.pdparams')
    ###### Restore Part END ######
    
    # create model
    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    if has_key('vgg19_model'):
        vggVarList = [i for i in generator_full.vgg.parameters()][2:]
        paramset = np.load(ckpt_config['vgg19_model'], allow_pickle=True)['arr_0']
        for var, v in zip(vggVarList, paramset):
            if list(var.shape) == list(v.shape):
                var.set_value(v)
            else:
                logging.warning('VGG19 cannot be loaded')
        logging.info('Pre-trained VGG19 is loaded from *.npz')
    generator_full.train()
    discriminator_full.train()
    for epoch in trange(start_epoch, train_params['num_epochs']):
        for _step, _x in enumerate(dataloader()):
            # prepear data
            x = dict()
            for _key in _x[0].keys():
                if str(_key) != 'name':
                    x[_key] = dygraph.to_variable(np.stack([_v[_key] for _v in _x], axis=0).astype(np.float32))
                else:
                    x[_key] = np.stack([_v[_key] for _v in _x], axis=0)
            # import pdb;pdb.set_trace();
            if TEST_MODE:
                logging.warning('TEST MODE: Input is Fixed train.py: L207')
                x['driving'] = dygraph.to_variable(fake_input)
                x['source'] = dygraph.to_variable(fake_input)
                x['name'] = ['test1', 'test2']
            # train generator
            losses_generator, generated = generator_full(x.copy())
            loss_values = [fluid.layers.reduce_sum(val) for val in losses_generator.values()]
            loss = fluid.layers.sum(loss_values)
            if TEST_MODE:
                print('Check Generator Loss')
                print('\n'.join(['%s:%1.5f'%(k,v.numpy()) for k,v in zip(losses_generator.keys(), loss_values)]))
                import pdb;pdb.set_trace();
            loss.backward()
            optimizer_generator.minimize(loss)
            optimizer_generator.clear_gradients()
            optimizer_kp_detector.minimize(loss)
            optimizer_kp_detector.clear_gradients()
            
            # train discriminator
            if train_params['loss_weights']['generator_gan'] != 0:
                optimizer_discriminator.clear_gradients()
                losses_discriminator = discriminator_full(x.copy(), generated)
                loss_values = [fluid.layers.reduce_mean(val) for val in losses_discriminator.values()]
                loss = fluid.layers.sum(loss_values)
                if TEST_MODE:
                    print('Check Discriminator Loss')
                    print('\n'.join(['%s:%1.5f'%(k,v.numpy()) for k,v in zip(losses_discriminator.keys(), loss_values)]))
                    import pdb;pdb.set_trace();
                loss.backward()
                optimizer_discriminator.minimize(loss)
                optimizer_discriminator.clear_gradients()
            else:
                losses_discriminator = {}
            
            losses_generator.update(losses_discriminator)
            losses = {key: fluid.layers.reduce_mean(value).detach().numpy() for key, value in losses_generator.items()}
            
            # print log
            if _step % 20 == 0:
                logging.info('Epoch:%i\tstep: %i\tLr:%1.7f' % (epoch, _step, optimizer_generator.current_step_lr()))
                logging.info('\t'.join(['%s:%1.4f' % (k, v) for k, v in losses.items()]))
        
        # save
        if epoch % 3 == 0:
            paddle.fluid.save_dygraph(generator.state_dict(), os.path.join(save_dir, 'epoch%i/G' % epoch))
            paddle.fluid.save_dygraph(discriminator.state_dict(), os.path.join(save_dir, 'epoch%i/D' % epoch))
            paddle.fluid.save_dygraph(kp_detector.state_dict(), os.path.join(save_dir, 'epoch%i/KP' % epoch))
            paddle.fluid.save_dygraph(optimizer_generator.state_dict(), os.path.join(save_dir, 'epoch%i/G' % epoch))
            paddle.fluid.save_dygraph(optimizer_discriminator.state_dict(), os.path.join(save_dir, 'epoch%i/D' % epoch))
            paddle.fluid.save_dygraph(optimizer_kp_detector.state_dict(), os.path.join(save_dir, 'epoch%i/KP' % epoch))
            logging.info('Model is saved to:%s' % os.path.join(save_dir, 'epoch%i/' % epoch))
        if paddle.version.full_version in ['1.8.4'] or paddle.version.major == '2':
            gen_lr.epoch()
            dis_lr.epoch()
            kp_lr.epoch()

if __name__ == "__main__":
    plac = fluid.CUDAPlace(0)
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train"])
    # parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--save_dir", default='/home/aistudio/train_ckpt', help="path to save in")
    # parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
    #                     help="Names of the devices comma separated.")
    parser.add_argument("--preload", action='store_true', help="preload dataset to RAM")
    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    with dygraph.guard(plac):
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                                **config['model_params']['common_params'])
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                **config['model_params']['common_params'])

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])
    if opt.preload:
        logging.info('PreLoad Dataset: Start')
        pre_list = list(range(len(dataset)))
        import multiprocessing.pool as pool
        with pool.Pool(4) as pl:
            buf = pl.map(dataset.preload, pre_list)
        for idx, (i,v) in enumerate(zip(pre_list, buf)):
            dataset.buffed[i] = v.copy()
            buf[idx] = None
        logging.info('PreLoad Dataset: End')

    if opt.mode == 'train':
        logging.info("Start training...")
        with dygraph.guard(plac):
            save_dir = opt.save_dir
            train(config, generator, discriminator, kp_detector, save_dir, dataset)
    # elif opt.mode == 'reconstruction':
    #     print("Reconstruction...")
    #     reconstruction(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    # elif opt.mode == 'animate':
    #     print("Animate...")
    #     animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)