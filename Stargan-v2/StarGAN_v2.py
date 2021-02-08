from utils import *
import time
from glob import glob
from tqdm import tqdm
from tqdm.contrib import tenumerate
from networks import *
import PIL.Image
from visualdl import LogWriter
from paddle.fluid.contrib.model_stat import summary
import shutil


class StarGAN_v2():
    def __init__(self, args, place):
        super(StarGAN_v2, self).__init__()
        self.place = place
        self.model_name = 'StarGAN_v2'
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.ds_iter = args.ds_iter
        self.iteration = args.iteration

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.lr = args.lr
        self.f_lr = args.f_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.ema_decay = args.ema_decay

        """ Weight """
        self.adv_weight = args.adv_weight
        self.sty_weight = args.sty_weight
        self.ds_weight = args.ds_weight
        self.cyc_weight = args.cyc_weight

        self.r1_weight = args.r1_weight

        """ Generator """
        self.latent_dim = args.latent_dim
        self.style_dim = args.style_dim
        self.num_style = args.num_style

        """ Mapping Network """
        self.hidden_dim = args.hidden_dim

        """ Discriminator """
        self.sn = args.sn

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.checkpoint_dir = os.path.join(args.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)

        self.log_dir = os.path.join(args.log_dir, self.model_dir)
        check_folder(self.log_dir)

        self.result_dir = os.path.join(args.result_dir, self.model_dir)
        check_folder(self.result_dir)

        dataset_path = './dataset'

        self.dataset_path = os.path.join(dataset_path, self.dataset_name, 'train')
        self.test_dataset_path = os.path.join(dataset_path, self.dataset_name, 'test')
        self.val_dataset_path = os.path.join(dataset_path, self.dataset_name, 'val')
        self.domain_list = sorted([os.path.basename(x) for x in glob(self.dataset_path + '/*')])
        self.num_domains = len(self.domain_list)

        self.w_hpf = 0
        if self.num_domains == 2:
            self.w_hpf = 1

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# domain_list : ", self.domain_list)

        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print("# ds iteration : ", self.ds_iter)

        print()

        print("##### Generator #####")
        print("# latent_dim : ", self.latent_dim)
        print("# style_dim : ", self.style_dim)
        print("# num_style : ", self.num_style)

        print()

        print("##### Mapping Network #####")
        print("# hidden_dim : ", self.hidden_dim)

        print()

        print("##### Discriminator #####")
        print("# spectral normalization : ", self.sn)

    def g_train_step(self, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):

        if z_trgs is not None:
            z_trg, z_trg2 = z_trgs
        if x_refs is not None:
            x_ref, x_ref2 = x_refs

        # adversarial loss
        if z_trgs is not None:
            s_trg = self.mapping_network([z_trg, y_trg])
        else:
            s_trg = self.style_encoder([x_ref, y_trg])
        x_fake = self.generator([x_real, s_trg], masks)
        fake_logit = self.discriminator([x_fake, y_trg])
        g_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_logit)

        # style reconstruction loss
        s_pred = self.style_encoder([x_fake, y_trg])
        g_sty_loss = self.sty_weight * L1_loss(s_pred, s_trg)

        # diversity sensitive loss
        if z_trgs is not None:
            s_trg2 = self.mapping_network([z_trg2, y_trg])
        else:
            s_trg2 = self.style_encoder([x_ref2, y_trg])

        x_fake2 = self.generator([x_real, s_trg2], masks)
        x_fake2.stop_gradient = True

        g_ds_loss = -self.ds_weight * L1_loss(x_fake, x_fake2)

        # cycle-consistency loss
        masks = self.fan.get_heatmap(x_fake) if self.w_hpf > 0 else None
        s_org = self.style_encoder([x_real, y_org])
        x_rec = self.generator([x_fake, s_org], masks)
        g_cyc_loss = self.cyc_weight * L1_loss(x_rec, x_real)

        g_loss = g_adv_loss + g_sty_loss + g_ds_loss + g_cyc_loss
        g_adv_loss.persistable = True
        g_sty_loss.persistable = True
        g_ds_loss.persistable = True
        g_cyc_loss.persistable = True

        return g_adv_loss, g_sty_loss, g_ds_loss, g_cyc_loss, g_loss

    def d_train_step(self, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
        # no_grad()下的内容不计算梯度,只训练D，可以减少计算以显存占用
        with fluid.dygraph.no_grad():
            if z_trg is not None:
                s_trg = self.mapping_network([z_trg, y_trg])

            else:  # x_ref is not None
                s_trg = self.style_encoder([x_ref, y_trg])

            x_fake = self.generator([x_real, s_trg], masks)

        real_logit = self.discriminator([x_real, y_org])
        fake_logit = self.discriminator([x_fake, y_trg])

        real_loss, fake_loss = self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit)
        d_adv_loss = real_loss + fake_loss

        if self.gan_type == 'gan-gp':
            d_adv_loss += self.r1_weight * r1_gp_req(self.discriminator, x_real, y_org)
        d_loss = d_adv_loss

        return real_loss, fake_loss, d_adv_loss, d_loss

    def train(self):
        with fluid.dygraph.guard(self.place):
            if self.phase == 'train':
                """ Input Image"""
                img_class = Image_data(self.img_size, self.img_ch, self.dataset_path, self.domain_list,
                                       self.augment_flag, self.batch_size)
                img_and_domain = img_class.preprocess()

                dataset_num = len(img_class.images)
                print("Dataset number : ", dataset_num)

                batch_img_and_domain = img_class.image_processing(img_class.records, img_class.images,
                                                                  img_class.shuffle_images, img_class.domains)


                self.img_and_domain_iter = batch_img_and_domain

                """ Network """
                self.generator = Generator(self.img_size, self.img_ch, self.style_dim,
                                           max_conv_dim=self.hidden_dim, sn=False, w_hpf=self.w_hpf)
                self.mapping_network = MappingNetwork(self.style_dim, self.hidden_dim,
                                                      self.num_domains, sn=False, )
                self.style_encoder = StyleEncoder(self.img_size, self.style_dim, self.num_domains,
                                                  max_conv_dim=self.hidden_dim, sn=False)

                self.discriminator = Discriminator(self.img_size, self.num_domains, max_conv_dim=self.hidden_dim,
                                                   sn=self.sn)

                self.generator_ema = Generator(self.img_size, self.img_ch, self.style_dim,
                                               max_conv_dim=self.hidden_dim, sn=False, w_hpf=self.w_hpf)
                self.mapping_network_ema = MappingNetwork(self.style_dim, self.hidden_dim,
                                                          self.num_domains, sn=False, )
                self.style_encoder_ema = StyleEncoder(self.img_size, self.style_dim, self.num_domains,
                                                      max_conv_dim=self.hidden_dim, sn=False)

                if self.w_hpf > 0:
                    self.fan = FAN(fname_pretrained='fan')
                    self.fan.eval()

                self.generator.train()
                self.mapping_network.train()
                self.style_encoder.train()
                self.discriminator.train()
                self.generator_ema.train()
                self.mapping_network_ema.train()
                self.style_encoder_ema.train()

                """ Optimizer """
                self.g_optimizer = fluid.optimizer.Adam(learning_rate=self.lr, beta1=self.beta1,
                                                        beta2=self.beta2,
                                                        epsilon=1e-08,
                                                        parameter_list=self.generator.parameters(),
                                                        regularization=fluid.regularizer.L2Decay(
                                                            regularization_coeff=1e-4))
                self.e_optimizer = fluid.optimizer.Adam(learning_rate=self.lr, beta1=self.beta1,
                                                        beta2=self.beta2,
                                                        epsilon=1e-08,
                                                        parameter_list=self.style_encoder.parameters(),
                                                        regularization=fluid.regularizer.L2Decay(
                                                            regularization_coeff=1e-4))
                self.f_optimizer = fluid.optimizer.Adam(learning_rate=self.f_lr, beta1=self.beta1,
                                                        beta2=self.beta2,
                                                        epsilon=1e-08,
                                                        parameter_list=self.mapping_network.parameters(),
                                                        regularization=fluid.regularizer.L2Decay(
                                                            regularization_coeff=1e-4))
                self.d_optimizer = fluid.optimizer.Adam(learning_rate=self.lr, beta1=self.beta1,
                                                        beta2=self.beta2,
                                                        epsilon=1e-08,
                                                        parameter_list=self.discriminator.parameters(),
                                                        regularization=fluid.regularizer.L2Decay(
                                                            regularization_coeff=1e-4))

                # 加载模型
                #self.load_model(choice='train')
                #self.load_model(choice='fine_tune')

                # """ Checkpoint """
                self.start_iteration = 0



            start_time = time.time()

            # setup VisulDL
            train_summary_writer = LogWriter(self.log_dir)
            ds_weight_init = self.ds_weight

            for idx in range(self.start_iteration, self.iteration):
                iter_start_time = time.time()

                # decay weight for diversity sensitive loss
                if self.ds_weight > 0:
                    self.ds_weight = ds_weight_init - (ds_weight_init / self.ds_iter) * idx
                x_real, _, y_org = next(self.img_and_domain_iter)
                x_ref, x_ref2, y_trg = next(self.img_and_domain_iter)

                z_trg = np.random.normal(size=[self.batch_size, self.latent_dim])
                z_trg2 = np.random.normal(size=[self.batch_size, self.latent_dim])

                x_real = np.array([item for item in x_real], dtype='float32')
                x_ref = np.array([item for item in x_ref], dtype='float32')
                x_ref2 = np.array([item for item in x_ref2], dtype='float32')
                y_org = np.array([item for item in y_org], dtype='float32')
                y_trg = np.array([item for item in y_trg], dtype='float32')

                x_real = fluid.dygraph.to_variable(np.array(x_real, dtype=np.float32))
                y_org = fluid.dygraph.to_variable(np.array(y_org, dtype=np.int32))
                x_ref = fluid.dygraph.to_variable(np.array(x_ref, dtype=np.float32))
                x_ref2 = fluid.dygraph.to_variable(np.array(x_ref2, dtype=np.float32))
                y_trg = fluid.dygraph.to_variable(np.array(y_trg, dtype=np.int32))
                z_trg = fluid.dygraph.to_variable(np.array(z_trg, dtype=np.float32))
                z_trg2 = fluid.dygraph.to_variable(np.array(z_trg2, dtype=np.float32))

                masks = self.fan.get_heatmap(x_real) if self.w_hpf > 0 else None

                # update discriminator
                real_loss_latent, fake_loss_latent, d_adv_loss_latent, d_loss_latent = \
                    self.d_train_step(x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
                self.clear_grad()
                d_loss_latent.backward()
                self.d_optimizer.minimize(d_loss_latent)


                real_loss_ref, fake_loss_ref, d_adv_loss_ref, d_loss_ref = \
                    self.d_train_step(x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
                self.clear_grad()
                d_loss_ref.backward()
                self.d_optimizer.minimize(d_loss_ref)


                # update generator
                g_adv_loss_latent, g_sty_loss_latent, g_ds_loss_latent, g_cyc_loss_latent, g_loss_latent = self.g_train_step(
                    x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
                self.clear_grad()
                g_loss_latent.backward()
                self.g_optimizer.minimize(g_loss_latent)


                self.f_optimizer.minimize(g_loss_latent)


                self.e_optimizer.minimize(g_loss_latent)


                g_adv_loss_ref, g_sty_loss_ref, g_ds_loss_ref, g_cyc_loss_ref, g_loss_ref = self.g_train_step(
                    x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
                self.clear_grad()
                g_loss_ref.backward()
                self.g_optimizer.minimize(g_loss_ref)


                print(d_adv_loss_latent.numpy(), d_loss_latent.numpy())
                print(d_adv_loss_ref.numpy(), d_loss_ref.numpy())

                print(g_adv_loss_latent.numpy(), g_sty_loss_latent.numpy(), g_ds_loss_latent.numpy(),
                      g_cyc_loss_latent.numpy(), g_loss_latent.numpy())

                print(g_adv_loss_ref.numpy(), g_sty_loss_ref.numpy(), g_ds_loss_ref.numpy(), g_cyc_loss_ref.numpy(),
                      g_loss_ref.numpy())
                # compute moving average of network parameters

                soft_update(self.generator_ema, self.generator, self.ema_decay)
                soft_update(self.style_encoder_ema, self.style_encoder, self.ema_decay)
                soft_update(self.mapping_network_ema, self.mapping_network, self.ema_decay)

                if idx == 0:
                    g_params = self.generator.parameters()

                    d_params = self.discriminator.parameters()
                    e_params = self.style_encoder.parameters()
                    m_params = self.mapping_network.parameters()

                    print("G network parameters :{} ", len(g_params))
                    print("D network parameters : {}", len(d_params))
                    print("E network parameters :{} ", len(e_params))
                    print("M network parameters : {}", len(m_params))

                    print("Total network parameters : {}", len(g_params) + len(d_params))

                # save to tensorboard

                # with train_summary_writer.as_default():
                with train_summary_writer as writer:
                    writer.add_scalar('g/latent/adv_loss', g_adv_loss_latent, step=idx)
                    writer.add_scalar('g/latent/sty_loss', g_sty_loss_latent, step=idx)
                    writer.add_scalar('g/latent/ds_loss', g_ds_loss_latent, step=idx)
                    writer.add_scalar('g/latent/cyc_loss', g_cyc_loss_latent, step=idx)
                    writer.add_scalar('g/latent/loss', g_loss_latent, step=idx)

                    writer.add_scalar('g/ref/adv_loss', g_adv_loss_ref, step=idx)
                    writer.add_scalar('g/ref/sty_loss', g_sty_loss_ref, step=idx)
                    writer.add_scalar('g/ref/ds_loss', g_ds_loss_ref, step=idx)
                    writer.add_scalar('g/ref/cyc_loss', g_cyc_loss_ref, step=idx)
                    writer.add_scalar('g/ref/loss', g_loss_ref, step=idx)

                    writer.add_scalar('g/ds_weight', self.ds_weight, step=idx)

                    writer.add_scalar('d/latent/adv_loss', d_adv_loss_latent, step=idx)
                    writer.add_scalar('d/latent/loss', d_loss_latent, step=idx)

                    writer.add_scalar('d/ref/adv_loss', d_adv_loss_ref, step=idx)
                    writer.add_scalar('d/ref/loss', d_loss_ref, step=idx)

                # save every self.print_freq
                if np.mod(idx + 1, 10) == 0:  # 1 self.print_freq
                    latent_fake_save_path = './{}/latent_{:07d}.jpg'.format(self.sample_dir, idx + 1)
                    ref_fake_save_path = './{}/ref_{:07d}.jpg'.format(self.sample_dir, idx + 1)

                    self.latent_canvas(x_real, latent_fake_save_path)
                    self.refer_canvas(x_real, x_ref, y_trg, ref_fake_save_path, img_num=5)

                print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
                    idx, self.iteration, time.time() - iter_start_time, d_loss_latent + d_loss_ref,
                    g_loss_latent + g_loss_ref))

                # # save every self.save_freq
                if idx>1 and idx % 200 == 0:  # self.save_freq
                    self.save_model()
                    print("idx:", idx)

            # save model for final step
            # self.save_model()

            print("Total train time: %4.4f" % (time.time() - start_time))

    def save_model(self):

        fluid.save_dygraph(self.generator.state_dict(), "{}/generator".format(self.checkpoint_dir))
        fluid.save_dygraph(self.g_optimizer.state_dict(), "{}/generator".format(self.checkpoint_dir))
        fluid.save_dygraph(self.discriminator.state_dict(), "{}/discriminator".format(self.checkpoint_dir))
        if len(self.d_optimizer.state_dict()) > 0:
            print(len(self.d_optimizer.state_dict()))
            fluid.save_dygraph(self.d_optimizer.state_dict(), "{}/discriminator".format(self.checkpoint_dir))
        else:
            print('self.d_optimizer.state_dict() is None')
        # fluid.save_dygraph(self.d_optimizer.state_dict(), "{}/driscriminator".format(self.checkpoint_dir)
        fluid.save_dygraph(self.mapping_network.state_dict(), "{}/mapping_network".format(self.checkpoint_dir))
        fluid.save_dygraph(self.f_optimizer.state_dict(), "{}/mapping_network".format(self.checkpoint_dir))
        fluid.save_dygraph(self.style_encoder.state_dict(), "{}/style_encoder".format(self.checkpoint_dir))
        fluid.save_dygraph(self.e_optimizer.state_dict(), "{}/style_encoder".format(self.checkpoint_dir))

        fluid.save_dygraph(self.generator_ema.state_dict(), "{}/generator_ema".format(self.checkpoint_dir))
        fluid.save_dygraph(self.mapping_network_ema.state_dict(), "{}/mapping_network_ema".format(self.checkpoint_dir))
        fluid.save_dygraph(self.style_encoder_ema.state_dict(), "{}/style_encoder_ema".format(self.checkpoint_dir))
        print('<<save model success>>')

    def load_model(self, choice=None):
        if choice == 'fine_tune':
            gpara_state_dict, _ = fluid.load_dygraph("{}/generator_torch".format(self.checkpoint_dir))
            self.generator.load_dict(gpara_state_dict)
            mpara_state_dict, _ = fluid.load_dygraph("{}/mapping_network_torch".format(self.checkpoint_dir))
            self.mapping_network.load_dict(mpara_state_dict)

            spara_state_dict, _ = fluid.load_dygraph("{}/style_encoder_torch".format(self.checkpoint_dir))
            self.style_encoder.load_dict(spara_state_dict)

            gema_para_state_dict, _ = fluid.load_dygraph("{}/generator_torch".format(self.checkpoint_dir))
            self.generator_ema.load_dict(gema_para_state_dict)
            sema_para_state_dict, _ = fluid.load_dygraph("{}/style_encoder_torch".format(self.checkpoint_dir))
            self.style_encoder_ema.load_dict(sema_para_state_dict)
            mema_para_state_dict, _ = fluid.load_dygraph("{}/mapping_network_torch".format(self.checkpoint_dir))
            self.mapping_network_ema.load_dict(mema_para_state_dict)

        if choice == 'train':
            gpara_state_dict, gopti_state_dict = fluid.load_dygraph("{}/generator".format(self.checkpoint_dir))
            self.generator.load_dict(gpara_state_dict)
            self.g_optimizer.set_dict(gopti_state_dict)
            dpara_state_dict, dopti_state_dict = fluid.load_dygraph("{}/discriminator".format(self.checkpoint_dir))
            self.discriminator.load_dict(dpara_state_dict)
            if dopti_state_dict is not None:
                self.d_optimizer.set_dict(dopti_state_dict)
                print('dopti_state_dict load success')
            print('D load success')
            spara_state_dict, sopti_state_dict = fluid.load_dygraph("{}/style_encoder".format(self.checkpoint_dir))
            self.style_encoder.load_dict(spara_state_dict)
            self.e_optimizer.set_dict(sopti_state_dict)

            mpara_state_dict, mopti_state_dict = fluid.load_dygraph("{}/mapping_network".format(self.checkpoint_dir))
            self.mapping_network.load_dict(mpara_state_dict)
            self.f_optimizer.set_dict(mopti_state_dict)

            gema_para_state_dict, _ = fluid.load_dygraph("{}/generator_ema".format(self.checkpoint_dir))
            self.generator_ema.load_dict(gema_para_state_dict)
            sema_para_state_dict, _ = fluid.load_dygraph("{}/style_encoder_ema".format(self.checkpoint_dir))
            self.style_encoder_ema.load_dict(sema_para_state_dict)
            mema_para_state_dict, _ = fluid.load_dygraph("{}/mapping_network_ema".format(self.checkpoint_dir))
            self.mapping_network_ema.load_dict(mema_para_state_dict)
        if choice == 'test':
            gema_para_state_dict, _ = fluid.load_dygraph("{}/generator_ema".format(self.checkpoint_dir))
            self.generator_ema.load_dict(gema_para_state_dict)
            sema_para_state_dict, _ = fluid.load_dygraph("{}/style_encoder_ema".format(self.checkpoint_dir))
            self.style_encoder_ema.load_dict(sema_para_state_dict)
            mema_para_state_dict, _ = fluid.load_dygraph("{}/mapping_network_ema".format(self.checkpoint_dir))
            self.mapping_network_ema.load_dict(mema_para_state_dict)

        print('<<load model success>>')

    def clear_grad(self):
        self.g_optimizer.clear_gradients()
        self.f_optimizer.clear_gradients()
        self.e_optimizer.clear_gradients()
        self.d_optimizer.clear_gradients()

        self.generator.clear_gradients()
        self.discriminator.clear_gradients()
        self.mapping_network.clear_gradients()
        self.style_encoder.clear_gradients()


    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''
        return "{}_{}_{}{}".format(self.model_name, self.dataset_name, self.gan_type, sn)

    def refer_canvas(self, x_real, x_ref, y_trg, path, img_num):
        if type(img_num) == list:
            # In test phase
            src_img_num = img_num[0]
            ref_img_num = img_num[1]
        else:
            src_img_num = min(img_num, self.batch_size)
            ref_img_num = min(img_num, self.batch_size)

        x_real = x_real[:src_img_num]
        x_ref = x_ref[:ref_img_num]
        y_trg = y_trg[:ref_img_num].numpy()
        canvas = PIL.Image.new('RGB', (self.img_size * (src_img_num + 1) + 10, self.img_size * (ref_img_num + 1) + 10),
                               'white')
        px_real = paddle.fluid.layers.transpose(x_real, [0, 2, 3, 1])
        px_ref = paddle.fluid.layers.transpose(x_ref, [0, 2, 3, 1])

        x_real_post = postprocess_images(px_real)
        x_ref_post = postprocess_images(px_ref)

        for col, src_image in enumerate(list(x_real_post)):
            canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), ((col + 1) * self.img_size + 10, 0))

        for row, dst_image in enumerate(list(x_ref_post)):
            canvas.paste(PIL.Image.fromarray(np.uint8(dst_image), 'RGB'), (0, (row + 1) * self.img_size + 10))

            dst_image = np.transpose(dst_image, [2, 0, 1])
            row_images = np.stack([dst_image] * src_img_num)
            row_images = preprocess_fit_train_image(row_images)
            row_images = fluid.dygraph.to_variable(row_images)
            row_images_y = np.stack([y_trg[row]] * src_img_num)
            row_images_y = fluid.dygraph.to_variable(row_images_y)
            masks = self.fan.get_heatmap(x_real) if self.w_hpf > 0 else None
            s_trg = self.style_encoder_ema([row_images, row_images_y])
            row_fake_images = postprocess_images(self.generator_ema([x_real, s_trg], masks=masks))
            row_fake_images = np.transpose(row_fake_images, [0, 2, 3, 1])
            for col, image in enumerate(list(row_fake_images)):
                canvas.paste(PIL.Image.fromarray(np.uint8(image), 'RGB'),
                             ((col + 1) * self.img_size + 10, (row + 1) * self.img_size + 10))

        canvas.save(path)

    def latent_canvas(self, x_real, path):
        canvas = PIL.Image.new('RGB', (self.img_size * (self.num_domains + 1) + 10, self.img_size * self.num_style),
                               'white')
        px_real = paddle.fluid.layers.transpose(x_real, [0, 2, 3, 1])
        src_image = postprocess_images(px_real)[0]
        canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), (0, 0))
        domain_fix_list = [idx for idx in range(self.num_domains)]

        z_trgs = np.random.normal(size=[self.num_style, self.latent_dim]).astype('float32')
        z_trgs = fluid.dygraph.to_variable(z_trgs)
        masks = self.fan.get_heatmap(x_real) if self.w_hpf > 0 else None
        for row in range(self.num_style):
            z_trg = paddle.fluid.layers.unsqueeze(z_trgs[row], axes=[0])

            for col, y_trg in enumerate(domain_fix_list):
                y_trg = fluid.dygraph.to_variable(np.array(y_trg))
                y_trg = paddle.fluid.layers.reshape(y_trg, shape=[1, 1])

                s_trg = self.mapping_network_ema([z_trg, y_trg])
                x_fake = self.generator_ema([x_real, s_trg], masks=masks)
                x_fake = postprocess_images(x_fake)
                x_fake = np.transpose(x_fake, [0, 2, 3, 1])
                col_image = x_fake[0]
                canvas.paste(PIL.Image.fromarray(np.uint8(col_image), 'RGB'),
                             ((col + 1) * self.img_size + 10, row * self.img_size))

        canvas.save(path)

    def latent_val(self):
        with fluid.dygraph.guard():
            """ Test """
            """ Network """
            self.generator_ema = Generator(self.img_size, self.img_ch, self.style_dim,
                                           max_conv_dim=self.hidden_dim, sn=False, w_hpf=self.w_hpf)
            self.mapping_network_ema = MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains, sn=False)
            self.style_encoder_ema = StyleEncoder(self.img_size, self.style_dim, self.num_domains,
                                                  max_conv_dim=self.hidden_dim, sn=False)
            self.fan = FAN(fname_pretrained='fan')

            self.generator_ema.eval()
            self.mapping_network_ema.eval()
            self.style_encoder_ema.eval()
            self.fan.eval()


            """ Load model """
            self.load_model(choice='test')

            domains = os.listdir(self.val_dataset_path)
            domains.sort()
            num_domains = len(domains)
            print('Number of domains: %d' % num_domains)
            for trg_idx, trg_domain in enumerate(domains):
                src_domains = [x for x in domains if x != trg_domain]

                for src_idx, src_domain in enumerate(src_domains):

                    path_src = os.path.join(self.val_dataset_path, src_domain)
                    loader_src = glob(os.path.join(path_src, '*.png')) + glob(os.path.join(path_src, '*.jpg'))
                    loader_src.sort()
                    print('path_src', path_src)
                    print('len(loader_src)', len(loader_src))

                    task = '%s2%s' % (src_domain, trg_domain)
                    path_fake = os.path.join('./result/eval', task)
                    shutil.rmtree(path_fake, ignore_errors=True)
                    os.makedirs(path_fake)

                    print('Generating images and calculating FID for %s...' % task)

                    for i, src_img_path in enumerate(tqdm(loader_src, total=len(loader_src))):
                        src_name, src_extension = os.path.splitext(src_img_path)
                        src_name = os.path.basename(src_name)

                        src_img_ = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                        src_img = paddle.fluid.layers.unsqueeze(src_img_, axes=[0])

                        x_src = fluid.dygraph.to_variable(src_img)
                        x_srcs = []
                        for _ in range(10):
                            x_srcs.append(x_src)
                        src_img = paddle.fluid.layers.stack(src_img, axis=1)
                        x_src = fluid.layers.reshape(src_img, [-1, 3, 256, 256])

                        y_trg = fluid.dygraph.to_variable(np.array([trg_idx] * 10))

                        masks = self.fan.get_heatmap(x_src) if self.w_hpf > 0 else None

                        # generate 10 outputs from the same input

                        z_trgs = np.random.normal(size=[10, self.latent_dim]).astype('float32')
                        z_trgs = fluid.dygraph.to_variable(z_trgs)


                        y_trg = paddle.fluid.layers.reshape(y_trg, shape=[-1, 1])
                        s_trg = self.mapping_network_ema([z_trgs, y_trg])

                        x_fake = self.generator_ema([x_src, s_trg], masks=masks)
                        x_fake = postprocess_images(x_fake)
                        x_fake = np.transpose(x_fake, [0, 2, 3, 1])

                        for j in range(10):
                            # save generated images to calculate FID later
                            filename = os.path.join(path_fake, '%.4i_%.2i.png' % (i * 10, j + 1))
                            ox_fake = np.reshape(x_fake[j], [256, 256, 3])
                            img = Image.fromarray(np.uint8(ox_fake))

                            img.save(filename)

    @fluid.dygraph.no_grad
    def test(self, merge=False, merge_size=0):
        with fluid.dygraph.guard():
            """ Test """
            """ Network """
            self.generator_ema = Generator(self.img_size, self.img_ch, self.style_dim,
                                           max_conv_dim=self.hidden_dim, sn=False, w_hpf=self.w_hpf)
            self.mapping_network_ema = MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains, sn=False)
            self.style_encoder_ema = StyleEncoder(self.img_size, self.style_dim, self.num_domains,
                                                  max_conv_dim=self.hidden_dim, sn=False)
            self.fan = FAN(fname_pretrained='fan')

            """ Load model """
            self.load_model(choice='test')

            source_path = os.path.join(self.test_dataset_path, 'src_imgs')
            source_images = glob(os.path.join(source_path, '*.png')) + glob(os.path.join(source_path, '*.jpg'))
            source_images = sorted(source_images)

            # reference-guided synthesis
            print('reference-guided synthesis')
            reference_path = os.path.join(self.test_dataset_path, 'ref_imgs')
            reference_images = []
            reference_domain = []

            for idx, domain in enumerate(self.domain_list):
                image_list = glob(os.path.join(reference_path, domain) + '/*.png') + glob(
                    os.path.join(reference_path, domain) + '/*.jpg')
                image_list = sorted(image_list)
                domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]

                reference_images.extend(image_list)
                reference_domain.extend(domain_list)

            if merge:
                src_img = None
                ref_img = None
                ref_img_domain = None

                if merge_size == 0:
                    # [len_src_imgs : len_ref_imgs] matching
                    for src_idx, src_img_path in tenumerate(source_images):
                        src_name, src_extension = os.path.splitext(src_img_path)
                        src_name = os.path.basename(src_name)

                        src_img_ = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]

                        src_img_ = paddle.fluid.layers.unsqueeze(src_img_, axes=[0])

                        if src_idx == 0:
                            src_img = src_img_
                        else:
                            src_img = paddle.fluid.layers.concat([src_img, src_img_], axis=0)

                    for ref_idx, (ref_img_path, ref_img_domain_) in tenumerate(zip(reference_images, reference_domain)):
                        ref_name, ref_extension = os.path.splitext(ref_img_path)
                        ref_name = os.path.basename(ref_name)

                        ref_img_ = load_images(ref_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                        ref_img_ = paddle.fluid.layers.unsqueeze(ref_img_, axes=[0])
                        ref_img_domain_ = np.mat(ref_img_domain_)
                        ref_img_domain_ = fluid.dygraph.to_variable(np.array(ref_img_domain_))
                        if ref_idx == 0:
                            ref_img = ref_img_
                            ref_img_domain = ref_img_domain_
                        else:
                            ref_img = paddle.fluid.layers.concat([ref_img, ref_img_], axis=0)
                            ref_img_domain = paddle.fluid.layers.concat([ref_img_domain, ref_img_domain_], axis=0)
                    save_path = './{}/ref_all.jpg'.format(self.result_dir)
                    self.refer_canvas(src_img, ref_img, ref_img_domain, save_path,
                                      img_num=[len(source_images), len(reference_images)])


                else:
                    # [merge_size : merge_size] matching
                    src_size = 0
                    for src_idx, src_img_path in tenumerate(source_images):
                        src_name, src_extension = os.path.splitext(src_img_path)
                        src_name = os.path.basename(src_name)

                        src_img_ = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                        src_img_ = paddle.fluid.layers.unsqueeze(src_img_, axes=[0])

                        if src_size < merge_size:
                            if src_idx % merge_size == 0:
                                src_img = src_img_
                            else:
                                src_img = paddle.fluid.layers.concat([src_img, src_img_], axis=0)
                            src_size += 1

                            if src_size == merge_size:
                                src_size = 0

                                ref_size = 0
                                for ref_idx, (ref_img_path, ref_img_domain_) in enumerate(
                                        zip(reference_images, reference_domain)):
                                    ref_name, ref_extension = os.path.splitext(ref_img_path)
                                    ref_name = os.path.basename(ref_name)

                                    ref_img_ = load_images(ref_img_path, self.img_size,
                                                           self.img_ch)  # [img_size, img_size, img_ch]
                                    ref_img_ = paddle.fluid.layers.unsqueeze(ref_img_, axes=[0])
                                    ref_img_domain_ = paddle.fluid.layers.unsqueeze(ref_img_domain_, axes=[0])

                                    if ref_size < merge_size:
                                        if ref_idx % merge_size == 0:
                                            ref_img = ref_img_
                                            ref_img_domain = ref_img_domain_
                                        else:
                                            ref_img = paddle.fluid.layers.concat([ref_img, ref_img_], axis=0)
                                            ref_img_domain = paddle.fluid.layers.concat(
                                                [ref_img_domain, ref_img_domain_],
                                                axis=0)

                                        ref_size += 1
                                        if ref_size == merge_size:
                                            ref_size = 0

                                            save_path = './{}/ref_{}_{}.jpg'.format(self.result_dir, src_idx + 1,
                                                                                    ref_idx + 1)

                                            self.refer_canvas(src_img, ref_img, ref_img_domain, save_path,
                                                              img_num=merge_size)

            else:
                # [1:1] matching
                for src_img_path in tqdm(source_images):
                    src_name, src_extension = os.path.splitext(src_img_path)
                    src_name = os.path.basename(src_name)

                    src_img = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                    src_img = paddle.fluid.layers.unsqueeze(src_img, axes=[0])

                    for ref_img_path, ref_img_domain in zip(reference_images, reference_domain):
                        ref_name, ref_extension = os.path.splitext(ref_img_path)
                        ref_name = os.path.basename(ref_name)

                        ref_img = load_images(ref_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                        ref_img = paddle.fluid.layers.unsqueeze(ref_img, axes=[0])
                        ref_img_domain = paddle.fluid.layers.unsqueeze(ref_img_domain, axes=[0])

                        save_path = './{}/ref_{}_{}{}'.format(self.result_dir, src_name, ref_name, src_extension)

                        self.refer_canvas(src_img, ref_img, ref_img_domain, save_path, img_num=1)

            # latent-guided synthesis
            print('latent-guided synthesis')
            for src_img_path in tqdm(source_images):
                src_name, src_extension = os.path.splitext(src_img_path)
                src_name = os.path.basename(src_name)

                src_img = load_images(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
                src_img = paddle.fluid.layers.unsqueeze(src_img, axes=[0])

                save_path = './{}/latent_{}{}'.format(self.result_dir, src_name, src_extension)

                self.latent_canvas(src_img, save_path)
