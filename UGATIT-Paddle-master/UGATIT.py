from dataset import *

import transforms
import time
import os
from networks import ResnetGenerator,Discriminator
from utils import denorm,tensor2numpy,RGB2BGR,cam,BCEWithLogitsLoss,RhoClipper
from paddle.fluid.layers import ones_like,zeros_like
import paddle.fluid as fluid
from paddle.fluid.dygraph import L1Loss,MSELoss,to_variable
import numpy as np
import cv2
import paddle

DATASET = "datasets"
A_TEST_LIST_FILE = "data/" + DATASET + "/testA.txt"
B_TEST_LIST_FILE = "data/" + DATASET + "/testB.txt"
IMAGES_ROOT = "data/" + DATASET + "/"


class UGATIT(object):
    def __init__(self, args):
        self.light = args.light

        if self.light:
            self.model_name = 'UGATIT_light'
        else:
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume


    ##################################################################################
    # Model
    ##################################################################################
    def optimizer_setting(self,parameters):
        lr = 0.0001
        optimizer = fluid.optimizer.Adam(
            learning_rate=lr,
            parameter_list=parameters,
            beta1=0.5, beta2=0.999, regularization=fluid.regularizer.L2Decay(self.weight_decay))
        return optimizer

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size + 30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        self.trainA_loader = paddle.batch(a_reader(shuffle=True,transforms=train_transform), self.batch_size)()
        self.trainB_loader = paddle.batch(b_reader(shuffle=True,transforms=train_transform), self.batch_size)()
        self.testA_loader = a_test_reader(transforms=test_transform)
        self.testB_loader = b_test_reader(transforms=test_transform)


        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size,
                                      light=self.light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size,
                                      light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        """ Define Loss """
        self.L1_loss = L1Loss()
        self.MSE_loss = MSELoss()
        self.BCE_loss = BCEWithLogitsLoss()

        """ Trainer """
        self.G_optim = self.optimizer_setting(self.genA2B.parameters() + self.genB2A.parameters())
        self.D_optim = self.optimizer_setting(self.disGA.parameters() + self.disGB.parameters() + self.disLA.parameters() + self.disLB.parameters())

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_iter = 1
        if self.resume:
            model_list = os.listdir(os.path.join(self.result_dir, self.dataset, 'model'))
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1])
                print("[*]load %d"%(iter))
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
                print("[*] Load SUCCESS")

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            real_A = next(self.trainA_loader)
            real_B = next(self.trainB_loader)
            real_A = np.array([real_A[0].reshape(3, 256, 256)]).astype("float32")
            real_B = np.array([real_B[0].reshape(3, 256, 256)]).astype("float32")
            real_A = to_variable(real_A)
            real_B = to_variable(real_B)
            # Update D
            
            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit,
                                                                                                    zeros_like(
                                                                                                        fake_GA_logit))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, ones_like(real_GA_cam_logit)) + self.MSE_loss(
                fake_GA_cam_logit, zeros_like(fake_GA_cam_logit))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit,
                                                                                                    zeros_like(
                                                                                                        fake_LA_logit))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, ones_like(real_LA_cam_logit)) + self.MSE_loss(
                fake_LA_cam_logit, zeros_like(fake_LA_cam_logit))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit,
                                                                                                    zeros_like(
                                                                                                        fake_GB_logit))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, ones_like(real_GB_cam_logit)) + self.MSE_loss(
                fake_GB_cam_logit, zeros_like(fake_GB_cam_logit))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit,
                                                                                                    zeros_like(
                                                                                                        fake_LB_logit))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, ones_like(real_LB_cam_logit)) + self.MSE_loss(
                fake_LB_cam_logit, zeros_like(fake_LB_cam_logit))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.minimize(Discriminator_loss)
            self.genB2A.clear_gradients()
            self.genA2B.clear_gradients()
            self.disGA.clear_gradients()
            self.disLA.clear_gradients()
            self.disGB.clear_gradients()
            self.disLB.clear_gradients()
            self.D_optim.clear_gradients()

            # Update G


            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, ones_like(fake_GA_logit))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, ones_like(fake_GA_cam_logit))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, ones_like(fake_LA_logit))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, ones_like(fake_LA_cam_logit))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, ones_like(fake_GB_logit))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, ones_like(fake_GB_cam_logit))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, ones_like(fake_LB_logit))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, ones_like(fake_LB_cam_logit))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit,
                                            ones_like(fake_B2A_cam_logit)) + self.BCE_loss(
                fake_A2A_cam_logit, zeros_like(fake_A2A_cam_logit))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit,
                                            ones_like(fake_A2B_cam_logit)) + self.BCE_loss(
                fake_B2B_cam_logit, zeros_like(fake_B2B_cam_logit))

            G_loss_A = self.adv_weight * (
                        G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (
                        G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.minimize(Generator_loss)
            self.genB2A.clear_gradients()
            self.genA2B.clear_gradients()
            self.disGA.clear_gradients()
            self.disLA.clear_gradients()
            self.disGB.clear_gradients()
            self.disLB.clear_gradients()
            self.G_optim.clear_gradients()

            self.Rho_clipper(self.genA2B)
            self.Rho_clipper(self.genB2A)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))

            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for _ in range(train_sample_num):
                    real_A = next(self.trainA_loader)
                    real_B = next(self.trainB_loader)
                    real_A = np.array([real_A[0].reshape(3, 256, 256)]).astype("float32")
                    real_B = np.array([real_B[0].reshape(3, 256, 256)]).astype("float32")
                    real_A = to_variable(real_A)
                    real_B = to_variable(real_B)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    real_A = next(self.testA_loader())
                    real_B = next(self.testB_loader())
                    real_A = np.array([real_A[0].reshape(3, 256, 256)]).astype("float32")
                    real_B = np.array([real_B[0].reshape(3, 256, 256)]).astype("float32")
                    real_A = to_variable(real_A)
                    real_B = to_variable(real_B)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                                RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % 1000 == 0:
                fluid.save_dygraph(self.genA2B.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/genA2B"))
                fluid.save_dygraph(self.genB2A.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/genB2A"))
                fluid.save_dygraph(self.disGA.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/disGA"))
                fluid.save_dygraph(self.disGB.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/disGB"))
                fluid.save_dygraph(self.disLA.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/disLA"))
                fluid.save_dygraph(self.disLB.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/disLB"))
                fluid.save_dygraph(self.D_optim.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/D_optim"))
                fluid.save_dygraph(self.G_optim.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/G_optim"))
                fluid.save_dygraph(self.genA2B.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/D_optim"))
                fluid.save_dygraph(self.genB2A.state_dict(),
                                    os.path.join(self.result_dir, self.dataset + "/latest/new/G_optim"))

    def save(self, result_dir, step):
        fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(result_dir, "{}/genA2B".format(step)))
        fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(result_dir, "{}/genB2A".format(step)))
        fluid.save_dygraph(self.disGA.state_dict(), os.path.join(result_dir, "{}/disGA".format(step)))
        fluid.save_dygraph(self.disGB.state_dict(), os.path.join(result_dir, "{}/disGB".format(step)))
        fluid.save_dygraph(self.disLA.state_dict(), os.path.join(result_dir, "{}/disLA".format(step)))
        fluid.save_dygraph(self.disLB.state_dict(), os.path.join(result_dir, "{}/disLB".format(step)))
        fluid.save_dygraph(self.genA2B.state_dict(), os.path.join(result_dir, "{}/D_optim".format(step)))
        fluid.save_dygraph(self.genB2A.state_dict(), os.path.join(result_dir, "{}/G_optim".format(step)))
        fluid.save_dygraph(self.D_optim.state_dict(), os.path.join(result_dir, "{}/D_optim".format(step)))
        fluid.save_dygraph(self.G_optim.state_dict(), os.path.join(result_dir, "{}/G_optim".format(step)))

    def load(self, dir, step):
        genA2B, _ = fluid.load_dygraph(os.path.join(dir, "{}/genA2B".format(step)))
        genB2A, _ = fluid.load_dygraph(os.path.join(dir, "{}/genB2A".format(step)))
        disGA, _ = fluid.load_dygraph(os.path.join(dir, "{}/disGA".format(step)))
        disGB, _ = fluid.load_dygraph(os.path.join(dir, "{}/disGB".format(step)))
        disLA, _ = fluid.load_dygraph(os.path.join(dir, "{}/disLA".format(step)))
        disLB, _ = fluid.load_dygraph(os.path.join(dir, "{}/disLB".format(step)))
        _, D_optim = fluid.load_dygraph(os.path.join(dir, "{}/D_optim".format(step)))
        _, G_optim = fluid.load_dygraph(os.path.join(dir, "{}/G_optim".format(step)))
        self.genA2B.load_dict(genA2B)
        self.genB2A.load_dict(genB2A)
        self.disGA.load_dict(disGA)
        self.disGB.load_dict(disGB)
        self.disLA.load_dict(disLA)
        self.disLB.load_dict(disLB)
        self.G_optim.set_dict(G_optim)
        self.D_optim.set_dict(D_optim)

    def test(self):
        model_list = os.listdir(os.path.join(self.result_dir, self.dataset, 'model'))
        if not len(model_list) == 0:

            model_list.sort()
            iter = int(model_list[-1])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print("[*] Load SUCCESS")
        else:
            print("[*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        for n, (real_A, _) in enumerate(self.testA_loader()):

            real_A = np.array([real_A.reshape(3, 256, 256)]).astype("float32")

            real_A = to_variable(real_A)

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate(
                (RGB2BGR(tensor2numpy(denorm(real_A[0]))), cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))), cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                    cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

        for n, (real_B, _) in enumerate(self.testB_loader()):

            real_B = np.array([real_B.reshape(3, 256, 256)]).astype("float32")

            real_B = to_variable(real_B)

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate(
                (RGB2BGR(tensor2numpy(denorm(real_B[0]))), cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))), cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                    cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)



    def test_change(self):
        model_list = os.listdir(os.path.join(self.result_dir, self.dataset, 'model'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('/')[-1])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print("[*] Load SUCCESS")
        else:
            print("[*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        for n, (real_A, fname) in enumerate(self.testA_loader()):
            real_A = np.array([real_A[0].reshape(3, 256, 256)]).astype("float32")
            real_A = to_variable(real_A)
            fake_A2B, _, _ = self.genA2B(real_A)

            A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))

            cv2.imwrite(os.path.join(self.result_dir, self.dataset,'test', 'testA2B', '%s_fake.%s' % (fname.split('.')[0],fname.split('.')[-1])),
                        A2B * 255.0)

        for n, (real_B, fname) in enumerate(self.testB_loader()):
            real_B = np.array([real_B[0].reshape(3, 256, 256)]).astype("float32")
            real_B = to_variable(real_B)
            fake_B2A, _, _ = self.genB2A(real_B)

            B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))

            cv2.imwrite(os.path.join(self.result_dir, self.dataset,'test', 'testB2A', '%s_fake.%s' % (fname.split('.')[0],fname.split('.')[-1])),
                        B2A * 255.0)

