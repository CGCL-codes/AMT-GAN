#!/usr/bin/python
# -*- encoding: utf-8 -*-
import datetime
import os
import os.path as osp
import time
import warnings

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

import tools.plot as plot_fig
from assets.models import irse, ir152, facenet
from crtiterions import GANLoss, HistogramLoss, LPIPS
from . import net

warnings.filterwarnings("ignore")


class Solver():
    def __init__(self, config, target_image, data_loader=None, inference=None):
        self.device = config.DEVICE.device
        self.G = net.Generator()
        self.H = net.H_RRDB(nb=4)
        if inference is not None:
            self.G.load_state_dict(torch.load(inference))
            self.G = self.G.to(self.device).eval()
            return

        self.target_image = target_image
        self.start_time = time.time()
        self.checkpoint = config.MODEL.WEIGHTS
        self.log_path = config.LOG.LOG_PATH
        # local model zoo: self.train_model_name_list
        self.train_model_name_list = config.MODELZOO.MODELS
        self.result_path = os.path.join(self.log_path, config.LOG.VIS_PATH)
        self.snapshot_path = os.path.join(self.log_path, config.LOG.SNAPSHOT_PATH)
        self.log_step = config.LOG.LOG_STEP
        self.vis_step = config.LOG.VIS_STEP
        self.snapshot_step = config.LOG.SNAPSHOT_STEP

        # Data loader
        self.data_loader_train = data_loader
        self.img_size = config.DATA.IMG_SIZE

        self.num_epochs = config.TRAINING.NUM_EPOCHS
        self.num_epochs_decay = config.TRAINING.NUM_EPOCHS_DECAY
        self.g_lr = config.TRAINING.G_LR
        self.d_lr = config.TRAINING.D_LR
        self.h_lr = config.TRAINING.H_LR
        self.g_step = config.TRAINING.G_STEP
        self.beta1 = config.TRAINING.BETA1
        self.beta2 = config.TRAINING.BETA2

        self.lamda_gan = config.LOSS.GAN
        self.lamda_reg = config.LOSS.CYCLE
        self.lambda_adv = config.LOSS.ADVATTACK
        self.lambda_his_lip = config.LOSS.LAMBDA_HIS_LIP
        self.lambda_his_skin = config.LOSS.LAMBDA_HIS_SKIN
        self.lambda_his_eye = config.LOSS.LAMBDA_HIS_EYE
        self.lamda_make = config.LOSS.MAKEUP
        self.lambda_idt = config.LOSS.IDT
        self.d_conv_dim = config.MODEL.D_CONV_DIM
        self.d_repeat_num = config.MODEL.D_REPEAT_NUM
        self.norm = config.MODEL.NORM

        self.resize_rate = 0.9
        self.diversity_prob = 0.5
        self.diversity = 5

        self.build_model()
        super(Solver, self).__init__()

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal(m.weight.data, gain=1.0)

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def cos_simi(self, emb_1, emb_2):
        return torch.mean(torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1))

    def cal_adv_loss(self, source, target, model_name, target_models):
        input_size = target_models[model_name][0]
        fr_model = target_models[model_name][1]
        source_resize = F.interpolate(source, size=input_size, mode='bilinear')
        target_resize = F.interpolate(target, size=input_size, mode='bilinear')
        emb_source = fr_model(source_resize)
        emb_target = fr_model(target_resize).detach()
        cos_loss = 1 - self.cos_simi(emb_source, emb_target)
        return cos_loss

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False).to(self.device)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0).to(
            self.device)

        return padded if torch.rand(1) < self.diversity_prob else x

    def input_noise(self, x):
        rnd = torch.rand(1).to(self.device)
        noise = torch.randn_like(x).to(self.device)
        x_noised = x + rnd * (0.1 ** 0.5) * noise
        x_noised.to(self.device)
        return x_noised if torch.rand(1) < self.diversity_prob else x

    def build_model(self):
        self.D_A = net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm)
        self.D_B = net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm)

        self.targe_models = {}
        for model in self.train_model_name_list:
            if model == 'ir152':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                fr_model = ir152.IR_152((112, 112))
                fr_model.load_state_dict(torch.load('./assets/models/ir152.pth'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)
            if model == 'irse50':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                fr_model = irse.Backbone(50, 0.6, 'ir_se')
                fr_model.load_state_dict(torch.load('./assets/models/irse50.pth'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)
            if model == 'facenet':
                self.targe_models[model] = []
                self.targe_models[model].append((160, 160))
                fr_model = facenet.InceptionResnetV1(num_classes=8631, device=self.device)
                fr_model.load_state_dict(torch.load('./assets/models/facenet.pth'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)
            if model == 'mobile_face':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                fr_model = irse.MobileFaceNet(512)
                fr_model.load_state_dict(torch.load('./assets/models/mobile_face.pth'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)

        self.G.apply(self.weights_init_xavier)
        self.H.apply(self.weights_init_xavier)
        self.D_A.apply(self.weights_init_xavier)
        self.D_B.apply(self.weights_init_xavier)

        self.load_checkpoint()
        self.criterionGAN = GANLoss(self.device, use_lsgan=True, tensor=torch.FloatTensor)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionHis = HistogramLoss()
        self.LPIPS = LPIPS(net='alex').to(self.device).eval()

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.h_optimizer = torch.optim.Adam(self.H.parameters(), self.h_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.H, 'H')
        self.print_network(self.D_A, 'D_A')
        self.print_network(self.D_B, 'D_B')

        self.G.to(self.device)
        self.D_A.to(self.device)
        self.D_B.to(self.device)
        self.H.to(self.device)
        self.criterionHis.to(self.device)
        self.criterionGAN.to(self.device)
        self.criterionL1.to(self.device)
        self.criterionL2.to(self.device)


    def load_checkpoint(self):
        G_path = os.path.join(self.checkpoint, 'G.pth')
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path))
            print('loaded trained generator {}..!'.format(G_path))
        D_A_path = os.path.join(self.checkpoint, 'D_A.pth')
        if os.path.exists(D_A_path):
            self.D_A.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(D_A_path).items()})
            print('loaded trained discriminator A {}..!'.format(D_A_path))
        D_B_path = os.path.join(self.checkpoint, 'D_B.pth')
        if os.path.exists(D_B_path):
            self.D_B.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(D_B_path).items()})
            print('loaded trained discriminator B {}..!'.format(D_B_path))
        H_path = os.path.join(self.checkpoint, 'H.pth')
        if os.path.exists(H_path):
            self.H.load_state_dict(torch.load(H_path))
            print('loaded trained purifier H {}..!'.format(H_path))

    def generate(self, org_A, ref_B, lms_A=None, lms_B=None, mask_A=None, mask_B=None,
                 diff_A=None, diff_B=None, gamma=None, beta=None, ret=False):
        """org_A is content, ref_B is style"""
        res = self.G(org_A, ref_B, mask_A, mask_B, diff_A, diff_B, gamma, beta, ret)
        return res

    def test(self, real_A, mask_A, diff_A, real_B, mask_B, diff_B):
        cur_prama = None
        with torch.no_grad():
            cur_prama = self.generate(real_A, real_B, None, None, mask_A, mask_B,
                                      diff_A, diff_B, ret=True)
            fake_A = self.generate(real_A, real_B, None, None, mask_A, mask_B,
                                   diff_A, diff_B, gamma=cur_prama[0], beta=cur_prama[1])
        fake_A = fake_A.squeeze(0)
        # normalize
        min_, max_ = fake_A.min(), fake_A.max()
        fake_A.add_(-min_).div_(max_ - min_ + 1e-5)
        return ToPILImage()(fake_A.cpu())

    def train(self):
        # The number of iterations per epoch
        self.iters_per_epoch = len(self.data_loader_train)
        # Start with trained model if exists
        g_lr = self.g_lr
        h_lr = self.h_lr
        d_lr = self.d_lr
        start = 0

        for self.e in range(start, self.num_epochs):
            for self.i, (source_input, reference_input) in enumerate(self.data_loader_train):
                # image, mask, dist
                image_s, image_r = source_input[0].to(self.device), reference_input[0].to(self.device)
                mask_s, mask_r = source_input[1].to(self.device), reference_input[1].to(self.device)
                dist_s, dist_r = source_input[2].to(self.device), reference_input[2].to(self.device)
                ############################################ Train D ###################################################
                # Real
                out = self.D_A(image_r)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                fake_A = self.G(image_s, image_r, mask_s, mask_r, dist_s, dist_r)
                fake_A = Variable(fake_A.data).detach()
                out = self.D_A(fake_A)
                d_loss_fake = self.criterionGAN(out, False)
                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                self.d_A_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_A_optimizer.step()
                # Logging
                self.loss = {}
                self.loss['D-A-loss_real'] = d_loss_real.mean().item()
                # Real
                out = self.D_B(image_s)
                d_loss_real = self.criterionGAN(out, True)
                # Fake
                fake_B = self.G(image_r, image_s, mask_r, mask_s, dist_r, dist_s)
                fake_B = Variable(fake_B.data).detach()
                out = self.D_B(fake_B)
                d_loss_fake = self.criterionGAN(out, False)
                # Backward + Optimize
                d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
                self.d_B_optimizer.zero_grad()
                d_loss.backward(retain_graph=False)
                self.d_B_optimizer.step()
                # Logging
                self.loss['D-B-loss_real'] = d_loss_real.mean().item()


                ############################################ Train G ###################################################
                if (self.i + 1) % self.g_step == 0:
                    # loss_gan
                    fake_A = self.G(image_s, image_r, mask_s, mask_r, dist_s, dist_r)
                    pred_fake = self.D_A(fake_A)
                    g_A_loss_adv = self.criterionGAN(pred_fake, True)
                    fake_B = self.G(image_r, image_s, mask_r, mask_s, dist_r, dist_s)
                    pred_fake = self.D_B(fake_B)
                    g_B_loss_adv = self.criterionGAN(pred_fake, True)
                    loss_gan = (g_A_loss_adv + g_B_loss_adv) * self.lamda_gan * 0.5

                    # loss_idt
                    idt_A = self.G(image_s, image_s, mask_s, mask_s, dist_s, dist_s)
                    idt_B = self.G(image_r, image_r, mask_r, mask_r, dist_r, dist_r)
                    loss_idt_A = self.criterionL1(idt_A, image_s) + self.LPIPS(idt_A, image_s) + self.LPIPS(fake_A, image_s)
                    loss_idt_B = self.criterionL1(idt_B, image_r) + self.LPIPS(idt_B, image_r) + self.LPIPS(fake_B, image_r)
                    loss_idt = (loss_idt_A + loss_idt_B) * self.lambda_idt * 0.5

                    # loss_make
                    g_A_loss_his = 0
                    g_B_loss_his = 0
                    # lip
                    g_A_lip_loss_his = self.criterionHis(fake_A, image_r, mask_s[:, 0], mask_r[:, 0], self.device) * self.lambda_his_lip
                    g_B_lip_loss_his = self.criterionHis(fake_B, image_s, mask_r[:, 0], mask_s[:, 0], self.device) * self.lambda_his_lip
                    g_A_loss_his += g_A_lip_loss_his
                    g_B_loss_his += g_B_lip_loss_his
                    # skin
                    g_A_skin_loss_his = self.criterionHis(fake_A, image_r, mask_s[:, 1], mask_r[:, 1], self.device) * self.lambda_his_skin
                    g_B_skin_loss_his = self.criterionHis(fake_B, image_s, mask_r[:, 1], mask_s[:, 1], self.device) * self.lambda_his_skin
                    g_A_loss_his += g_A_skin_loss_his
                    g_B_loss_his += g_B_skin_loss_his
                    # eye
                    g_A_eye_loss_his = self.criterionHis(fake_A, image_r, mask_s[:, 2], mask_r[:, 2], self.device) * self.lambda_his_eye
                    g_B_eye_loss_his = self.criterionHis(fake_B, image_s, mask_r[:, 2], mask_s[:, 2], self.device) * self.lambda_his_eye
                    g_A_loss_his += g_A_eye_loss_his
                    g_B_loss_his += g_B_eye_loss_his

                    loss_his = (g_A_loss_his + g_B_loss_his) * self.lamda_make * 0.5

                    # loss_reg
                    purified_A = self.H(fake_A)
                    purified_B = self.H(fake_B)
                    rec_A = self.H(self.G(purified_A, image_s, mask_s, mask_s, dist_s, dist_s))
                    rec_B = self.H(self.G(purified_B, image_r, mask_r, mask_r, dist_r, dist_r))
                    g_loss_rec_A = self.criterionL1(rec_A, image_s)
                    g_loss_rec_B = self.criterionL1(rec_B, image_r)
                    loss_reg = (g_loss_rec_A + g_loss_rec_B) * self.lamda_reg * 0.5

                    # loss_adv
                    targeted_loss_list = []
                    fake_A_diversity = []
                    fake_B_diversity = []
                    for i in range(self.diversity):
                        fake_A_diversity.append(self.input_diversity(self.input_noise(fake_A)).to(self.device))
                        fake_B_diversity.append(self.input_diversity(self.input_noise(fake_B)).to(self.device))
                    for model_name in self.targe_models.keys():
                        for i in range(self.diversity):
                            target_loss_A = self.cal_adv_loss(fake_A_diversity[i], self.target_image, model_name, self.targe_models) * self.lambda_adv * 0.5
                            target_loss_B = self.cal_adv_loss(fake_B_diversity[i], self.target_image, model_name, self.targe_models) * self.lambda_adv * 0.5
                            targeted_loss_list.append(target_loss_A)
                            targeted_loss_list.append(target_loss_B)
                    loss_adv = torch.mean(torch.stack(targeted_loss_list))

                    # total loss
                    g_loss = (loss_idt + loss_gan + loss_his + loss_reg + loss_adv).mean()
                    self.g_optimizer.zero_grad()
                    g_loss.backward(retain_graph=False)
                    self.g_optimizer.step()

                    # Logging
                    self.loss['G-idt-Loss'] = loss_idt.mean().item()
                    self.loss['G-gan-Loss'] = loss_gan.mean().item()
                    self.loss['G-his-Loss'] = loss_his.mean().item()
                    self.loss['G-reg-Loss'] = loss_reg.mean().item()
                    self.loss['G-adv-Loss'] = loss_adv.mean().item()

                ############################################ Train H ###################################################
                # G's output
                fake_A = self.G(image_s, image_r, mask_s, mask_r, dist_s, dist_r).detach()
                fake_B = self.G(image_r, image_s, mask_r, mask_s, dist_r, dist_s).detach()
                idt_A = self.G(image_s, image_s, mask_s, mask_s, dist_s, dist_s).detach()
                idt_B = self.G(image_r, image_r, mask_r, mask_r, dist_r, dist_r).detach()

                # loss_gan
                purified_A = self.H(fake_A)
                purified_B = self.H(fake_B)
                pred_fake = self.D_A(purified_A)
                h_A_loss_adv = self.criterionGAN(pred_fake, True)
                pred_fake = self.D_B(purified_B)
                h_B_loss_adv = self.criterionGAN(pred_fake, True)
                loss_gan_h = (h_A_loss_adv + h_B_loss_adv) * self.lamda_gan * 0.5

                # loss_idt
                purified_idt_A = self.H(idt_A)
                purified_idt_B = self.H(idt_B)
                loss_idt_A_h = self.criterionL1(purified_idt_A, image_s) + self.LPIPS(purified_idt_A, image_s) + self.LPIPS(purified_A, image_s)
                loss_idt_B_h = self.criterionL1(purified_idt_B, image_r) + self.LPIPS(purified_idt_B, image_r) + self.LPIPS(purified_B, image_r)
                loss_idt_h = (loss_idt_A_h + loss_idt_B_h) * self.lambda_idt * 0.5

                # loss_make
                h_A_loss_his = 0
                h_B_loss_his = 0
                # lip
                h_A_lip_loss_his = self.criterionHis(purified_A, image_r, mask_s[:, 0], mask_r[:, 0], self.device) * self.lambda_his_lip
                h_B_lip_loss_his = self.criterionHis(purified_B, image_s, mask_r[:, 0], mask_s[:, 0], self.device) * self.lambda_his_lip
                h_A_loss_his += h_A_lip_loss_his
                h_B_loss_his += h_B_lip_loss_his
                # skin
                h_A_skin_loss_his = self.criterionHis(purified_A, image_r, mask_s[:, 1], mask_r[:, 1], self.device) * self.lambda_his_skin
                h_B_skin_loss_his = self.criterionHis(purified_B, image_s, mask_r[:, 1], mask_s[:, 1], self.device) * self.lambda_his_skin
                h_A_loss_his += h_A_skin_loss_his
                h_B_loss_his += h_B_skin_loss_his
                # eye
                h_A_eye_loss_his = self.criterionHis(purified_A, image_r, mask_s[:, 2], mask_r[:, 2], self.device) * self.lambda_his_eye
                h_B_eye_loss_his = self.criterionHis(purified_B, image_s, mask_r[:, 2], mask_s[:, 2], self.device) * self.lambda_his_eye
                h_A_loss_his += h_A_eye_loss_his
                h_B_loss_his += h_B_eye_loss_his

                loss_his_h = (h_A_loss_his + h_B_loss_his) * self.lamda_make * 0.5

                # loss_reg
                rec_A = self.H(self.G(purified_A, image_s, mask_s, mask_s, dist_s, dist_s))
                rec_B = self.H(self.G(purified_B, image_r, mask_r, mask_r, dist_r, dist_r))
                h_loss_rec_A = self.criterionL1(rec_A, image_s)
                h_loss_rec_B = self.criterionL1(rec_B, image_r)
                loss_reg_h = (h_loss_rec_A + h_loss_rec_B) * self.lamda_reg * 0.5

                # loss_adv
                self_loss_list = []
                for model_name in self.targe_models.keys():
                    target_loss_A = self.cal_adv_loss(purified_A, image_s, model_name, self.targe_models) * self.lambda_adv * 0.5
                    target_loss_B = self.cal_adv_loss(purified_B, image_r, model_name, self.targe_models) * self.lambda_adv * 0.5
                    self_loss_list.append(target_loss_A)
                    self_loss_list.append(target_loss_B)
                loss_adv_h = torch.mean(torch.stack(self_loss_list))

                # total loss
                h_loss = (loss_idt_h + loss_gan_h + loss_his_h + loss_reg_h + loss_adv_h).mean()
                self.h_optimizer.zero_grad()
                h_loss.backward(retain_graph=False)
                self.h_optimizer.step()

                # Logging
                self.loss['H-idt-Loss'] = loss_idt_h.mean().item()
                self.loss['H-gan-Loss'] = loss_gan_h.mean().item()
                self.loss['H-his-Loss'] = loss_his_h.mean().item()
                self.loss['H-reg-Loss'] = loss_reg_h.mean().item()
                self.loss['H-adv-Loss'] = loss_adv_h.mean().item()


                # Print out log info
                if (self.i + 1) % self.log_step == 0:
                    self.log_terminal()

                # plot the figures
                for key_now in self.loss.keys():
                    plot_fig.plot(key_now, self.loss[key_now])

                # save the images
                if (self.i) % self.vis_step == 0:
                    print("Saving middle output...")
                    # self.vis_train([mask_s[:, :, 0], mask_r[:, :, 0],
                    #                 image_s, image_r,
                    #                 idt_A, idt_B, purified_idt_A, purified_idt_B,
                    #                 fake_A, fake_B, purified_A, purified_B,
                    #                 rec_A, rec_B
                    #                 ])
                    self.vis_train([mask_s[:, :, 0], image_s, image_r, idt_A, purified_idt_A, fake_A, purified_A, rec_A])

                # Save model checkpoints
                if (self.i) % self.snapshot_step == 0:
                    self.save_models()

                if (self.i % 100 == 99):
                    plot_fig.flush(self.log_path)

                plot_fig.tick()

            # Decay learning rate
            if (self.e + 1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                h_lr -= (self.h_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr, h_lr)
                print('Decay learning rate to g_lr: {}, d_lr:{}, h_lr:{}.'.format(g_lr, d_lr, h_lr))

    def update_lr(self, g_lr, d_lr, h_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_A_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.d_B_optimizer.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.h_optimizer.param_groups:
            param_group['lr'] = h_lr

    def save_models(self):
        if not osp.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)
        torch.save(
            self.G.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_A.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_A.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.D_B.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_D_B.pth'.format(self.e + 1, self.i + 1)))
        torch.save(
            self.H.state_dict(),
            os.path.join(
                self.snapshot_path, '{}_{}_H.pth'.format(self.e + 1, self.i + 1)))

    def vis_train(self, img_train_list):
        # saving training results
        mode = "train_vis"
        img_train_list = torch.cat(img_train_list, dim=3)
        result_path_train = osp.join(self.result_path, mode)
        if not osp.exists(result_path_train):
            os.makedirs(result_path_train)
        save_path = os.path.join(result_path_train, '{}_{}_fake.jpg'.format(self.e, self.i))
        save_image(self.de_norm(img_train_list.data), save_path, normalize=True)

    def log_terminal(self):
        elapsed = time.time() - self.start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
            elapsed, self.e + 1, self.num_epochs, self.i + 1, self.iters_per_epoch)

        for tag, value in self.loss.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)
