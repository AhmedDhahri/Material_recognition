import torch
import math
from ssim import SSIM
from vgg_layers import Perceptual_Loss
import torchvision.models as models
from torchvision.transforms.functional import rgb_to_grayscale

class Combined_Loss_VAE:
    def __init__(self):
        self.vgg_loss = Perceptual_Loss()
        self.ssim = SSIM()

    def compute(self, Ir, Iv, mu, sigma, If):
        #color loss

        #mse loss
        lambda_m = 0.3
        l_mse = torch.nn.L1Loss()(If, Iv) #+ torch.nn.L1Loss()(rgb_to_grayscale(If), Ir)
        #l_mse = torch.nn.L1Loss()(If[:,0:1,:,:], Ir)

        #structural similarity loss
        lambda_s = 0.4
        l_ssim =  1 - self.ssim(Ir, rgb_to_grayscale(If))
        #l_ssim =  3 - self.ssim(Iv, If) - self.ssim(Iv[:,1:2,:,:], If[:,1:2,:,:]) - self.ssim(Iv[:,2:3,:,:], If[:,2:3,:,:])
        #MSE(Yv, Ir) SSIM(Cr, Cb)
        #vae loss
        mu_2 = torch.square(mu)
        sigma_2 = torch.square(sigma)
        l_vae = 0.5 * (torch.sum(mu_2 + sigma_2 - torch.log(sigma_2) - 1))

        #perceptual loss
        #lambda_p = 0.2
        #l_vgg = self.vgg_loss.compute(torch.cat([Ir, Ir, Ir],1), If)
        #combined loss
        return lambda_m * l_mse + lambda_s * l_ssim + l_vae #+ lambda_p * l_vgg

class Combined_Loss:
    def __init__(self):
        self.vgg_loss = Perceptual_Loss()
        self.ssim = SSIM()


    def compute(self, Ir, Iv, If):
        #color loss

        #mse loss
        lambda_m = 0.3
        l_mse = torch.nn.L1Loss()(Ir, rgb_to_grayscale(If).cuda()) #+ torch.nn.L1Loss()(rgb_to_grayscale(If), Ir)
        #l_mse = torch.nn.L1Loss()(If[:,0:1,:,:], Ir)

        #structural similarity loss
        lambda_s = 0.4
        l_ssim =  1 - self.ssim(Iv, If)
        #l_ssim =  3 - self.ssim(Iv, If) - self.ssim(Iv[:,1:2,:,:], If[:,1:2,:,:]) - self.ssim(Iv[:,2:3,:,:], If[:,2:3,:,:])
        #MSE(Yv, Ir) SSIM(Cr, Cb)
        #vae loss


        #perceptual loss
        #lambda_p = 0.2
        #l_vgg = self.vgg_loss.compute(torch.cat([Ir, Ir, Ir],1), If)
        #combined loss
        return lambda_m * l_mse #+ lambda_s * l_ssim

class Metrics:
    def __init__(self, m2irh=False):
        self.loss = torch.nn.CrossEntropyLoss().cuda()
        self.dict_minc_2_irh = {1:11, 2:2, 3:1, 4:2, 5:7, 6:7,
                                7:3, 8:7, 9:4, 10:5, 11:6, 12:7,
                                13:8, 14:9, 15:10, 16:1, 17:7,
                                18:7, 19:11, 20:7, 21:9, 22:7, 23:12}
        self.m2irh = m2irh

    def compute(self, y_pred, y):
        y_true = torch.zeros(y_pred.shape)
        for i in range(y_pred.shape[0]):
            y_true[i][y[i].item()] = 1
        return self.loss(y_pred, y_true.cuda())


    def accuracy(self, y_pred, y):
        y_pred = torch.argmax(y_pred, 1)
        y_pred = torch.reshape(y_pred, (-1,))
        return 100 * torch.sum(y.cuda() == y_pred)/y_pred.shape[0]

    def accuracies(self, y_pred, y):
        true_pred_5 = 0
        true_pred_1 = 0
        y_pred = torch.argsort(y_pred, dim=1, descending=True)[:, :5]
        for i in range(y_pred.shape[0]):
            if y[i] == y_pred[i, 0]:
                true_pred_1 +=1
                true_pred_5 +=1
            elif y[i] in y_pred[i]:
                true_pred_5 +=1
        return true_pred_1/y_pred.shape[0], true_pred_5/y_pred.shape[0]

class CosineDecayLR:
    def __init__(self, optimizer, initial_lr, max_iter):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_iter = max_iter

    def step(self, iter):
        lr = self.initial_lr * 0.5 * (1 + math.cos(math.pi * iter / self.max_iter))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
