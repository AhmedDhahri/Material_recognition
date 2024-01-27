import torch
import math
from ssim import SSIM
#from vgg_layers import Perceptual_Loss
import torchvision.models as models
from torchvision.transforms.functional import rgb_to_grayscale

class Combined_Loss_VAE:
    def __init__(self):
 #       self.vgg_loss = Perceptual_Loss()
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
#        self.vgg_loss = Perceptual_Loss()
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
    def __init__(self, gpu=0, m2irh=False):
        self.minc_irh = {1: 1, 3: 2, 16: 2, 2: 3, 4: 3, 7: 4, 9: 5, 10: 6, 11: 7, 5: 8, 6: 8, 8: 8, 12: 8, 17: 8, 18: 8, 22: 8, 13: 9, 14: 10, 21: 10, 15: 11, 19: 12, 20: 13, 23: 14}
        self.irh_minc = {1:[1], 2:[3, 16], 3:[2, 4], 4:[7], 5:[9], 6:[10], 7:[11], 8:[5, 6, 8, 12, 17, 18, 22], 9:[13], 10:[14, 21], 11:[15],  12:[19], 13:[20], 14:[23], 15:[]}
        self.m2irh = m2irh
        self.gpu = gpu

    def accuracy_irh(self, y_pred, y):
        y_pred = torch.argmax(y_pred, 1).detach().cpu()
        for i in range(y_pred.shape[0]):
            y_pred[i] = self.minc_irh[y_pred[i].item()]
            y[i] = self.minc_irh[y[i].item()]
        
        return 100 * torch.sum(y == y_pred)/y_pred.shape[0]

    def accuracy(self, y_pred, y):
        y_pred = torch.argmax(y_pred, 1).detach().cpu()
        return 100 * torch.sum(y == y_pred)/y_pred.shape[0]

    def accuracies(self, y_pred, y):
        true_pred_5 = 0
        true_pred_1 = 0
        y_pred = torch.argsort(y_pred, dim=1, descending=True)[:, :5].detach().cpu()
        y = torch.argmax(y, 1)
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
