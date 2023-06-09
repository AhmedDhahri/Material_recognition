import torch
import torch.nn as nn
from torch.autograd import Variable

def pyramid(ir, rgb, depth):
    ir, rgb = [ir.cuda()], [rgb.cuda()]
    for i in range(1, depth):
        ir.append(nn.functional.interpolate(ir[-1], scale_factor=0.5).cuda())
        rgb.append(nn.functional.interpolate(rgb[-1], scale_factor=0.5).cuda())
    return ir, rgb
class Encoder(nn.Module):
    def __init__(self, ch_s, ch_t):
        super().__init__()
        f, p = 3, 1
        self.At_0 = nn.Sequential(
            nn.Conv2d(ch_t, 16, f, stride=1, padding=p), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.Conv2d(16, 16, f, stride=1, padding=p), nn.BatchNorm2d(16), nn.Sigmoid()
        )
        self.At_1 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, f, stride=1, padding=p), nn.BatchNorm2d(32), nn.Sigmoid(),
            nn.Conv2d(32, 32, f, stride=1, padding=p), nn.BatchNorm2d(32), nn.Sigmoid()
        )
        self.At_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, f, stride=1, padding=p), nn.BatchNorm2d(64), nn.Sigmoid(),
            nn.Conv2d(64, 64, f, stride=1, padding=p), nn.BatchNorm2d(64), nn.Sigmoid()
        )

        self.As_0 = nn.Sequential(
            nn.Conv2d(ch_s, 16, f, stride=1, padding=p), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.Conv2d(16, 16, f, stride=1, padding=p), nn.BatchNorm2d(16), nn.Sigmoid()
        )
        self.As_1 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, f, stride=1, padding=p), nn.BatchNorm2d(32), nn.Sigmoid(),
            nn.Conv2d(32, 32, f, stride=1, padding=p), nn.BatchNorm2d(32), nn.Sigmoid()
        )
        self.As_2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, f, stride=1, padding=p), nn.BatchNorm2d(64), nn.Sigmoid(),
            nn.Conv2d(64, 64, f, stride=1, padding=p), nn.BatchNorm2d(64), nn.Sigmoid()
        )
    def forward(self, s0, s1, s2, s3, s4, s5,
                       t0, t1, t2, t3, t4, t5):
        #stage 0
        s0 = self.As_0(s0)
        t0 = self.At_0(t0)

        #stage 1
        s10 = self.As_0(s1)
        s11 = self.As_1(s0)
        s1 = torch.cat((s10, s11), dim=-3)
        t10 = self.At_0(t1)
        t11 = self.At_1(t0)
        t1 = torch.cat((t10, t11), dim=-3)

        #stage 2
        s20 = self.As_0(s2)
        s21 = self.As_1(s10)
        s2 = torch.cat((s20, s21, self.As_2(s11)), dim=-3)
        del s10, s11
        t20 = self.At_0(t2)
        t21 = self.At_1(t10)
        t2 = torch.cat((t20, t21, self.At_2(t11)), dim=-3)
        del t10, t11

        #stage 3
        s30 = self.As_0(s3)
        s31 = self.As_1(s20)
        s3 = torch.cat((s30, s31, self.As_2(s21)), dim=-3)
        del s20, s21
        t30 = self.At_0(t3)
        t31 = self.At_1(t20)
        t3 = torch.cat((t30, t31, self.At_2(t21)), dim=-3)
        del t20, t21

        #stage 4
        s40 = self.As_0(s4)
        s41 = self.As_1(s30)
        s4 = torch.cat((s40, s41, self.As_2(s31)), dim=-3)
        del s30, s31
        t40 = self.At_0(t4)
        t41 = self.At_1(t30)
        t4 = torch.cat((t40, t41, self.At_2(t31)), dim=-3)
        del t30, t31

        #stage 5
        s5 = torch.cat((self.As_0(s5), self.As_1(s40), self.As_2(s41)), dim=-3)
        del s40, s41
        t5 = torch.cat((self.At_0(t5), self.At_1(t40), self.At_2(t41)), dim=-3)
        del t40, t41


        return s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5


class Flow_Module(nn.Module):
    def __init__(self, n_rank):
        super(Flow_Module, self).__init__()

        ch_in = 224
        if n_rank == 0:
            ch_in = 32
        elif n_rank == 1:
            ch_in = 96
        self.pk = nn.Sequential(nn.Conv2d(ch_in, 32, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(32, 64, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(64, 16, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(16, 2, 1)
          )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, f, s, t):
        t = self.pk(torch.cat((t, self.warp(s, f)),-3)) #delta_fk
        f = t + f #fk
        t = self.warp(s, f) #w(sk)
        f = self.upsample(f) #f_k-1

        return t, f


    def warp(self, x, flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid).cuda() + flo.cuda()

        # scale grid to [-1,1]
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=False)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=False)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1

        return output*mask

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow_pred_0 = Flow_Module(0)
        self.flow_pred_1 = Flow_Module(1)
        self.flow_pred_2 = Flow_Module(2)

    def forward(self, s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5):
        B, _, H, H = s5.size()
        s5, fk = self.flow_pred_2(torch.zeros(B, 2, H, H).cuda(), s5, t5)
        s4, fk = self.flow_pred_2(fk, s4, t4)
        s3, fk = self.flow_pred_2(fk, s3, t3)
        s2, fk = self.flow_pred_2(fk, s2, t2)
        s1, fk = self.flow_pred_1(fk, s1, t1)
        s0, fk = self.flow_pred_0(fk, s0, t0)

        return s0, s1, s2, s3, s4, s5

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.F0 = nn.Sequential(nn.Conv2d(48, 16, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(16, 16, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(16, 3, 1))

        self.F1 =  nn.Sequential(nn.Conv2d(128, 32, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(32, 32, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Upsample(scale_factor=2),
                                  nn.Conv2d(32, 16, 3, 1, 1)# size 2 not 3
                                  )

        self.F2 =  nn.Sequential(nn.Conv2d(288, 64, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(64, 64, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Upsample(scale_factor=2),
                                  nn.Conv2d(64, 32, 3, 1, 1)# size 2 not 3
                                  )

        self.F3 =  nn.Sequential(nn.Conv2d(352, 128, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(128, 128, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Upsample(scale_factor=2),
                                  nn.Conv2d(128, 64, 3, 1, 1)# size 2 not 3
                                  )

        self.F4 =  nn.Sequential(nn.Conv2d(480, 256, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(256, 256, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Upsample(scale_factor=2),
                                  nn.Conv2d(256, 128, 3, 1, 1)# size 2 not 3
                                  )

        self.F5 =  nn.Sequential(nn.Conv2d(224, 512, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Conv2d(512, 512, 3, 1, 1),
                                  nn.Sigmoid(),
                                  nn.Upsample(scale_factor=2),
                                  nn.Conv2d(512, 256, 3, 1, 1)# size 2 not 3
                                  )
    def forward(self, ws0, ws1, ws2, ws3, ws4, ws5, t0, t1, t2, t3, t4, t5):
        t5 = self.F5(torch.cat((ws5, t5), dim=-3))
        t4 = self.F4(torch.cat((ws4, t4, t5), dim=-3))
        t3 = self.F3(torch.cat((ws3, t3, t4), dim=-3))
        t2 = self.F2(torch.cat((ws2, t2, t3), dim=-3))
        t1 = self.F1(torch.cat((ws1, t1, t2), dim=-3))
        t0 = self.F0(torch.cat((ws0, t0, t1), dim=-3))
        return t0


class OF_AE(nn.Module):
    def __init__(self, full=True):
        super().__init__()
        self.encoder = Encoder(1, 3).cuda()
        self.fusion = Fusion().cuda()
        self.decoder = Decoder().cuda()
        self.FULL = full

    def forward(self, I_r, I_v):
        I_r, I_v = pyramid(I_r, I_v, 6)
        s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5 = self.encoder(
            I_r[0], I_r[1], I_r[2], I_r[3], I_r[4], I_r[5],
            I_v[0], I_v[1], I_v[2], I_v[3], I_v[4], I_v[5])

        s0, s1, s2, s3, s4, s5 = self.fusion(s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5)
        if not self.FULL:
            return s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5
        else:
            return self.decoder(s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5)

class OF_AE_bb(nn.Module):
    def __init__(self, full=True):
        super().__init__()
        self.of_ae = OF_AE(full = False)

        self.adapt_conv0 = nn.Conv2d(32, 64, 3, 4, 1)
        self.adapt_conv1 = nn.Conv2d(96, 64, 3, 2, 1)
        self.adapt_conv2 = nn.Conv2d(224, 64, 3, 1, 1)

    def forward(self, Ir, Iv):
        s0, s1, s2, s3, s4, s5, t0, t1, t2, t3, t4, t5 = self.of_ae(Ir, Iv)
        #add features at each resnet block

        Iv = self.adapt_conv0(torch.cat((s0, t0), dim=-3)) + self.adapt_conv1(torch.cat((s1, t1), dim=-3)) + self.adapt_conv2(torch.cat((s2, t2), dim=-3))
        return Iv
