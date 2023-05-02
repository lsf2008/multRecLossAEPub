import torch
from aemodel.base import BaseModule
class VarWght1D(BaseModule):
    def __int__(self):
        super(VarWght, self).__int__()

    def forward(self, x):
        '''compute the variance weight for the feature maps

            Parameters
            ----------
            x : tensor (b, c, t, h, w)
                input tensor

            Returns
            -------
            _type_
                weighted feature maps
            '''
        # c,t, h, w
        # x = torch.randn(200, 64, 3, 5, 5)
        b, c, t, h, w = x.shape
        x = x.transpose(2, 1)
        # combine dimension of time axis into 1
        if t >1:
            conv11 = torch.nn.Conv3d(in_channels=t, out_channels=1, kernel_size=1)
            # if conv11.device != x.device:
            conv11 = conv11.to(x.device)
            # (b, 1, c, h, w)
            x = conv11(x)
            # print(f'1*1convolution shape:{x.shape}')

            dim_var = self.cmpWght(x)

            # print(f'variance shape:{dim_var.shape}')

            # return to the original size
            conv1t = torch.nn.Conv3d(1, out_channels=t, kernel_size=1)
            # if conv1t.device != x.device:
            conv1t = conv1t.to(x.device)
            x = conv1t(x)
        else:
            dim_var = self.cmpWght(x)
            # print(f'1*1 return convolution shape:{x.shape}')

            x = x.transpose(2, 1)
        # print(f'1*1 return convolution shape:{x.shape}')
        x = x * dim_var
        # print(x.shape)
        return x
    def cmpWght(self, x):
        # b, c, t, h, w / b, t, c, h, w
        dim_var = torch.var(x, dim=2)+10e-7  # b, c, h, w
        # dim_var = torch.nn.functional.normalize(dim_var, p=2, dim=[-2,-1])
        softMax = torch.nn.Softmax2d()
        soft_wgh = softMax(dim_var)  # b, c, h, w
        # dim_var = soft_wgh.unsqueeze(1)
        return soft_wgh

class VarWght(BaseModule):
    def __init__(self, is_app=False):
        super(VarWght, self).__init__()
        self.is_app = is_app
    def forward(self, x, alph1=1, alph2=1):
        b, c, t, h, w = x.shape
        if self.is_app:
            feaWght = self.cmpFeaWhgt(x)
            x = x * feaWght
        else:
            if t>1:
                # feature weight
                feaWght = self.cmpFeaWhgt(x)
                x1 = x * feaWght
                tmpWght = self.cmpTempWght(x)
                x2 = x * tmpWght
                x = alph1*x1 + alph2*x2
            else:
                feaWght = self.cmpFeaWhgt(x)
                x = x * feaWght
        return x


    def cmpWght(self, x):
        # b, c, t, h, w / b, t, c, h, w
        dim_var = torch.var(x, dim=2)+10e-7  # b, c, h, w
        # dim_var = torch.nn.functional.normalize(dim_var, p=2, dim=[-2,-1])
        softMax = torch.nn.Softmax2d()
        soft_wgh = softMax(dim_var)  # b, c, h, w
        # dim_var = soft_wgh.unsqueeze(1)
        return soft_wgh

    def cmpFeaWhgt(self, x):
        # b, c, t, h, w
        x = x.transpose(2, 1)
        wght = self.cmpWght(x)
        wght = wght.unsqueeze(1)
        return wght

    def cmpTempWght(self, x):
        # b, c, t, h, w
        wght = self.cmpWght(x)
        wght = wght.unsqueeze(2)
        return wght

if __name__=='__main__':
    x = torch.randn(200, 64, 3, 5,5)
    vw = VarWght()
    x = vw(x)
    print(x.shape)