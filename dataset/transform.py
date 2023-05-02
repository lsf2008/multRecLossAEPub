import torch
from torchvision.transforms import Grayscale
from scipy.ndimage.morphology import binary_dilation
from torchvision.transforms import ColorJitter

class MyColorJitter(ColorJitter):
    def __init__(self,brightness=0, contrast=0, saturation=0, hue=0):
        super(MyColorJitter, self).__init__()
    def forward(self, x):
        '''
        :param x:   (c,t,h,w)
        :return:
        '''
        x = x.permute(1,0,2,3)
        x = super().forward(x)
        x = x.permute(1,0,2,3)
        return x

class FilterCrops(object):
    def __init__(self, thr=0.1):
        self.thr = thr
    def __call__(self, x):
        '''
        Parameters
        ----------
        x；c, t, h, w
        Returns
        -------
        '''
        x1= x >0.001
        bt = x.shape[0]
        xn = []
        v = torch.prod(torch.tensor(x1.shape[1:])) * self.thr
        for i in range(bt):
            t = x[i, :, :, :, :]
            # print(torch.sum(x1[i,:,:,:,:]) )
            if torch.sum(x1[i,:,:,:,:]) > v:
                xn.append(t)
        return torch.stack(xn)


class ToCrops(object):
    """ Crops input clips """

    def __init__(self, raw_shape, crop_shape):
        self.raw_shape = raw_shape
        self.crop_shape = crop_shape

    def __call__(self, sample):
        X= sample

        c, t, h, w = self.raw_shape
        cc, tc, hc, wc = self.crop_shape

        crops_X = []
        # crops_Y = []

        for k in range(0, t, tc):
            for i in range(0, h, hc // 2):
                for j in range(0, w, wc // 2):
                    if (i + hc <= h) and (j + wc <=w) and k+tc<=t:
                        # print(f'{k, i, j}')
                        crops_X.append(X[:, k:k + tc, i:i + hc, j:j + wc])
                        # crops_Y.append(Y[:, k:k + tc, i:i + hc, j:j + wc])
        if len(crops_X)!=0:
            X = torch.stack(crops_X, dim=0)
        else:
            X = None
        # Y = torch.stack(crops_Y, dim=0)

        return X
class ToGrays(object):
    def __init__(self):
        super(ToGrays, self).__init__()
    def __call__(self, x):
        # x = c, t, h, w

        c, t, h, w = x.shape
        assert (c == 3), 'c must be 3'
        x = x.permute([1, 0, 2, 3])
        x = Grayscale()(x)
        x = x.permute([1, 0, 2, 3])
        return x

class ShortScaleImgs(torch.nn.Module):
    def __init__(
            self, size: int, interpolation: str = "bilinear"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return torch.nn.functional.interpolate(
            x, size=self._size, mode=self._interpolation, align_corners=False
        )

class RemoveBackground(torch.nn.Module):

    def __init__(self, threshold: float, background):
        super(RemoveBackground, self).__init__()
        self.threshold = threshold
        # h,w,c
        self.background= background

    def forward(self, X):
            # X: c, t, h, w
        X = X.permute([1, 2, 3, 0])
        mask = (torch.sum(torch.abs(X - self.background), dim=-1) > self.threshold)

        mask = torch.unsqueeze(mask, dim=-1).int()
        l = [torch.tensor(binary_dilation(mask_frame, iterations=5)) for mask_frame in mask]
        mask = torch.stack(l, 0)
        X *= mask
        X = X.permute([3,0,1,2])
        return X