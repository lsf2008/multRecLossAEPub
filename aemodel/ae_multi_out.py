import torch
import torch.nn as nn

from aemodel.base import BaseModule
from aemodel.blocks_3d import DownsampleBlock
from aemodel.blocks_3d import UpsampleBlock
# from aemodel.estimator_2D import Estimator2D
from aemodel.layers.tsc import TemporallySharedFullyConnection
# from pytorch_lightning.core.mixins import HyperparametersMixin
from aemodel.varWght import VarWght
class Encoder(BaseModule):
    """
    ShanghaiTech model encoder.
    """
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of UCSD Ped2 samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, t, h, w = input_shape

        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=8, activation_fn=activation_fn, stride=(1, 2, 2)),
            DownsampleBlock(channel_in=8, channel_out=16, activation_fn=activation_fn, stride=(1, 2, 1)),
            DownsampleBlock(channel_in=16, channel_out=32, activation_fn=activation_fn, stride=(2, 1, 2)),
            # stride=(1,2,2)-->(2,2,1)
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn, stride=(2, 2, 1)),
            DownsampleBlock(channel_in=64, channel_out=64, activation_fn=activation_fn, stride=(2, 1, 2))
        )

        self.deepest_shape = (64, t // 8, h // 8, w // 8)

        # FC network
        dc, dt, dh, dw = self.deepest_shape
        self.tdl = nn.Sequential(
            TemporallySharedFullyConnection(in_features=(dc * dh * dw), out_features=512),
            activation_fn,

            TemporallySharedFullyConnection(in_features=512, out_features=code_length),
            # # nn.Sigmoid()
            activation_fn
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of patches.
        :return: the batch of latent vectors.
        """
        end_out = []
        h = x
        end_out.append(h)
        for conv in self.conv:
            h = conv(h)
            end_out.append(h)
        # h = self.conv(h)

        # Reshape for fully connected sub-network (flatten)
        c, t, height, width = self.deepest_shape

        # h = h.permute(0, 2, 3, 4, 1).contiguous()
        # h = h.view(-1, height, width, c)
        # h = GRN(c)(h)
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(-1, t, (c * height * width))

        o = self.tdl(h)

        # end_out.append(o)
        return o, end_out

class Encoder1(BaseModule):
    """
    ShanghaiTech model encoder.
    """
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int, int], int) -> None
        """
        Class constructor:

        :param input_shape: the shape of UCSD Ped2 samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder1, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, t, h, w = input_shape

        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=48, activation_fn=activation_fn, stride=(1, 2, 2)),
            DownsampleBlock(channel_in=48, channel_out=96, activation_fn=activation_fn, stride=(1, 2, 1)),
            DownsampleBlock(channel_in=96, channel_out=128, activation_fn=activation_fn, stride=(2, 1, 2)),
            # stride=(1,2,2)-->(2,2,1)
            DownsampleBlock(channel_in=128, channel_out=256, activation_fn=activation_fn, stride=(2, 2, 1)),
            DownsampleBlock(channel_in=256, channel_out=256, activation_fn=activation_fn, stride=(2, 1, 2))
        )

        self.deepest_shape = (256, t // 8, h // 8, w // 8)

        # FC network
        dc, dt, dh, dw = self.deepest_shape
        self.tdl = nn.Sequential(
            TemporallySharedFullyConnection(in_features=(dc * dh * dw), out_features=512),
            activation_fn,

            TemporallySharedFullyConnection(in_features=512, out_features=code_length),
            # # nn.Sigmoid()
            activation_fn
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the input batch of patches.
        :return: the batch of latent vectors.
        """
        end_out = []
        h = x
        end_out.append(h)
        for conv in self.conv:
            h = conv(h)
            end_out.append(h)
        # h = self.conv(h)

        # Reshape for fully connected sub-network (flatten)
        c, t, height, width = self.deepest_shape

        # h = h.permute(0, 2, 3, 4, 1).contiguous()
        # h = h.view(-1, height, width, c)
        # h = GRN(c)(h)
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(-1, t, (c * height * width))

        o = self.tdl(h)

        # end_out.append(o)
        return o, end_out

class Decoder1(BaseModule):
    """
    ShanghaiTech model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int, int], Tuple[int, int, int, int]) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of UCSD Ped2 samples.
        """
        super(Decoder1, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        dc, dt, dh, dw = deepest_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.tdl = nn.Sequential(
            TemporallySharedFullyConnection(in_features=code_length, out_features=512),
            activation_fn,

            TemporallySharedFullyConnection(in_features=512, out_features=(dc * dh * dw)),
            # activation_fn
            nn.Tanh()
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=dc, channel_out=256,
                          activation_fn=activation_fn, stride=(2, 1, 2), output_padding=(1, 0, 1)),
            # stride=(1,2,2)->(2,2,2) padding (0,1,1)_>(1,1,1)
            UpsampleBlock(channel_in=256, channel_out=128,
                          activation_fn=activation_fn, stride=(2, 2, 1), output_padding=(1, 1, 0)),
            UpsampleBlock(channel_in=128, channel_out=96,
                          activation_fn=activation_fn, stride=(2, 1, 2), output_padding=(1, 0, 1)),
            UpsampleBlock(channel_in=96, channel_out=48,
                          activation_fn=activation_fn, stride=(1, 2, 1), output_padding=(0, 1, 0)),
            UpsampleBlock(channel_in=48, channel_out=output_shape[0],
                          activation_fn=activation_fn, stride=(1, 2, 2), output_padding=(0, 1, 1)),
            # nn.Conv3d(in_channels=8, out_channels=output_shape[0], kernel_size=1, bias=False)
        )

    # noinspection LanguageDetectionInspection
    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        dec_out=[]
        h = self.tdl(h)

        # Reshape to encoder's deepest convolutional shape
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(len(h), *self.deepest_shape)
        dec_out.append(h)

        for con in self.conv:
            h = con(h)
            dec_out.append(h)
        # h = self.conv(h)

        o = h

        return o, dec_out



# noinspection LanguageDetectionInspection
class Decoder(BaseModule):
    """
    ShanghaiTech model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int, int], Tuple[int, int, int, int]) -> None
        """
        Class constructor.

        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of UCSD Ped2 samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        dc, dt, dh, dw = deepest_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.tdl = nn.Sequential(
            TemporallySharedFullyConnection(in_features=code_length, out_features=512),
            activation_fn,
            TemporallySharedFullyConnection(in_features=512, out_features=(dc * dh * dw)),
            # activation_fn
            nn.Tanh()
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=dc, channel_out=64,
                          activation_fn=activation_fn, stride=(2, 1, 2), output_padding=(1, 0, 1)),
            # stride=(1,2,2)->(2,2,2) padding (0,1,1)_>(1,1,1)
            UpsampleBlock(channel_in=64, channel_out=32,
                          activation_fn=activation_fn, stride=(2, 2, 1), output_padding=(1, 1, 0)),
            UpsampleBlock(channel_in=32, channel_out=16,
                          activation_fn=activation_fn, stride=(2, 1, 2), output_padding=(1, 0, 1)),
            UpsampleBlock(channel_in=16, channel_out=8,
                          activation_fn=activation_fn, stride=(1, 2, 1), output_padding=(0, 1, 0)),
            UpsampleBlock(channel_in=8, channel_out=output_shape[0],
                          activation_fn=activation_fn, stride=(1, 2, 2), output_padding=(0, 1, 1)),
            # nn.Conv3d(in_channels=8, out_channels=output_shape[0], kernel_size=1, bias=False)
        )

    # noinspection LanguageDetectionInspection
    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        dec_out=[]
        h = self.tdl(h)

        # Reshape to encoder's deepest convolutional shape
        h = torch.transpose(h, 1, 2).contiguous()
        h = h.view(len(h), *self.deepest_shape)
        dec_out.append(h)

        for con in self.conv:
            h = con(h)
            dec_out.append(h)
        # h = self.conv(h)

        o = h

        return o, dec_out

class AeMultiOut(BaseModule):
    """
    LSA model for ShanghaiTech video anomaly detection.
    output at each layer and use them for loss function
    """
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int, int], int, int) -> None
        """
        Class constructor.

        :param input_shape: the shape of UCSD Ped2 samples.
        :param code_length: the dimensionality of latent vectors.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(AeMultiOut, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.vw = VarWght()
        # self.norm = torch.nn.LayerNorm(code_length)
        # self.norm = torch.nn.BatchNorm1d(code_length)
        # if bacth_sz is not None:
        #     self.batch_size = bacth_sz
        # self.cpd_channels = cpd_channels
        # self.att = MultiheadAttention(code_length, 8, batch_first=True, )
        # self.gat = RegularGAT(h, w)
        # Build encoder
        self.encoder = Encoder(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = Decoder(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape
        )

        self.init_params()
    def init_params(self):
        self._init_params(self.encoder)
        self._init_params(self.decoder)
    def _init_params(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # 使用均匀分布初始化权重和偏置
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias:
                    torch.nn.init.uniform_(m.bias, -0.1, 0.1)
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias:
                    torch.nn.init.xavier_uniform_(m.bias)
    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Forward propagation.

        :param x: the input batch of patches.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors and CPD estimates.
        """
        h = x

        # Produce representations
        z, enc_out = self.encoder(h)

        # Reconstruct x
        x_r, dec_out = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z, enc_out, dec_out

class AeMultiOut1(BaseModule):
    """
    LSA model for ShanghaiTech video anomaly detection.
    output at each layer and use them for loss function
    """
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int, int], int, int) -> None
        """
        Class constructor.

        :param input_shape: the shape of UCSD Ped2 samples.
        :param code_length: the dimensionality of latent vectors.
        :param cpd_channels: number of bins in which the multinomial works.
        """
        super(AeMultiOut1, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        # self.norm = torch.nn.LayerNorm(code_length)
        # self.norm = torch.nn.BatchNorm1d(code_length)
        # if bacth_sz is not None:
        #     self.batch_size = bacth_sz
        # self.cpd_channels = cpd_channels
        # self.att = MultiheadAttention(code_length, 8, batch_first=True, )
        # self.gat = RegularGAT(h, w)
        # Build encoder
        self.encoder = Encoder1(
            input_shape=input_shape,
            code_length=code_length
        )

        # Build decoder
        self.decoder = Decoder1(
            code_length=code_length,
            deepest_shape=self.encoder.deepest_shape,
            output_shape=input_shape
        )

    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        Forward propagation.

        :param x: the input batch of patches.
        :return: a tuple of torch.Tensors holding reconstructions, latent vectors and CPD estimates.
        """
        h = x

        # Produce representations
        z, enc_out = self.encoder(h)

        # Reconstruct x
        x_r, dec_out = self.decoder(z)
        x_r = x_r.view(-1, *self.input_shape)

        return x_r, z, enc_out, dec_out
if __name__=='__main__':
    import torch

    input_shape = (3, 8, 32, 32)
    code_length = 64
    x = torch.randn((400, 3, 8, 32, 32))

    # aveShAe = AvenueShAE(input_shape, code_length)
    # x_r, z = aveShAe(x)
    # ------------------encoder---------------------
    end = AeMultiOut(input_shape, code_length, 2)
    x_r, z, enc_out, dec_out = end(x)
    dec_out=reversed(dec_out)
    for i in enc_out:
        print(i.shape)
    print('-------------------')
    for i in dec_out:
        print(i.shape)