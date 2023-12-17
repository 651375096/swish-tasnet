import torch
# from asteroid_filterbanks import make_enc_dec
from src.asteroid_filterbanks import make_enc_dec
from ..masknn import TDConvNet
from .base_models import BaseEncoderMaskerDecoder
import warnings
import pdb

import matplotlib.pyplot as plt

# def huatu(data,name):
#     y = data
#     x = range(len(y))
#     plt.plot(y)
#     plt.title(name)
#     plt.show()



class ConvTasNet(BaseEncoderMaskerDecoder):
    """ConvTasNet separation model, as described in [1].

    Args:
      

        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References
        - [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
          for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
          https://arxiv.org/abs/1809.07454
    """
    print("BaseEncoderMaskerDecoder",BaseEncoderMaskerDecoder)
    def __init__(
        self,
        n_src,#输入混合中的源数
        out_chan=None,#估计掩码中的箱数。
        n_blocks=8,#每个卷积块的数量
        n_repeats=3,#重复次数。默认为3。
        bn_chan=128,#瓶颈之后的通道数
        hid_chan=512,#卷积中的信道数
        skip_chan=128,#跳过连接中的通道数。
        conv_kernel_size=3,#卷积块中的内核大小。
        norm_type="gLN",#：要从“`'BN'``”、“`'gLN``”中选择，“cLN”“”。
        mask_act="sigmoid",#生成掩码的非线性函数。
        in_chan=None,#输入通道数，应等于n_过滤器。
        causal=False,#卷积是否是因果的。
        fb_name="free",#用于制作编码器的Filterbank系列和解码器。要在[``'free'``、``'alytic_free'`、，``'param_sinc'``，``'tft'``]。
        kernel_size=16,#过滤器的长度
        n_filters=512,#过滤器数量/掩码网络的输入维度。
        stride=8,#卷积的步幅。
        encoder_activation=None,
        sample_rate=8000,
        **fb_kwargs,
    ):
        print("1111本地文件夹下的Tasnet")


        encoder, decoder = make_enc_dec(
            fb_name,####fb_name== free
            kernel_size=kernel_size,#        kernel_size=16,#过滤器的长度
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        # huatu(encoder,"encoder")
        # pdb.set_trace()
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        if causal and norm_type not in ["cgLN", "cLN"]:
            norm_type = "cLN"
            warnings.warn(
                "In causal configuration cumulative layer normalization (cgLN)"
                "or channel-wise layer normalization (chanLN)  "
                f"must be used. Changing {norm_type} to cLN"
            )
        # Update in_chan
        masker = TDConvNet(
            n_feats,
            n_src,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            causal=causal,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)


class VADNet(ConvTasNet):
    def forward_decoder(self, masked_tf_rep: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.sigmoid(self.decoder(masked_tf_rep))
