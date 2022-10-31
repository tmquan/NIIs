from math import degrees
from cv2 import norm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
import torchvision
import math
from argparse import ArgumentParser

import typing
from typing import Optional, Tuple, Type, Sequence, Union, Dict

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.blocks import Upsample
from monai.networks.layers import Norm, Act
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import InterpolateMode, deprecated_arg

from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.structures import Volumes
from pytorch3d.ops.utils import eyes
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer.cameras import (
    CamerasBase,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    SfMOrthographicCameras,
    SfMPerspectiveCameras,
    look_at_rotation,
    look_at_view_transform, 
    get_world_to_view_transform, 
    camera_position_from_spherical_angles,
)

from pytorch3d.renderer import (
    ray_bundle_to_ray_points, 
    RayBundle, 
    ImplicitRenderer,
    VolumeRenderer, 
    VolumeSampler, 
    GridRaysampler, 
    NDCMultinomialRaysampler, NDCGridRaysampler, MonteCarloRaysampler, 
    EmissionAbsorptionRaymarcher, AbsorptionOnlyRaymarcher, 
)

from pytorch3d.renderer.implicit.raymarching import (
    _check_density_bounds,
    _check_raymarcher_inputs,
    _shifted_cumprod,
)

from pytorch3d.renderer.implicit.raysampling import (
    _xy_to_ray_bundle
)

from pytorch3d.renderer.implicit.utils import (
    _validate_ray_bundle_variables, 
    ray_bundle_variables_to_ray_points
)

from pytorch3d.transforms import (
    so3_exp_map,
)

from pytorch_lightning import LightningModule

from data import *
# Init random cameras
# https://github.com/facebookresearch/pytorch3d/blob/main/tests/test_cameras.py
cam_mu = {
    "dist": 3.0,
    "elev": 0.0,
    "azim": 180,
    "fov": 60.,
    # "aspect_ratio": 1.0,
}
cam_bw = {
    "dist": 0.3,
    "elev": 90., #"elev": 0.,
    "azim": 180,   #"azim": 0,
    "fov": 30.,
    # "aspect_ratio": 0.2
}

def init_random_cameras(
    cam_type: typing.Type[CamerasBase], 
    batch_size: int, 
    random: bool = False, 
    cam_mu: Dict = cam_mu, 
    cam_bw: Dict = cam_bw, 
    cam_ft: torch.Tensor = None,
    device: torch.device = torch.device("cpu"),
):
    if cam_ft is not None:
        assert cam_ft.shape[0] == batch_size
        dist = cam_ft[:, 0] * cam_bw["dist"] + cam_mu["dist"]
        elev = cam_ft[:, 1] * cam_bw["elev"] + cam_mu["elev"] 
        azim = cam_ft[:, 2] * cam_bw["azim"] + cam_mu["azim"] 
        fov = cam_ft[:, 3] * cam_bw["fov"] + cam_mu["fov"] 
    else:
        dist = torch.Tensor(batch_size).uniform_(cam_mu["dist"] - cam_bw["dist"], cam_mu["dist"] + cam_bw["dist"]) if random else cam_mu["dist"]
        elev = torch.Tensor(batch_size).uniform_(cam_mu["elev"] - cam_bw["elev"], cam_mu["elev"] + cam_bw["elev"]) if random else cam_mu["elev"]
        azim = torch.Tensor(batch_size).uniform_(cam_mu["azim"] - cam_bw["azim"], cam_mu["azim"] + cam_bw["azim"]) if random else cam_mu["azim"]
        fov = torch.Tensor(batch_size).uniform_(cam_mu["fov"] - cam_bw["fov"], cam_mu["fov"] + cam_bw["fov"]) if random else cam_mu["fov"]

    cam_params = {}
    
    R, T = look_at_view_transform(dist.float(), elev.float(), azim.float(), degrees=True, device=device)
    R = R.float()
    T = T.float()
    cam_params = {"R": R, "T": T, "fov": fov}
    return cam_type(**cam_params)

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc

class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc

class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels

class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size, penc.size
        )
        return tensor + penc

class CustomUNet(nn.Module):
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.
    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.
    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.
    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.
    Examples::
        from monai.networks.nets import UNet
        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )
        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )
    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.
    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.
    """

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        mode: str = "conv",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.mode = mode 

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool, mode: str
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False, mode)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top, mode)  # create layer in upsampling path
            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True, self.mode)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and and decoding (up) sides of the network.
        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool, mode: str) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        if mode == "conv":
            conv = Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.up_kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                conv_only=is_top and self.num_res_units == 0,
                is_transposed=True,
                adn_ordering=self.adn_ordering,
            )
        elif mode == "deconv" or mode == "nontrainable" or mode == "pixelshuffle":
            conv = Upsample(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                scale_factor=2,
                size=None,
                mode=mode, #"nontrainable", "pixelshuffle", "deconv"
                pre_conv ="default",
                interp_mode=InterpolateMode.LINEAR,
                align_corners=True,
                bias=True,
                apply_pad_pool=True,
            )
        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                in_channels,
                in_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            # conv = nn.Sequential(conv, ru)
            conv = nn.Sequential(ru, conv)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--shape", type=int, default=256, help="isotropic shape")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    hparams = parser.parse_args()
    # Create data module
    train_image3d_folders = [
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),

        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
    ]
    train_label3d_folders = [
    ]

    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]

    train_label2d_folders = [
    ]

    val_image3d_folders = train_image3d_folders
    val_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = CustomDataModule(
        train_image3d_folders = train_image3d_folders, 
        train_image2d_folders = train_image2d_folders, 
        val_image3d_folders = val_image3d_folders, 
        val_image2d_folders = val_image2d_folders, 
        test_image3d_folders = test_image3d_folders, 
        test_image2d_folders = test_image2d_folders, 
        batch_size = hparams.batch_size, 
        shape = hparams.shape
    )
    datamodule.setup(seed=hparams.seed)

    # Set up the environment
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # cameras = RandomCameras(batch_size=hparams.batch_size, random=True).to(device)

    # render_size describes the size of both sides of the 
    # rendered images in pixels. We set this to the same size
    # as the target images. I.e. we render at the same
    # size as the ground truth images.
    render_size = hparams.shape

    # Our rendered scene is centered around (0,0,0) 
    # and is enclosed inside a bounding box
    # whose side is roughly equal to 3.0 (world units).
    volume_extent_world = 5.0

    # 1) Instantiate the raysampler.
    # Here, NDCMultinomialRaysampler generates a rectangular image
    # grid of rays whose coordinates follow the PyTorch3D
    # coordinate conventions.
    # Since we use a volume of size 256^3, we sample n_pts_per_ray=150,
    # which roughly corresponds to a one ray-point per voxel.
    # We further set the min_depth=0.1 since there is no surface within
    # 0.1 units of any camera plane.
    raysampler = NDCMultinomialRaysampler(
        image_width=render_size,
        image_height=render_size,
        n_pts_per_ray=512,
        min_depth=0.001,
        max_depth=volume_extent_world,
    )


    # 2) Instantiate the raymarcher.
    # Here, we use the standard EmissionAbsorptionRaymarcher 
    # which marches along each ray in order to render
    # each ray into a single 3D color vector 
    # and an opacity scalar.
    # raymarcher = EmissionAbsorptionRaymarcher()
    raymarcher = EmissionAbsorptionRaymarcher()

    # Finally, instantiate the volumetric render
    # with the raysampler and raymarcher objects.
    renderer = VolumeRenderer(
        raysampler=raysampler, 
        raymarcher=raymarcher,
    )

    # Instantiate the volumetric model.
    # We use a cubical volume with the size of 
    # one side = 256. The size of each voxel of the volume 
    # is set to volume_extent_world / volume_shape s.t. the
    # volume represents the space enclosed in a 3D bounding box
    # centered at (0, 0, 0) with the size of each side equal to 3.
    volume_shape = hparams.shape
    volume_model = VolumeModel(
        renderer,
        # volume_shape = [volume_shape] * 3, 
        # voxel_size = volume_extent_world / volume_shape,
    ).to(device)


    debug_data = first(datamodule.val_dataloader())
    image3d = debug_data['image3d'].to(device)
    volumes = Volumes(
        features = torch.cat([image3d]*3, dim=1),
        densities = torch.ones_like(image3d) / 512., #image3d / 512., 
        #torch.ones_like(image3d) / 512., # modify here
        voxel_size = 3.2 / volume_shape,
    )

    #
    # Set up the camera
    #
    cameras = init_random_cameras(cam_type=FoVPerspectiveCameras, batch_size=hparams.batch_size)
    cameras = cameras.to(device)

    #
    # Smoke test random cameras
    #
    screens = volume_model(cameras=cameras, volumes=volumes)

    for idx in range(hparams.batch_size):
        torchvision.utils.save_image(screens[idx,0,:,:].detach().cpu(), f'test_camera_standardization_{idx}.png')
