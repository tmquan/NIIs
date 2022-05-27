from math import degrees
from cv2 import norm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
import torchvision

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
    RayBundle, 
    VolumeRenderer, 
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

from pytorch3d.transforms import (
    so3_exp_map,
)

from pytorch_lightning import LightningModule

from data import *
# Init random cameras
# https://github.com/facebookresearch/pytorch3d/blob/main/tests/test_cameras.py
cam_mu = {
    "dist": 3.7,
    "elev": 0.0,
    "azim": 0.0,
    "fov": 60.,
    "aspect_ratio": 1.0,
}
cam_bw = {
    "dist": 0.3,
    "elev": 20.,
    "azim": 20.,
    "fov": 20.,
    "aspect_ratio": 0.1
}

def init_random_cameras(
    cam_type: typing.Type[CamerasBase], 
    batch_size: int, 
    random: bool = False, 
    cam_mu: Dict = cam_mu, 
    cam_bw: Dict = cam_bw, 
    cam_ft: torch.Tensor = None,
):
    if cam_ft is not None:
        assert cam_ft.shape[0] == batch_size
        dist = cam_ft[:, 0] * cam_bw["dist"] + cam_mu["dist"]
        elev = cam_ft[:, 1] * cam_bw["elev"] + cam_mu["elev"]
        azim = cam_ft[:, 2] * cam_bw["azim"] + cam_mu["azim"]
    else:
        dist = torch.Tensor(batch_size).uniform_(cam_mu["dist"] - cam_bw["dist"], cam_mu["dist"] + cam_bw["dist"]) if random else cam_mu["dist"]
        elev = torch.Tensor(batch_size).uniform_(cam_mu["elev"] - cam_bw["elev"], cam_mu["elev"] + cam_bw["elev"]) if random else cam_mu["elev"]
        azim = torch.Tensor(batch_size).uniform_(cam_mu["azim"] - cam_bw["azim"], cam_mu["azim"] + cam_bw["azim"]) if random else cam_mu["azim"]

    cam_params = {}
    # T = torch.randn(batch_size, 3) * 0.03
    # if not random:
    #     T[:, 2] = 4
    # R = so3_exp_map(torch.randn(batch_size, 3) * 3.0)


    R, T = look_at_view_transform(dist, elev, azim, degrees=True)

    cam_params = {"R": R, "T": T}
    if cam_type in (OpenGLPerspectiveCameras, OpenGLOrthographicCameras):
        cam_params["znear"] = torch.rand(batch_size) * 10 + 0.1
        cam_params["zfar"] = torch.rand(batch_size) * 4 + 1 + cam_params["znear"]
        if cam_type == OpenGLPerspectiveCameras:
            cam_params["fov"] = torch.rand(batch_size) * 60 + 30
            cam_params["aspect_ratio"] = torch.rand(batch_size) * 0.5 + 0.5
        else:
            cam_params["top"] = torch.rand(batch_size) * 0.2 + 0.9
            cam_params["bottom"] = -(torch.rand(batch_size)) * 0.2 - 0.9
            cam_params["left"] = -(torch.rand(batch_size)) * 0.2 - 0.9
            cam_params["right"] = torch.rand(batch_size) * 0.2 + 0.9
    elif cam_type in (FoVPerspectiveCameras, FoVOrthographicCameras):
        # cam_params["znear"] = torch.rand(batch_size) * 10 + 0.1
        # cam_params["zfar"] = torch.rand(batch_size) * 4 + 1 + cam_params["znear"]
        cam_params["znear"] = torch.ones(batch_size) * .01 #torch.rand(batch_size) * 10 + 0.1
        cam_params["zfar"] = torch.ones(batch_size) * 3.5 #torch.rand(batch_size) * 4 + 1 + cam_params["znear"]
        
        if cam_type == FoVPerspectiveCameras:
            # cam_params["fov"] = torch.rand(batch_size) * 60 + 30
            if cam_ft is not None:
                assert cam_ft.shape[0] == batch_size
                cam_params["fov"] = cam_ft[:, 3] * cam_bw["fov"] + cam_mu["fov"]
                cam_params["aspect_ratio"] = cam_ft[:, 4] * cam_bw["aspect_ratio"] + cam_mu["aspect_ratio"]
            else:
                cam_params["fov"] = torch.Tensor(batch_size).uniform_(cam_mu["fov"] - cam_bw["fov"], cam_mu["fov"] + cam_bw["fov"]) if random else cam_mu["fov"]
                cam_params["aspect_ratio"] = torch.Tensor(batch_size).uniform_(cam_mu["aspect_ratio"] - cam_bw["aspect_ratio"], cam_mu["aspect_ratio"] + cam_bw["aspect_ratio"]) if random else cam_mu["aspect_ratio"]    
        else:
            cam_params["max_y"] = torch.rand(batch_size) * 0.2 + 0.9
            cam_params["min_y"] = -(torch.rand(batch_size)) * 0.2 - 0.9
            cam_params["min_x"] = -(torch.rand(batch_size)) * 0.2 - 0.9
            cam_params["max_x"] = torch.rand(batch_size) * 0.2 + 0.9
    elif cam_type in (
        SfMOrthographicCameras,
        SfMPerspectiveCameras,
        OrthographicCameras,
        PerspectiveCameras,
    ):
        cam_params["focal_length"] = torch.rand(batch_size) * 10 + 0.1
        cam_params["principal_point"] = torch.randn((batch_size, 2))

    else:
        raise ValueError(str(cam_type))
    return cam_type(**cam_params)

class VolumeModel(nn.Module):
    def __init__(self, renderer):
        super().__init__()
        # # After evaluating torch.sigmoid(self.log_colors), we get 
        # # densities close to zero.
        # self.log_densities = nn.Parameter(-4.0 * torch.ones(1, *volume_shape))
        # # After evaluating torch.sigmoid(self.log_colors), we get 
        # # a neutral gray color everywhere.
        # self.log_colors = nn.Parameter(torch.zeros(3, *volume_shape))

        # self._voxel_size = voxel_size
        # Store the renderer module as well.
        self._renderer = renderer
        
    def forward(self, cameras, volumes, norm_type="standardized"):
        # batch_size = cameras.R.shape[0]

        # # Convert the log-space values to the densities/colors
        # densities = torch.sigmoid(self.log_densities)
        # colors = torch.sigmoid(self.log_colors)
        
        # # Instantiate the Volumes object, making sure
        # # the densities and colors are correctly
        # # expanded batch_size-times.
        # volumes = Volumes(
        #     densities = densities[None].expand(
        #         batch_size, *self.log_densities.shape),
        #     features = colors[None].expand(
        #         batch_size, *self.log_colors.shape),
        #     voxel_size=self._voxel_size,
        # )
        
        # Given cameras and volumes, run the renderer
        # and return only the first output value 
        # (the 2nd output is a representation of the sampled
        # rays which can be omitted for our purpose).
        # return self._renderer(cameras=cameras, volumes=volumes)[0]
        # screen_RGBA = screen_RGBA.reshape(B, self.shape, self.shape, 4).permute(0,3,2,1) # 3 for NeRF
        screen_RGBA, _ = self._renderer(cameras=cameras, volumes=volumes) #[...,:3]
        # print(screen_RGBA)
        screen_RGBA = screen_RGBA.permute(0, 3, 2, 1) # 3 for NeRF
        screen_RGB = screen_RGBA[:, :3].mean(dim=1, keepdim=True)
        normalized = lambda x: (x - x.min())/(x.max() - x.min())
        standardized = lambda x: (x - x.mean())/(x.std() + 1e-8) # 1e-8 to avoid zero division
        if norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))
        return screen_RGB

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

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Sequential:
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
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return nn.Sequential(down, SkipConnection(subblock), up)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
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
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        # conv = Convolution(
        #     self.dimensions,
        #     in_channels,
        #     out_channels,
        #     strides=strides,
        #     kernel_size=self.up_kernel_size,
        #     act=self.act,
        #     norm=self.norm,
        #     dropout=self.dropout,
        #     bias=self.bias,
        #     conv_only=is_top and self.num_res_units == 0,
        #     is_transposed=True,
        # )

        conv = Upsample(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=2,
            size=None,
            mode= "nontrainable", #"nontrainable", #"pixelshuffle", "deconv"
            pre_conv ="default",
            interp_mode=InterpolateMode.LINEAR,
            align_corners=True,
            bias=True,
            apply_pad_pool=True,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
            )
            conv = nn.Sequential(conv, ru)

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
