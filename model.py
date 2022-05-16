import torch
from torch.nn import functional as F
from torch.optim import Adam
import torchvision

from argparse import ArgumentParser

import typing
from typing import Optional, Tuple, Type, Sequence, Union

from monai.networks.nets import *
from monai.networks.blocks import Upsample
from monai.networks.layers import Norm, Act
from monai.losses import DiceLoss
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
    VolumeRenderer, 
    GridRaysampler, 
    NDCMultinomialRaysampler, NDCGridRaysampler, MonteCarloRaysampler, 
    EmissionAbsorptionRaymarcher, AbsorptionOnlyRaymarcher, 
)

from pytorch3d.transforms import (
    so3_exp_map,
)

from pytorch_lightning import LightningModule

# Init random cameras
# https://github.com/facebookresearch/pytorch3d/blob/main/tests/test_cameras.py

def init_random_cameras(
    cam_type: typing.Type[CamerasBase], batch_size: int, random_z: bool = False
):
    cam_params = {}
    # T = torch.randn(batch_size, 3) * 0.03
    # if not random_z:
    #     T[:, 2] = 4
    # R = so3_exp_map(torch.randn(batch_size, 3) * 3.0)
    dist = torch.rand(batch_size) * 2 + 3.0
    elev = torch.randn(batch_size) * 0.5
    azim = torch.randn(batch_size) * 0.5
    R, T = look_at_view_transform(dist, elev, azim)

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
            cam_params["fov"] = torch.rand(batch_size) * 20 + 50
            cam_params["aspect_ratio"] = torch.rand(batch_size) * 0.5 + 0.5
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

class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_shape=[256] * 3, voxel_size=0.1):
        super().__init__()
        # # After evaluating torch.sigmoid(self.log_colors), we get 
        # # densities close to zero.
        # self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_shape))
        # # After evaluating torch.sigmoid(self.log_colors), we get 
        # # a neutral gray color everywhere.
        # self.log_colors = torch.nn.Parameter(torch.zeros(3, *volume_shape))

        self._voxel_size = voxel_size
        # Store the renderer module as well.
        self._renderer = renderer
        
    def forward(self, cameras, volumes):
        batch_size = cameras.R.shape[0]

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
        # (the 2nd output is a re.5presentation of the sampled
        # rays which can be omitted for our purpose).
        # return self._renderer(cameras=cameras, volumes=volumes)[0]
        # screen_RGBA = screen_RGBA.reshape(B, self.shape, self.shape, 4).permute(0,3,2,1) # 3 for NeRF
        screen_RGBA, _ = self._renderer(cameras=cameras, volumes=volumes) #[...,:3]
        screen_RGBA = screen_RGBA.permute(0,3,2,1) # 3 for NeRF
        screen_RGB = screen_RGBA[:,:3].mean(dim=1, keepdim=True)
        normalized = lambda x: (x - x.min())/(x.max() - x.min())
        screen_RGB = normalized(screen_RGB)
        return screen_RGB

from data import *

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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # cameras = RandomCameras(batch_size=hparams.batch_size, random=True).to(device)

    # render_size describes the size of both sides of the 
    # rendered images in pixels. We set this to the same size
    # as the target images. I.e. we render at the same
    # size as the ground truth images.
    render_size = hparams.shape

    # Our rendered scene is centered around (0,0,0) 
    # and is enclosed inside a bounding box
    # whose side is roughly equal to 3.0 (world units).
    volume_extent_world = 4.0

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
        volume_shape = [volume_shape] * 3, 
        voxel_size = volume_extent_world / volume_shape,
    ).to(device)


    debug_data = first(datamodule.train_dataloader())
    image3d = debug_data['image3d'].to(device)
    volumes = Volumes(
        features = torch.cat([image3d]*3, dim=1),
        densities = torch.ones_like(image3d) / 1000., 
        voxel_size = volume_extent_world / volume_shape,
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
        torchvision.utils.save_image(screens[idx,0,:,:].detach().cpu(), f'test_camera_{idx}.png')