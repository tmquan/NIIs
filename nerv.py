import os
import math 
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
# from sympy import re

# from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
import torch
torch.cuda.empty_cache()
# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# from kornia.color import GrayscaleToRgb
# from kornia.augmentation import Normalize

from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.networks.layers import * #Reshape
from monai.networks.nets import * #UNet, DenseNet121, Generator
from monai.losses import DiceLoss

from model import *
from data import *

def _shifted_cumprod(x, shift=1):
    """
    Computes `torch.cumprod(x, dim=-1)` and prepends `shift` number of
    ones and removes `shift` trailing elements to/from the last dimension
    of the result.
    """
    x_cumprod = torch.cumprod(x, dim=-1)
    x_cumprod_shift = torch.cat(
        [torch.ones_like(x_cumprod[..., :shift]), x_cumprod[..., :-shift]], dim=-1
    )
    return x_cumprod_shift

class EmissionAbsorptionRaymarcherFrontToBack(EmissionAbsorptionRaymarcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,
        **kwargs,
    ) -> torch.Tensor:
        rays_densities = rays_densities[..., 0]
        # print(rays_densities.shape)
        # absorption = _shifted_cumprod(
        #     (1.0 + eps) - rays_densities, shift=self.surface_thickness
        # )
        # weights = rays_densities * absorption
        # features = (weights[..., None] * rays_features).sum(dim=-2)
        # opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities.flip(dims=(-1,)), shift=-self.surface_thickness
        ).flip(dims=(-1,)) # Reverse the direction of the absorption to match X-ray detector
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        return torch.cat((features, opacities), dim=-1)

class ScreenModel(nn.Module):
    def __init__(self, renderer):
        super().__init__()
        self._renderer = renderer
        
    def forward(self, cameras, volumes, norm_type="standardized"):
        screen_RGBA, ray_bundles = self._renderer(cameras=cameras, volumes=volumes) #[...,:3]
        rays_points = ray_bundle_to_ray_points(ray_bundles)

        screen_RGBA = screen_RGBA.permute(0, 3, 2, 1) # 3 for NeRF
        screen_RGB = screen_RGBA[:, :3].mean(dim=1, keepdim=True)
        normalized = lambda x: (x - x.min())/(x.max() - x.min() + 1e-8)
        standardized = lambda x: (x - x.mean())/(x.std() + 1e-8) # 1e-8 to avoid zero division
        if norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))
        return screen_RGB

class NeRVDataModule(LightningDataModule):
    def __init__(self, 
        train_image3d_folders: str = "path/to/folder", 
        train_image2d_folders: str = "path/to/folder", 
        val_image3d_folders: str = "path/to/folder", 
        val_image2d_folders: str = "path/to/folder", 
        test_image3d_folders: str = "path/to/folder", 
        test_image2d_folders: str = "path/to/dir", 
        shape: int = 256,
        batch_size: int = 32
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shape = shape
        # self.setup() 
        self.train_image3d_folders = train_image3d_folders
        self.train_image2d_folders = train_image2d_folders
        self.val_image3d_folders = val_image3d_folders
        self.val_image2d_folders = val_image2d_folders
        self.test_image3d_folders = test_image3d_folders
        self.test_image2d_folders = test_image2d_folders

        # self.setup()
        def glob_files(folders: str=None, extension: str='*.nii.gz'):
            assert folders is not None
            paths = [glob.glob(os.path.join(folder, extension), recursive = True) for folder in folders]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files
            
        self.train_image3d_files = glob_files(folders=train_image3d_folders, extension='**/*.nii.gz')
        self.train_image2d_files = glob_files(folders=train_image2d_folders, extension='**/*.png')
        
        self.val_image3d_files = glob_files(folders=val_image3d_folders, extension='**/*.nii.gz') # TODO
        self.val_image2d_files = glob_files(folders=val_image2d_folders, extension='**/*.png')
        
        self.test_image3d_files = glob_files(folders=test_image3d_folders, extension='**/*.nii.gz') # TODO
        self.test_image2d_files = glob_files(folders=test_image2d_folders, extension='**/*.png')

    def setup(self, seed: int=2222, stage: Optional[str]=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image3d", "image2d"]),
                AddChanneld(keys=["image3d", "image2d"],),
                Spacingd(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"]),  
                Rotate90d(keys=["image2d"], k=3),
                OneOf([
                    Orientationd(keys=('image3d'), axcodes="ARI"),
                    # Orientationd(keys=('image3d'), axcodes="PRI"),
                    # Orientationd(keys=('image3d'), axcodes="ALI"),
                    # Orientationd(keys=('image3d'), axcodes="PLI"),
                    # Orientationd(keys=["image3d"], axcodes="LAI"),
                    # Orientationd(keys=["image3d"], axcodes="RAI"),
                    # Orientationd(keys=["image3d"], axcodes="LPI"),
                    # Orientationd(keys=["image3d"], axcodes="RPI"),
                    ],
                ),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                OneOf([
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                            a_min=-200, 
                            a_max=1500,
                            b_min=0.0,
                            b_max=1.0),
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                            a_min=-500, #-200, 
                            a_max=3071, #1500,
                            b_min=0.0,
                            b_max=1.0),
                ]),
                # RandZoomd(keys=["image3d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["trilinear"], align_corners=True), 
                # RandZoomd(keys=["image2d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["area"]), 
                RandFlipd(keys=["image2d"], prob=1.0, spatial_axis=1),
                # RandFlipd(keys=["image3d"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image3d"], prob=0.5, spatial_axis=1),

                RandScaleCropd(keys=["image3d"], 
                               roi_scale=(0.9, 0.9, 0.8), 
                               max_roi_scale=(1.0, 1.0, 0.8), 
                               random_center=False, 
                               random_size=False),
                # RandAffined(keys=["image3d"], rotate_range=None, shear_range=None, translate_range=20, scale_range=None),
                # CropForegroundd(keys=["image3d"], source_key="image3d", select_fn=lambda x: x>0, margin=0),
                # CropForegroundd(keys=["image2d"], source_key="image2d", select_fn=lambda x: x>0, margin=0),
                Resized(keys=["image3d"], spatial_size=self.shape, size_mode="longest", mode=["trilinear"], align_corners=True),
                Resized(keys=["image2d"], spatial_size=self.shape, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image3d", "image2d"], k=self.shape, mode="constant", constant_values=0),
                
                ToTensord(keys=["image3d", "image2d"],),
            ]
        )

        self.train_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.train_image3d_files, self.train_image2d_files], 
            transform=self.train_transforms,
            length=1000,
            batch_size=self.batch_size,
        )

        self.train_loader = DataLoader(
            self.train_datasets, 
            batch_size=self.batch_size, 
            num_workers=4, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image3d", "image2d"]),
                AddChanneld(keys=["image3d", "image2d"],),
                Spacingd(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"]),  
                Rotate90d(keys=["image2d"], k=3),
                RandFlipd(keys=["image2d"], prob=1.0, spatial_axis=1), #Right cardio
                OneOf([
                    Orientationd(keys=('image3d'), axcodes="ARI"),
                    # Orientationd(keys=('image3d'), axcodes="PRI"),
                    # Orientationd(keys=('image3d'), axcodes="ALI"),
                    # Orientationd(keys=('image3d'), axcodes="PLI"),
                    # Orientationd(keys=["image3d"], axcodes="LAI"),
                    # Orientationd(keys=["image3d"], axcodes="RAI"),
                    # Orientationd(keys=["image3d"], axcodes="LPI"),
                    # Orientationd(keys=["image3d"], axcodes="RPI"),
                    ],
                ), 
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                OneOf([
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                            a_min=-200, 
                            a_max=1500,
                            b_min=0.0,
                            b_max=1.0),
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                            a_min=-500, #-200, 
                            a_max=3071, #1500,
                            b_min=0.0,
                            b_max=1.0),
                ]),
                Resized(keys=["image3d"], spatial_size=self.shape, size_mode="longest", mode=["trilinear"], align_corners=True),
                Resized(keys=["image2d"], spatial_size=self.shape, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image3d", "image2d"], k=self.shape, mode="constant", constant_values=0),
            
                ToTensord(keys=["image3d", "image2d"],),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.val_image3d_files, self.val_image2d_files], 
            transform=self.val_transforms,
            length=400,
            batch_size=self.batch_size,
        )
        
        self.val_loader = DataLoader(
            self.val_datasets, 
            batch_size=self.batch_size, 
            num_workers=4, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader

def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

cam_mu = {
    "dist": 3.0,
    "elev": 0.0,
    "azim": 0.0,
    "fov": 60.,
    "aspect_ratio": 1.0,
}
cam_bw = {
    "dist": 0.4,
    "elev": 30.,
    "azim": 30.,
    "fov": 30.,
    "aspect_ratio": 0.2
}

# NeRPLightningModule
class NeRVLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.logsdir = hparams.logsdir
        self.lr = hparams.lr
        self.b1 = hparams.b1
        self.b2 = hparams.b2
        self.wd = hparams.wd
        self.eps = hparams.eps
        self.shape = hparams.shape
        self.batch_size = hparams.batch_size

        # self.theta = hparams.theta # use for factor to promote xray effect
        # self.alpha = hparams.alpha
        # self.kappa = hparams.kappa
        # self.gamma = hparams.gamma
        # self.delta = hparams.delta
        # self.reduction = hparams.reduction
        self.save_hyperparameters()

        self.raysampler = NDCMultinomialRaysampler( #NDCGridRaysampler(
            image_width = self.shape,
            image_height = self.shape,
            n_pts_per_ray = 200, #self.shape,
            min_depth = 0.001,
            max_depth = 4.5,
        )

        # self.raymarcher = EmissionAbsorptionRaymarcher()
        self.raymarcher = EmissionAbsorptionRaymarcherFrontToBack() # X-Ray Raymarcher

        self.visualizer = VolumeRenderer(
            raysampler = self.raysampler, 
            raymarcher = self.raymarcher,
        )
        
        print("Self Device: ", self.device)

        self.viewer = ScreenModel(
            renderer = self.visualizer,
        )

        self.opaque_net = nn.Sequential(
            UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1, 
                channels=(32, 64, 128, 256, 512, 1024),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                # dropout=0.5,
                # norm=Norm.BATCH,
                # mode="nontrainable",
            ), 
            nn.Sigmoid()  
        )

        self.reform_net = nn.Sequential(
            UNet(
                spatial_dims=2,
                in_channels=16, #self.shape,
                out_channels=self.shape,
                channels=(64, 128, 256, 512, 1024, 1600),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                # dropout=0.5,
                # norm=Norm.BATCH,
                # mode="nontrainable",
            ), 
            Reshape(*[1, self.shape, self.shape, self.shape]),
            nn.Sigmoid(), 
        )

        self.refine_net = nn.Sequential(
            UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1, 
                channels=(32, 64, 128, 256, 512, 1024),
                strides=(2, 2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                # dropout=0.5,
                # norm=Norm.BATCH,
                # mode="nontrainable",
            ), 
            nn.Sigmoid()  
        )

        self.camera_net = nn.Sequential(
            DenseNet201(
                spatial_dims=2,
                in_channels=1,
                out_channels=5,
                act=("LeakyReLU", {"inplace": True}),
                # dropout_prob=0.5,
                # norm=Norm.BATCH,
                pretrained=True, 
            ),
            nn.Sigmoid(),
        )
        self.l1loss = nn.L1Loss(reduction="mean")
        
    def forward(self, image3d):
        pass

    def forward_screen(self, image3d: torch.Tensor, camera_feat: torch.Tensor, 
        factor: float=1.0, 
        opacities: str= 'stochastic',
        norm_type: str="normalized"
    ) -> torch.Tensor:
        features = image3d.expand(-1, 3, -1, -1, -1) #torch.cat([image3d]*3, dim=1)
        
        if opacities=='stochastic':
            densities = self.opaque_net(image3d) #+ torch.randn_like(image3d)
        elif opacities=='deterministic':
            densities = self.opaque_net(image3d)
        elif opacities=='constant':
            densities = torch.ones_like(image3d)
        
        cameras = init_random_cameras(cam_type=FoVPerspectiveCameras, 
                            batch_size=self.batch_size, 
                            cam_mu=cam_mu,
                            cam_bw=cam_bw,
                            cam_ft=camera_feat*2. - 1.).to(image3d.device)
        volumes = Volumes(
            features = features, 
            densities = densities / factor,
            voxel_size = 3.2 / self.shape,
        )
            
        screen = self.viewer(volumes=volumes, cameras=cameras, norm_type=norm_type)
        return screen, densities

    def forward_volume(self, image2d: torch.Tensor, camera_feat: torch.Tensor):
        code2d = torch.zeros(image2d.shape[0], 10, self.shape, self.shape, device=image2d.device)
        penc2d = PositionalEncodingPermute2D(10)(code2d)
        concat = torch.cat([image2d, 
                            penc2d,
                            camera_feat.view(camera_feat.shape[0], 
                                             camera_feat.shape[1], 1, 1).repeat(1, 1, self.shape, self.shape)], dim=1)
        
        reform = self.reform_net(concat)# * 2.0 - 1.0) * 0.5 + 0.5
        refine = self.refine_net(reform)# * 2.0 - 1.0) * 0.5 + 0.5
        return reform, refine
    
    def forward_camera(self, image2d: torch.Tensor):
        camera = self.camera_net(image2d) #[0] # [0, 1] 
        return camera

    def training_step(self, batch, batch_idx, optimizer_idx=None, stage: Optional[str]='train'):
        return self._sharing_step(batch, batch_idx, optimizer_idx, stage)   

    def _sharing_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='evaluation'):   
        _device = batch["image3d"].device
        orgvol_ct = batch["image3d"]
        orgimg_xr = batch["image2d"]
        orgcam_ct = torch.distributions.uniform.Uniform(0, 1).sample([self.batch_size, 5]).to(_device)
    
        if batch_idx%5==1:
            orgvol_ct = torch.distributions.uniform.Uniform(0, 1).sample(batch["image3d"].shape).to(_device)
        elif batch_idx%5==2:
            orgimg_xr = torch.distributions.uniform.Uniform(0, 1).sample(batch["image2d"].shape).to(_device)
        elif batch_idx%5==3:
            orgvol_ct = torch.distributions.uniform.Uniform(0, 1).sample(batch["image3d"].shape).to(_device)
            orgimg_xr = torch.distributions.uniform.Uniform(0, 1).sample(batch["image2d"].shape).to(_device)

        # with torch.cuda.amp.autocast():
        if stage=='train':
            opacities = 'stochastic'
        elif stage=='validation' or stage=='test':
            opacities = 'deterministic'
        elif stage=='constant':
            opacities = 'constant'
        
        # XR path
        orgcam_xr = self.forward_camera(orgimg_xr)
        estmid_xr, estvol_xr = self.forward_volume(orgimg_xr, orgcam_xr)
        estimg_xr, estalp_xr = self.forward_screen(estvol_xr, orgcam_xr, factor=20.0, opacities=opacities, norm_type="normalized")
        reccam_xr = self.forward_camera(estimg_xr)
        recmid_xr, recvol_xr = self.forward_volume(estimg_xr, reccam_xr)

        # CT path
        estimg_ct, estalp_ct = self.forward_screen(orgvol_ct, orgcam_ct, factor=20.0, opacities=opacities, norm_type="normalized")
        estcam_ct = self.forward_camera(estimg_ct)
        estmid_ct, estvol_ct = self.forward_volume(estimg_ct, estcam_ct)
        recimg_ct, recalp_ct = self.forward_screen(estvol_ct, estcam_ct, factor=20.0, opacities=opacities, norm_type="normalized")
        
        # Loss
        im3d_loss = self.l1loss(orgvol_ct, estvol_ct) \
                  + self.l1loss(orgvol_ct, estmid_ct) \
                  + self.l1loss(estvol_xr, recmid_xr) \
                  + self.l1loss(estvol_xr, recvol_xr) \
                # + self.l1loss(estmid_xr, estvol_xr) \
                # + self.l1loss(estmid_xr, recmid_xr) \
                  
        im2d_loss = self.l1loss(estimg_ct, recimg_ct) \
                  + self.l1loss(orgimg_xr, estimg_xr) \
                    
        cams_loss = self.l1loss(orgcam_ct, estcam_ct) \
                  + self.l1loss(orgcam_xr, reccam_xr) \
        
        tran_loss = self.l1loss(estalp_ct, 1.0 + torch.randn_like(estalp_ct)) \
                  + self.l1loss(estalp_xr, 1.0 + torch.randn_like(estalp_xr)) \
                # + self.l1loss(recalp_ct, 1.0 + torch.randn_like(recalp_ct)) \


        # if stage=='train':
        #     opacities = 'stochastic'
        #     tran_loss = self.l1loss(estalp_ct, 1.0 + torch.randn_like(estalp_ct)) \
        #               + self.l1loss(estalp_xr, 1.0 + torch.randn_like(estalp_xr)) \
        #            #  + self.l1loss(recalp_ct, 1.0 + torch.randn_like(recalp_ct)) \
        # elif stage=='validation' or stage=='test':
        # else:
        #     opacities = 'deterministic'
        #     tran_loss = self.l1loss(estalp_ct, torch.ones_like(estalp_ct)) \
        #               + self.l1loss(estalp_xr, torch.ones_like(estalp_xr)) \
        #            #  + self.l1loss(recalp_ct, torch.ones_like(recalp_ct)) \

        info = {f'loss': 1e0*im3d_loss + 1e0*im2d_loss + 1e0*cams_loss+ 1e0*tran_loss} 

        self.log(f'{stage}_im2d_loss', im2d_loss, on_step=(stage=='train'), prog_bar=True, logger=True)
        self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage=='train'), prog_bar=True, logger=True)
        self.log(f'{stage}_cams_loss', cams_loss, on_step=(stage=='train'), prog_bar=True, logger=True)
        self.log(f'{stage}_tran_loss', tran_loss, on_step=(stage=='train'), prog_bar=True, logger=True)

        if batch_idx == 0:
            with torch.no_grad():
                viz = torch.cat([
                        torch.cat([orgvol_ct[...,self.shape//2], 
                                   estimg_ct,
                                   orgimg_xr], dim=-1),
                        torch.cat([estvol_ct[...,self.shape//2],
                                   recimg_ct, 
                                   estimg_xr], dim=-1),
                        ], dim=-2)
                grid = torchvision.utils.make_grid(viz, normalize=False, scale_each=False, nrow=1, padding=0)
                tensorboard = self.logger.experiment
                tensorboard.add_image(f'{stage}_samples', grid, self.current_epoch*self.batch_size + batch_idx)

                plot_2d_or_3d_image(data=torch.cat([torch.cat([orgvol_ct, estvol_ct, estvol_xr], dim=-2), 
                                                    torch.cat([estalp_ct, estalp_xr, recalp_ct], dim=-2)], dim=-3), 
                                                    tag=f'{stage}_gif', writer=tensorboard, step=self.current_epoch, frame_dim=-1)
        return info
        # if optimizer_idx==0:
        #     return {f'loss': 1e0*cams_loss} 
        # elif optimizer_idx==1:
        #     return {f'loss': 1e0*im2d_loss} 
        # elif optimizer_idx==2:
        #     return {f'loss': 1e0*im3d_loss} 
        # else:
        #     return {f'loss': 1e0*im3d_loss + 1e0*im2d_loss + 1e0*cams_loss} 

        
    def validation_step(self, batch, batch_idx):
        return self._sharing_step(batch, batch_idx, optimizer_idx=None, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._sharing_step(batch, batch_idx, optimizer_idx=None, stage='test')

    def evaluation_epoch_end(self, outputs, stage: Optional[str]='evaluation'):
        loss = torch.stack([x[f'loss'] for x in outputs]).mean()
        self.log(f'{stage}_loss_epoch', loss, on_step=False, prog_bar=True, logger=True)

    def train_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='validation')
    
    def test_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='test')

    def configure_optimizers(self):
        # opt_cam = torch.optim.RAdam([{'params': self.camera_net.parameters()},], lr=1e0*(self.lr or self.learning_rate))
        # opt_scr = torch.optim.RAdam([{'params': self.opaque_net.parameters()},], lr=1e0*(self.lr or self.learning_rate))
        # opt_vol = torch.optim.RAdam([{'params': self.reform_net.parameters()},
        #                              {'params': self.refine_net.parameters()},], lr=1e0*(self.lr or self.learning_rate))
        # # opt_all = torch.optim.RAdam(self.parameters(), lr=1e0*(self.lr or self.learning_rate))
        # return opt_cam, opt_scr, opt_vol #, opt_all
        return torch.optim.RAdam(self.parameters(), lr=1e0*(self.lr or self.learning_rate))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="NeRV")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    
    # Model arguments
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=501, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--wd", type=float, default=1e-6, help="adam: weight decay")
    parser.add_argument("--eps", type=float, default=1e-8, help="adam: epsilon")
    parser.add_argument("--b1", type=float, default=0.0, help="adam: 1st order momentum")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: 2nd order momentum")
    
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    
    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--reduction", type=str, default='sum', help="mean or sum")

    parser = Trainer.add_argparse_args(parser)
    
    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(2222)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename='{epoch:02d}-{validation_loss_epoch:.2f}',
        save_top_k=-1,
        save_last=True,
        # monitor='validation_r_loss', 
        # mode='min',
        every_n_epochs=5, 
    )
    lr_callback = LearningRateMonitor(logging_interval='step')
    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.logsdir, log_graph=True)

    # Init model with callbacks
    trainer = Trainer.from_argparse_args(
        hparams, 
        max_epochs=hparams.epochs,
        # ckpt_path = hparams.ckpt, #"logs/default/version_0/epoch=50.ckpt",
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback, 
            # DeviceStatsMonitor(), 
            # ModelSummary(max_depth=-1), 
            # tensorboard_callback
        ],
        accumulate_grad_batches=4, 
        strategy="ddp", #"horovod", #"deepspeed", #"ddp_sharded",
        precision=16,
        # stochastic_weight_avg=True,
        # auto_scale_batch_size=True, 
        # gradient_clip_val=5, 
        # gradient_clip_algorithm='norm', #'norm', #'value'
        # track_grad_norm=2, 
        # detect_anomaly=True, 
        # profiler="simple",
    )

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
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),
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

    val_image3d_folders = [
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
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MELA2022/raw/val/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/AMOS2022/raw/val/images'),
    ]
    val_image2d_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = NeRVDataModule(
        train_image3d_folders = train_image3d_folders, 
        train_image2d_folders = train_image2d_folders, 
        val_image3d_folders = val_image3d_folders, 
        val_image2d_folders = val_image2d_folders, 
        test_image3d_folders = test_image3d_folders, 
        test_image2d_folders = test_image2d_folders, 
        batch_size = hparams.batch_size, 
        shape = hparams.shape
    )
    datamodule.setup()

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = NeRVLightningModule(
        hparams = hparams
    )
    model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model


    trainer.fit(
        model, 
        datamodule,
    )

    # test

    # serve
