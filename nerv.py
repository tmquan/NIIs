import os
import warnings
warnings.filterwarnings("ignore")
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

class PictureModel(nn.Module):
    def __init__(self, renderer):
        super().__init__()
        self._renderer = renderer
        
    def forward(self, cameras, volumes, norm_type="standardized", scaler=1.0):
        screen_RGBA, ray_bundles = self._renderer(cameras=cameras, volumes=volumes) #[...,:3]
        # rays_points = ray_bundle_to_ray_points(ray_bundles)

        screen_RGBA = screen_RGBA.permute(0, 3, 2, 1) # 3 for NeRF
        screen_RGB = screen_RGBA[:, :3].mean(dim=1, keepdim=True)
        normalized = lambda x: (x - x.min())/(x.max() - x.min() + 1e-8)
        standardized = lambda x: (x - x.mean())/(x.std() + 1e-8) # 1e-8 to avoid zero division
        if norm_type == "normalized":
            screen_RGB = normalized(screen_RGB)
        elif norm_type == "standardized":
            screen_RGB = normalized(standardized(screen_RGB))
        screen_RGB *= scaler
        # screen_RGB = torch.clamp(screen_RGB, 0.0, 1.0)
        return screen_RGB

class NeRVDataModule(LightningDataModule):
    def __init__(self, 
        train_image3d_folders: str = "path/to/folder", 
        train_image2d_folders: str = "path/to/folder", 
        val_image3d_folders: str = "path/to/folder", 
        val_image2d_folders: str = "path/to/folder", 
        test_image3d_folders: str = "path/to/folder", 
        test_image2d_folders: str = "path/to/dir", 
        train_samples: int = 1000,
        val_samples: int = 400,
        test_samples: int = 400,
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
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

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
            length=self.train_samples,
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
            length=self.val_samples,
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

# def _weights_init(m):
#     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)):
#         torch.nn.init.normal_(m.weight, 0.0, 0.02)
#     if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
#         torch.nn.init.normal_(m.weight, 0.0, 0.02)
#         torch.nn.init.constant_(m.bias, 0)

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
        self.shape = hparams.shape
        self.factor = hparams.factor
        self.scaler = hparams.scaler
        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.devices = hparams.devices
        self.save_hyperparameters()

        raysampler = NDCMultinomialRaysampler( #NDCGridRaysampler(
            image_width = self.shape,
            image_height = self.shape,
            n_pts_per_ray = 400, #self.shape,
            min_depth = 0.001,
            max_depth = 4.5,
        )

        # self.raymarcher = EmissionAbsorptionRaymarcher()
        raymarcher = EmissionAbsorptionRaymarcherFrontToBack() # X-Ray Raymarcher

        visualizer = VolumeRenderer(
            raysampler = raysampler, 
            raymarcher = raymarcher,
        )
        
        print("Self Device: ", self.device)

        self.viewer = PictureModel(
            renderer = visualizer,
        )
    
        self.opacity_net = nn.Sequential(
            UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2, 
                channels=(48, 96, 192, 384, 768, 1024), #(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2, 2),
                num_res_units=3,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.5,
                # mode="nontrainable",
            ), 
            nn.Sigmoid(), 
        )

        self.clarity_net = nn.Sequential(
            UNet(
                spatial_dims=2,
                in_channels=16, #self.shape,
                out_channels=self.shape,
                channels=(64, 128, 256, 512, 1024, 2048),
                strides=(2, 2, 2, 2, 2),
                num_res_units=3,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.5,
                # mode="nontrainable",
            ), 
            Reshape(*[1, self.shape, self.shape, self.shape]),
            nn.Sigmoid(), 
        )

        self.density_net = nn.Sequential(
            UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1, 
                channels=(48, 96, 192, 384, 768, 1024),
                strides=(2, 2, 2, 2, 2),
                num_res_units=3,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.5,
                # mode="nontrainable",
            ), 
            nn.Sigmoid(),  
        )

        self.frustum_net = nn.Sequential(
            DenseNet201(
                spatial_dims=2,
                in_channels=1,
                out_channels=5,
                act=("LeakyReLU", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout_prob=0.5,
                pretrained=True, 
            ),
            nn.Sigmoid(),
        )

        self.l1loss = nn.L1Loss(reduction="mean")
        

    def forward(self, image3d):
        pass

    def forward_picture(self, image3d: torch.Tensor, frustum_feat: torch.Tensor, 
        factor: float=1.0, 
        scaler: float=1.0, 
        opacities: str= 'stochastic',
        norm_type: str="normalized"
    ) -> torch.Tensor:
        # features = image3d.repeat(1, 3, 1, 1, 1)
        if opacities=='stochastic':
            radiances = self.opacity_net(image3d) # * 2. - 1.) * .5 + .5 #+ torch.randn_like(image3d)
        elif opacities=='deterministic':
            radiances = self.opacity_net(image3d) # * 2. - 1.) * .5 + .5
        elif opacities=='constant':
            radiances = torch.ones_like(image3d)

        features = image3d.repeat(1, 3, 1, 1, 1)
        # features = radiances[:,[0]].repeat(1, 3, 1, 1, 1)
        densities = radiances[:,[1]]
        # with torch.no_grad():
        frustums = init_random_cameras(cam_type=FoVPerspectiveCameras, 
                            batch_size=self.batch_size, 
                            cam_mu=cam_mu,
                            cam_bw=cam_bw,
                            cam_ft=frustum_feat*2. - 1.).to(image3d.device)
        volumes = Volumes(
            features = features, 
            densities = densities / factor,
            voxel_size = 3.2 / self.shape,
        )
                
        pictures = self.viewer(volumes=volumes, cameras=frustums, norm_type=norm_type, scaler=scaler)
        return pictures, radiances

    def forward_density(self, image2d: torch.Tensor, frustum_feat: torch.Tensor):
        # with torch.no_grad():
        zeros_tensor = torch.zeros(self.batch_size, 10, self.shape, self.shape)
        pos_encoding = PositionalEncodingPermute2D(10)(zeros_tensor)
        cat_features = torch.cat([image2d, 
                                  pos_encoding.to(image2d.device),
                                  frustum_feat.view(frustum_feat.shape[0], 
                                                    frustum_feat.shape[1], 1, 1).repeat(1, 1, self.shape, self.shape)], dim=1)
        
        clarity = self.clarity_net(cat_features) # * 2. - 1.) * .5 + .5
        density = self.density_net(clarity) # * 2. - 1.) * .5 + .5
        return clarity, density
    
    def forward_frustum(self, image2d: torch.Tensor):
        frustum = self.frustum_net(image2d) # * 2. - 1.) * .5 + .5 #[0]# [0, 1] 
        return frustum

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='evaluation'):   
        _device = batch["image3d"].device
        orgvol_ct = batch["image3d"]
        orgimg_xr = batch["image2d"]
        # with torch.no_grad():
        orgcam_ct = torch.rand(self.batch_size, 5, device=_device)

        # if stage=='train':
        #     if (batch_idx % 3) == 1:
        #         orgvol_ct = torch.rand_like(orgvol_ct)
        #     elif (batch_idx % 3) == 2:
        #         # Calculate interpolation
        #         alpha = torch.rand(self.batch_size, 1, 1, 1, 1, device=_device)
        #         vol3d = orgvol_ct.detach().clone()
        #         noise = torch.rand_like(vol3d)
        #         alpha = alpha.expand_as(vol3d)
        #         orgvol_ct = alpha * vol3d + (1 - alpha) * noise
        
         
        # XR path
        orgcam_xr = self.forward_frustum(orgimg_xr)
        estmid_xr, estvol_xr = self.forward_density(orgimg_xr, orgcam_xr)
        estimg_xr, estrad_xr = self.forward_picture(estvol_xr, orgcam_xr, factor=self.factor, opacities='stochastic', scaler=self.scaler, norm_type='normalized')
        reccam_xr = self.forward_frustum(estimg_xr)
        recmid_xr, recvol_xr = self.forward_density(estimg_xr, reccam_xr)
        recimg_xr, recrad_xr = self.forward_picture(recvol_xr, reccam_xr, factor=self.factor, opacities='stochastic', scaler=self.scaler, norm_type='normalized')
        
        # CT path
        estimg_ct, estrad_ct = self.forward_picture(orgvol_ct, orgcam_ct, factor=self.factor, opacities='stochastic', scaler=self.scaler, norm_type='normalized')
        estcam_ct = self.forward_frustum(estimg_ct)
        estmid_ct, estvol_ct = self.forward_density(estimg_ct, estcam_ct)
        recimg_ct, recrad_ct = self.forward_picture(estvol_ct, estcam_ct, factor=self.factor, opacities='stochastic', scaler=self.scaler, norm_type='normalized')
        
        if batch_idx == 0:
            viz2d = torch.cat([
                        torch.cat([orgvol_ct[...,self.shape//2], 
                                   estimg_ct,
                                   orgimg_xr], dim=-1),
                        torch.cat([estvol_ct[...,self.shape//2],
                                   recimg_ct, 
                                   estimg_xr], dim=-1),
                        torch.cat([estrad_ct[:, [1], ..., self.shape//2],
                                   recrad_ct[:, [1], ..., self.shape//2],
                                   estrad_xr[:, [1], ..., self.shape//2]], dim=-1),
                    ], dim=-2)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
        
        # Loss
        im3d_loss = self.l1loss(orgvol_ct, estvol_ct) \
                  + self.l1loss(estvol_xr, recvol_xr) 

        tran_loss = self.l1loss(estrad_ct, recrad_ct) \
                  + self.l1loss(estrad_xr, recrad_xr) \
                  + self.l1loss(orgvol_ct, estrad_ct[:,[0]]) \
                  + self.l1loss(estvol_xr, estrad_xr[:,[0]]) \
                  + self.l1loss(torch.one_like(orgvol_ct), estrad_ct[:,[1]]) \
                  + self.l1loss(torch.one_like(estvol_xr), estrad_xr[:,[1]]) 
                
        im2d_loss = self.l1loss(estimg_ct, recimg_ct) \
                  + self.l1loss(orgimg_xr, estimg_xr) 
                    
        cams_loss = self.l1loss(orgcam_ct, estcam_ct) \
                  + self.l1loss(orgcam_xr, reccam_xr) 

        info = {f'loss': 1e0*im3d_loss + 1e0*tran_loss + 1e0*im2d_loss + 1e0*cams_loss} 
        
        self.log(f'{stage}_im2d_loss', im2d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_im3d_loss', im3d_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_cams_loss', cams_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_tran_loss', tran_loss, on_step=(stage=='train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        
        return info
        
        # train generator
        # if optimizer_idx == 0:
        #     g_loss = self.gen_step(
        #         fake_images=torch.cat([estimg_ct, recimg_ct, estimg_xr, recimg_xr], dim=0),
        #         real_images=orgimg_xr
        #     )
        #     self.log(f'{stage}_g_loss', g_loss, on_step=True, prog_bar=False, logger=True)
        #     info = {f'loss': 1e0*im3d_loss + 1e0*tran_loss + 1e0*im2d_loss + 1e0*cams_loss
        #                    + g_loss} 
        #     return info

        # # train discriminator
        # elif optimizer_idx == 1:
        #     d_loss = self.discrim_step(
        #         fake_images=torch.cat([estimg_ct, recimg_ct, estimg_xr, recimg_xr], dim=0),
        #         real_images=orgimg_xr)
        #     d_grad = self.compute_gradient_penalty(fake_samples=estimg_ct, real_samples=orgimg_xr)
        #     self.log(f'{stage}_d_loss', d_loss, on_step=True, prog_bar=False, logger=True)
        #     info = {f'loss': d_loss+10*d_grad} 
        #     return info

    # def discrim_step(self, fake_images: torch.Tensor, real_images: torch.Tensor):
    #     real_logits = self.discriminator(real_images) 
    #     fake_logits = self.discriminator(fake_images) 
    #     real_loss = F.softplus(-real_logits).mean() 
    #     fake_loss = F.softplus(+fake_logits).mean()
    #     return real_loss + fake_loss 

    # def gen_step(self, fake_images: torch.Tensor, real_images: torch.Tensor):
    #     fake_logits = self.discriminator(fake_images) 
    #     fake_loss = F.softplus(-fake_logits).mean()
    #     return fake_loss 

    # def compute_gradient_penalty(self, fake_samples, real_samples):
    #     """Calculates the gradient penalty loss for WGAN GP"""
    #     # Random weight term for interpolation between real and fake samples
    #     alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
    #     # Get random interpolation between real and fake samples
    #     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    #     interpolates = interpolates.to(self.device)
    #     d_interpolates = self.discriminator(interpolates)
    #     # fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
    #     fake = torch.ones_like(d_interpolates)
    #     # Get gradient w.r.t. interpolates
    #     gradients = torch.autograd.grad(
    #         outputs=d_interpolates,
    #         inputs=interpolates,
    #         grad_outputs=fake,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True,
    #     )[0]
    #     gradients = gradients.view(gradients.size(0), -1).to(self.device)
    #     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #     return gradient_penalty

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='test')

    def _common_epoch_end(self, outputs, stage: Optional[str]='common'):
        loss = torch.stack([x[f'loss'] for x in outputs]).mean()
        self.log(f'{stage}_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def train_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='validation')
    
    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='test')

    def configure_optimizers(self):
        # opt_g = torch.optim.RAdam([
        #     {'params': self.opacity_net.parameters()}, 
        #     {'params': self.clarity_net.parameters()}, 
        #     {'params': self.density_net.parameters()}, 
        #     {'params': self.frustum_net.parameters()}, 
        # ], lr=1e0*(self.lr or self.learning_rate))
        # opt_d = torch.optim.RAdam([
        #     {'params': self.discriminator.parameters()}
        # ], lr=1e0*(self.lr or self.learning_rate))
        # return opt_g, opt_d
        # return torch.optim.RAdam([
        #         {'params': self.opacity_net.parameters()}], lr=1e0*(self.lr or self.learning_rate)), \
        #        torch.optim.RAdam([
        #         {'params': self.clarity_net.parameters()}, 
        #         {'params': self.density_net.parameters()}], lr=1e0*(self.lr or self.learning_rate)), \
        #        torch.optim.RAdam([
        #         {'params': self.frustum_net.parameters()}], lr=1e0*(self.lr or self.learning_rate)), \
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=10, eta_min=self.lr / 10
        # )
        # return [optimizer], [scheduler]
        return torch.optim.RAdam(self.parameters(), lr=self.lr)
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="NeRV")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    
    # Model arguments
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=501, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--scaler", type=float, default=20.0, help="XRay amplification")
    parser.add_argument("--factor", type=float, default=64.0, help="XRay transparency")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    
    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--reduction", type=str, default='sum', help="mean or sum")

    parser = Trainer.add_argparse_args(parser)
    
    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename='{epoch:02d}-{validation_loss_epoch:.2f}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=5, 
    )
    lr_callback = LearningRateMonitor(logging_interval='step')
    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.logsdir, log_graph=True)

    # Init model with callbacks
    trainer = Trainer.from_argparse_args(
        hparams, 
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback, 
        ],
        accumulate_grad_batches=4, 
        strategy="ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        precision=16,  #if hparams.use_amp else 32,
        # amp_backend='apex',
        # amp_level='O3', # see https://nvidia.github.io/apex/amp.html#opt-levels
        # stochastic_weight_avg=True,
        # auto_scale_batch_size=True, 
        # gradient_clip_val=5, 
        # gradient_clip_algorithm='norm', #'norm', #'value'
        # track_grad_norm=2, 
        detect_anomaly=True, 
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
        train_samples = hparams.train_samples,
        val_samples = hparams.val_samples,
        test_samples = hparams.test_samples,
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