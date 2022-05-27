import os

from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from kornia.color import GrayscaleToRgb
from kornia.augmentation import Normalize

from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.networks.nets import UNet
from monai.losses import DiceLoss

from model import *
from data import *

class NeRPDataModule(LightningDataModule):
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
                    Orientationd(keys=('image3d'), axcodes="PRI"),
                    Orientationd(keys=('image3d'), axcodes="ALI"),
                    Orientationd(keys=('image3d'), axcodes="PLI"),
                    Orientationd(keys=["image3d"], axcodes="LAI"),
                    Orientationd(keys=["image3d"], axcodes="RAI"),
                    Orientationd(keys=["image3d"], axcodes="LPI"),
                    Orientationd(keys=["image3d"], axcodes="RPI"),
                    ],
                ),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                #         a_min=-500, #-200, 
                #         a_max=3071, #1500,
                #         b_min=0.0,
                #         b_max=1.0),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                #         a_min=-200, 
                #         a_max=1500,
                #         b_min=0.0,
                #         b_max=1.0),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                #         a_min=-500, #-200, 
                #         a_max=3071, #1500,
                #         b_min=-500,
                #         b_max=3071),
                ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                        a_min=-200, 
                        a_max=1500,
                        b_min=-200,
                        b_max=1500),
                Lambdad(keys=["image3d"], func=HU2Density),
                ScaleIntensityd(keys=["image3d"], 
                        minv=0.0,
                        maxv=1.0),
                RandFlipd(keys=["image3d"], prob=0.5, spatial_axis=0),
                RandZoomd(keys=["image3d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["trilinear"], align_corners=True), 
                RandZoomd(keys=["image2d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["area"]), 
                RandFlipd(keys=["image2d"], prob=0.5, spatial_axis=1),
                RandScaleCropd(keys=["image3d"], 
                               roi_scale=(0.9, 0.9, 0.8), 
                               max_roi_scale=(1.1, 1.1, 1.1), 
                               random_center=True, 
                               random_size=True),
                RandAffined(keys=["image3d"], rotate_range=None, shear_range=None, translate_range=20, scale_range=None),
                Resized(keys=["image3d"], spatial_size=256, size_mode="longest", mode=["trilinear"], align_corners=True),
                Resized(keys=["image2d"], spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image3d", "image2d"], k=256, mode="constant", constant_values=0),
                
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
            num_workers=8, 
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
                OneOf([
                    Orientationd(keys=('image3d'), axcodes="ARI"),
                    Orientationd(keys=('image3d'), axcodes="PRI"),
                    Orientationd(keys=('image3d'), axcodes="ALI"),
                    Orientationd(keys=('image3d'), axcodes="PLI"),
                    Orientationd(keys=["image3d"], axcodes="LAI"),
                    Orientationd(keys=["image3d"], axcodes="RAI"),
                    Orientationd(keys=["image3d"], axcodes="LPI"),
                    Orientationd(keys=["image3d"], axcodes="RPI"),
                    ],
                ), 
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                #         a_min=-500, #-200, 
                #         a_max=3071, #1500,
                #         b_min=0.0,
                #         b_max=1.0),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                #         a_min=-200, 
                #         a_max=1500,
                #         b_min=0.0,
                #         b_max=1.0),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                #         a_min=-500, #-200, 
                #         a_max=3071, #1500,
                #         b_min=-500,
                #         b_max=3071),
                ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                        a_min=-200, 
                        a_max=1500,
                        b_min=-200,
                        b_max=1500),
                Lambdad(keys=["image3d"], func=HU2Density),
                ScaleIntensityd(keys=["image3d"], 
                        minv=0.0,
                        maxv=1.0),
                Resized(keys=["image3d"], spatial_size=256, size_mode="longest", mode=["trilinear"], align_corners=True),
                Resized(keys=["image2d"], spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image3d", "image2d"], k=256, mode="constant", constant_values=0),
            
                ToTensord(keys=["image3d", "image2d"],),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.val_image3d_files, self.val_image2d_files], 
            transform=self.val_transforms,
            length=200,
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

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, source, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if source.shape[1] != 3:
            source = source.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        source = (source-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            source = self.transform(source, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = source
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

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
    "dist": 0.3,
    "elev": 20.,
    "azim": 20.,
    "fov": 20.,
    "aspect_ratio": 0.1
}

class CNNMapper(nn.Module):
    def __init__(self, 
                 input_dim: int = 1,
                 output_dim: int = 1,
    ): 
        super().__init__()
        self.vnet = nn.Sequential(
            CustomUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2, # value and alpha
                channels=(32, 64, 128, 256, 512), #(20, 40, 80, 160, 320), #(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=5,
                up_kernel_size=5,
                norm=Norm.BATCH,
                # act=("elu", {"inplace": True}),
                # dropout=0.5,
            ), 
            nn.Sigmoid()  
        )

    def forward(self, raw_data: torch.Tensor, factor=None, weight=0.0, is_deterministic=True) -> torch.Tensor:
        B, C, D, H, W = raw_data.shape   

        # values = raw_data
        # alphas = self.vnet(raw_data)
        # values = torch.ones_like(raw_data)
        # alphas = self.vnet(raw_data)
        concat = self.vnet( raw_data ) # / 2.0 + 0.5
        values = concat[:,[0],:,:,:] 
        alphas = concat[:,[1],:,:,:]

        # values = self.vnet( raw_data )
        # alphas = torch.ones_like(values)

        features = torch.cat([values, alphas], dim=1) 
        return features

# NeRPLightningModule
class NeRPLightningModule(LightningModule):
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

        self.theta = hparams.theta # use for factor to promote xray effect
        self.alpha = hparams.alpha
        self.kappa = hparams.kappa
        self.gamma = hparams.gamma
        self.delta = hparams.delta
        self.save_hyperparameters()

        self.mapper = CNNMapper()

        self.raysampler = NDCMultinomialRaysampler( #NDCGridRaysampler(
            image_width = self.shape,
            image_height = self.shape,
            n_pts_per_ray = self.shape * 2,
            min_depth = 0.001,
            max_depth = 4.5,
        )

        self.raymarcher = EmissionAbsorptionRaymarcher()

        self.visualizer = VolumeRenderer(
            raysampler = self.raysampler, 
            raymarcher = self.raymarcher,
        )
        
        print("Self Device: ", self.device)

        self.viewer = VolumeModel(
            renderer = self.visualizer,
        )

        self.gen = nn.Sequential(
            self.mapper,
            self.viewer,
        )

        self.discrim = nn.Sequential(
            CustomUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1, # value and alpha
                channels=(64, 128, 256, 512, 1024), #(20, 40, 80, 160, 320), #(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=5,
                up_kernel_size=5,
                norm=Norm.BATCH,
                # act=("elu", {"inplace": True}),
                # dropout=0.5,
            ),
            # nn.Sigmoid(),
        )
        
        # self.gen.apply(_weights_init)
        # self.discrim.apply(_weights_init)
        self.l1loss = nn.L1Loss()
        # self.dcloss = DiceLoss()
        self.ptloss = VGGPerceptualLoss() #PerceptualLoss(net_type='vgg')

    def discrim_step(self, fake_images: torch.Tensor, real_images: torch.Tensor, 
                     batch_idx: int, stage: Optional[str]='train', weight: float=1.0):
        real_logits = self.discrim(real_images) 
        fake_logits = self.discrim(fake_images) 
        # Log to tensorboard
        if batch_idx == 0:
            with torch.no_grad():
                viz = torch.cat([real_logits[:,[0]], fake_logits[:,[0]]], dim=-1)
                grid = torchvision.utils.make_grid(viz, normalize=True, scale_each=True, nrow=2, padding=0)
                tensorboard = self.logger.experiment
                tensorboard.add_image(f'{stage}_real_fake', grid, self.current_epoch) #*self.batch_size + batch_idx)
        # ___1_logits = torch.ones_like(real_images)
        # ___0_logits = torch.zeros_like(fake_images)
        # real_loss = self.dcloss(real_logits, ___1_logits)
        # fake_loss = self.dcloss(fake_logits, ___0_logits)
        # return real_loss + fake_loss 
        # ___1_logits = torch.ones_like(real_images)
        # ___0_logits = torch.zeros_like(fake_images)
        # __01_logits = torch.cat([___0_logits, ___1_logits], dim=1)
        # __10_logits = torch.cat([___1_logits, ___0_logits], dim=1)
        # real_loss = self.dcloss(real_logits, __10_logits)
        # fake_loss = self.dcloss(fake_logits, __01_logits)
        # return real_loss + fake_loss 
        real_loss = F.softplus(-real_logits).mean() 
        fake_loss = F.softplus(+fake_logits).mean()
        return real_loss + fake_loss 

    def gen_step(self, fake_images: torch.Tensor, real_images: torch.Tensor, 
                 batch_idx: int, stage: Optional[str]='train', weight: float=1.0):
        # fake_logits = self.discrim(fake_images) 
        # ___1_logits = torch.ones_like(real_images)
        # fake_loss = self.dcloss(fake_logits, ___1_logits)
        # return fake_loss
        # fake_logits = self.discrim(fake_images)
        # ___0_logits = torch.zeros_like(fake_images)
        # ___1_logits = torch.ones_like(real_images)
        # __10_logits = torch.cat([___1_logits, ___0_logits], dim=1)
        # fake_loss = self.dcloss(fake_logits, __10_logits)
        # return fake_loss
        fake_logits = self.discrim(fake_images) 
        fake_loss = F.softplus(-fake_logits).mean()
        return fake_loss 

    def forward(self, image3d: torch.Tensor, cameras: Type[CamerasBase]=None, 
        factor: float=None, weight: float=None, is_deterministic: bool=False,):
        if cameras is None:
            # cameras = self.detcams
            cameras = init_random_cameras(cam_type=FoVPerspectiveCameras, batch_size=self.batch_size, random=True).to(image3d.device)

        mapped_data = (self.gen[0].forward(image3d, factor=factor, weight=weight, is_deterministic=is_deterministic))
        # Padding mapped
        cloned_data = torch.cat([image3d, mapped_data[:,[1]]], dim=1) # Replace the reconstructed data with original HU value

        # Create a volume with xray effect and pass to the renderer
        volumes = Volumes(
            features = torch.cat([mapped_data[:,[0]]]*3, dim=1),
            densities = mapped_data[:,[1]] * self.theta, #
            voxel_size = 3.3 / self.shape,
        )

        viewed_data = self.gen[1].forward(volumes=volumes, cameras=cameras, norm_type="normalized")
        direct_data = viewed_data.clone()
        return viewed_data, mapped_data, direct_data

    def compute_gradient_penalty(self, fake_samples, real_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discrim(interpolates)
        # fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        fake = torch.ones_like(d_interpolates)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='train'):
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        # generate images
        with torch.no_grad():
            self.varcams = init_random_cameras(cam_type=FoVPerspectiveCameras, 
                                               batch_size=self.batch_size, 
                                               cam_mu=cam_mu,
                                               cam_bw=cam_bw,
                                               cam_ft=None, 
                                               random=True).to(image3d.device)
        
        # train generator
        if optimizer_idx == 0:
            viewed_, mapped_, direct = self.forward(image3d, self.varcams, factor=None, weight=0.0, is_deterministic=False)
            g_loss = self.gen_step(fake_images=viewed_, real_images=image2d, batch_idx=batch_idx, stage=stage)
            self.log(f'{stage}_g_loss', g_loss, on_step=True, prog_bar=True, logger=True)
            
            # Log to tensorboard, 
            if batch_idx == 0:
                with torch.no_grad():
                    viz = torch.cat([image3d[...,128], 
                                        mapped_[:,[0],...,128], 
                                        mapped_[:,[1],...,128], 
                                        viewed_, image2d], dim=-1)
                    grid = torchvision.utils.make_grid(viz, normalize=False, scale_each=False, nrow=1, padding=0)
                    tensorboard = self.logger.experiment
                    tensorboard.add_image(f'{stage}_samples', grid, self.current_epoch)#*self.batch_size + batch_idx)

                    plot_2d_or_3d_image(data=torch.cat([image3d, 
                                                        mapped_[:,[0]],
                                                        mapped_[:,[1]]], dim=-2), 
                                                        tag=f'{stage}_gif', writer=tensorboard, step=self.current_epoch, frame_dim=-1)
                    # plot_2d_or_3d_image(data=image3d, tag=f'{stage}_image3d', writer=tensorboard, step=self.current_epoch, frame_dim=-1)
                    # plot_2d_or_3d_image(data=mapped_[:,[0]], tag=f'{stage}_mapped_0', writer=tensorboard, step=self.current_epoch, frame_dim=-1)
                    # plot_2d_or_3d_image(data=mapped_[:,[1]], tag=f'{stage}_mapped_1', writer=tensorboard, step=self.current_epoch, frame_dim=-1)

            if self.gamma > 0:
                r_loss = self.l1loss(mapped_[:,[0]], image3d) #+ self.l1loss(mapped_[:,[1]], torch.ones_like(image3d))
                self.log(f'{stage}_r_loss', r_loss, on_step=True, prog_bar=True, logger=True)
                g_loss += self.gamma * r_loss
            if self.kappa > 0:
                p_loss = self.ptloss(viewed_, image2d)
                self.log(f'{stage}_p_loss', p_loss, on_step=True, prog_bar=True, logger=True)
                g_loss += self.kappa * p_loss

            info = {'loss': g_loss}
            return info

        # train discriminator
        elif optimizer_idx == 1:
            viewed_, mapped_, direct = self.forward(image3d, self.varcams, factor=None, weight=0.0, is_deterministic=False)
            d_loss = self.discrim_step(fake_images=viewed_.detach(), real_images=image2d, batch_idx=batch_idx, stage=stage)
            self.log(f'{stage}_d_loss', d_loss, on_step=True, prog_bar=True, logger=True)

            if self.delta > 0:
                d_grad = self.compute_gradient_penalty(fake_samples=viewed_, real_samples=image2d)
                self.log(f'{stage}_d_grad', d_grad, on_step=True, prog_bar=True, logger=True)
                d_loss += self.delta * d_grad

            info = {'loss': d_loss}
            return info

    def evaluation_step(self, batch, batch_idx, stage: Optional[str]='evaluation'):   
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        with torch.no_grad():
            self.detcams = init_random_cameras(cam_type=FoVPerspectiveCameras, 
                                               batch_size=self.batch_size, 
                                               cam_mu=cam_mu,
                                               cam_bw=cam_bw,
                                               cam_ft=None, 
                                               random=False).to(image3d.device)
        viewed_, mapped_, direct = self.forward(image3d, self.detcams, factor=None, weight=0.0, is_deterministic=True)
        
        # Log to tensorboard, 
        if batch_idx == 0:
            with torch.no_grad():
                viz = torch.cat([image3d[...,128], 
                                 mapped_[:,[0],...,128], 
                                 mapped_[:,[1],...,128], 
                                 viewed_, image2d], dim=-1)
                grid = torchvision.utils.make_grid(viz, normalize=False, scale_each=False, nrow=1, padding=0)
                tensorboard = self.logger.experiment
                tensorboard.add_image(f'{stage}_samples', grid, self.current_epoch)#*self.batch_size + batch_idx)
                plot_2d_or_3d_image(data=torch.cat([image3d, 
                                                    mapped_[:,[0]],
                                                    mapped_[:,[1]]], dim=-2), 
                                                    tag=f'{stage}_gif', writer=tensorboard, step=self.current_epoch, frame_dim=-1)
                # plot_2d_or_3d_image(data=image3d, tag=f'{stage}_image3d', writer=tensorboard, step=self.current_epoch, frame_dim=-1)
                # plot_2d_or_3d_image(data=mapped_[:,[0]], tag=f'{stage}_mapped_0', writer=tensorboard, step=self.current_epoch, frame_dim=-1)
                # plot_2d_or_3d_image(data=mapped_[:,[1]], tag=f'{stage}_mapped_1', writer=tensorboard, step=self.current_epoch, frame_dim=-1)

        g_loss = self.gen_step(fake_images=viewed_, real_images=image2d, batch_idx=batch_idx, stage=stage)
        d_loss = self.discrim_step(fake_images=viewed_, real_images=image2d, batch_idx=batch_idx, stage=stage)

        info = {'g_loss': g_loss, 'd_loss': d_loss}
        if self.gamma > 0:
            r_loss =  self.l1loss(mapped_[:,[0]], image3d)
            info['r_loss'] = r_loss
        if self.kappa > 0:
            p_loss = self.ptloss(viewed_, image2d)
            info['p_loss'] = p_loss
            
        return info

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, stage='validation')

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, stage='test')

    def evaluation_epoch_end(self, outputs, stage: Optional[str]='evaluation'):
        g_loss = torch.stack([x[f'g_loss'] for x in outputs]).mean()
        d_loss = torch.stack([x[f'd_loss'] for x in outputs]).mean()
        self.log(f'{stage}_g_loss_epoch', g_loss, on_step=False, prog_bar=True, logger=True)
        self.log(f'{stage}_d_loss_epoch', d_loss, on_step=False, prog_bar=True, logger=True)
        if self.gamma > 0:
            r_loss = torch.stack([x[f'r_loss'] for x in outputs]).mean()
            self.log(f'{stage}_r_loss_epoch', r_loss, on_step=False, prog_bar=True, logger=True)
        if self.kappa > 0:
            p_loss = torch.stack([x[f'p_loss'] for x in outputs]).mean()
            self.log(f'{stage}_p_loss_epoch', p_loss, on_step=False, prog_bar=True, logger=True)

    def train_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='validation')
    
    def test_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='test')

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=1e0*(self.lr or self.learning_rate))
        opt_d = torch.optim.Adam(self.discrim.parameters(), lr=1e0*(self.lr or self.learning_rate))

        return opt_g, opt_d

if __name__ == "__main__":
    parser = ArgumentParser()
    # System arguments: --gpus is default argument for cli
    # parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--conda_env", type=str, default="NeRP")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")
    
    # Model arguments
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--shape", type=int, default=256, help="spatial size of the tensor")
    # parser.add_argument("--reset", type=int, default=0, help="reset the training")
    parser.add_argument("--epochs", type=int, default=501, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--wd", type=float, default=1e-6, help="adam: weight decay")
    parser.add_argument("--eps", type=float, default=1e-8, help="adam: epsilon")
    parser.add_argument("--b1", type=float, default=0.0, help="adam: 1st order momentum")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: 2nd order momentum")
    
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    
    parser.add_argument("--theta", type=float, default=5e-2, help="density weight")
    parser.add_argument("--gamma", type=float, default=1e+1, help="luminance weight")
    parser.add_argument("--delta", type=float, default=1e+1, help="L1 compared to 3D")
    parser.add_argument("--alpha", type=float, default=1e-4, help="total variation weight")
    parser.add_argument("--kappa", type=float, default=0e+0, help="perceptual compared to DRR")

    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")

    parser = Trainer.add_argparse_args(parser)
    
    # Collect the hyper parameters
    hparams = parser.parse_args()

    # Seed the application
    seed_everything(2222)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename='{epoch:02d}-{validation_g_loss_epoch:.2f}-{validation_d_loss_epoch:.2f}',
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
        # resume_from_checkpoint = hparams.ckpt, #"logs/default/version_0/epoch=50.ckpt",
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback, 
            # tensorboard_callback
        ],
        # accumulate_grad_batches=10,
        # precision=16,
        # stochastic_weight_avg=True,
        # auto_scale_batch_size=True, 
        # gradient_clip_val=0.001, 
        # gradient_clip_algorithm='value', #'norm', #'value'
    )

    # Create data module
    train_image3d_folders = [
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),

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
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
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
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = NeRPDataModule(
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

    model = NeRPLightningModule(
        hparams = hparams
    )
    model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model


    trainer.fit(
        model, 
        datamodule,
    )

    # test

    # serve
