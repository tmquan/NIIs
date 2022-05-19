import os

from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from kornia.color import GrayscaleToRgb
from kornia.augmentation import Normalize

from monai.visualize.img2tensorboard import plot_2d_or_3d_image

from model import *
from data import *

class PerceptualLoss(LearnedPerceptualImagePatchSimilarity):
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) is used to judge the perceptual similarity between
    two images. LPIPS essentially computes the similarity between the activations of two image patches for some
    pre-defined network. This measure has been shown to match human perception well. A low LPIPS score means that
    image patches are perceptually similar.
    """
    def __init__(self, 
        net_type: str = "vgg",
        *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_name = 'ptloss' 
        self.net_type = net_type
        print(f"Using {self.net_type}")
        if self.net_type == "vgg":
            self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.gray2rgb = GrayscaleToRgb()

    def forward(self, source, target):
        if source.shape[1] == 1:
            source = self.gray2rgb(source)
        if target.shape[1] == 1:
            target = self.gray2rgb(target)
        # if self.net_type == "vgg":
        #     source = self.normalize(source)
        #     target = self.normalize(target)
        # source range from 0~1, change to -1~1
        source = source * 2 - 1
        target = target * 2 - 1

        return super().forward(source, target)

def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.04)
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.04)
        torch.nn.init.constant_(m.bias, 0)

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

        self.mapper = CNNMapper(
            input_dim = 1,
            output_dim = 2,
        )

        self.raysampler = NDCMultinomialRaysampler( #NDCGridRaysampler(
            image_width = self.shape,
            image_height = self.shape,
            n_pts_per_ray = self.shape * 2,
            min_depth = 0.001,
            max_depth = 4.0,
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
                # act=("LeakyReLU", {"negative_slope": 0.2, "inplace": True}),
                act=("elu", {"inplace": True}),
                norm=Norm.BATCH,
                # dropout=0.5,
            ),
        )
        
        self.gen.apply(_weights_init)
        self.discrim.apply(_weights_init)
        self.l1loss = nn.L1Loss()
        self.ptloss = PerceptualLoss(net_type='vgg')

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

        real_loss = F.softplus(-real_logits).mean() 
        fake_loss = F.softplus(+fake_logits).mean()

        return real_loss + fake_loss 

    def gen_step(self, fake_images: torch.Tensor, real_images: torch.Tensor, 
                 batch_idx: int, stage: Optional[str]='train', weight: float=1.0):
        
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
            features = torch.cat([image3d]*3, dim=1),
            densities = mapped_data[:,[1]] * self.theta, #image3d / 512., 
            #torch.ones_like(image3d) / 512., # modify here
            voxel_size = 3.2 / self.shape,
        )

        viewed_data = self.gen[1].forward(volumes=volumes, cameras=cameras)
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
            self.varcams = init_random_cameras(cam_type=FoVPerspectiveCameras, batch_size=self.batch_size, random=True).to(image3d.device)
        
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

    def evaluation_step(self, batch, batch_idx, stage: Optional[str]='evaluation'):   
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        with torch.no_grad():
            self.detcams = init_random_cameras(cam_type=FoVPerspectiveCameras, batch_size=self.batch_size, random=False).to(image3d.device)
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
        opt_g = torch.optim.RAdam(self.gen.parameters(), lr=1e0*(self.lr or self.learning_rate))
        opt_d = torch.optim.RAdam(self.discrim.parameters(), lr=1e0*(self.lr or self.learning_rate))

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
