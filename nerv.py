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

from model import *
from data import *

class NeRVDataModule(LightningDataModule):
# class CustomDataModule(LightningDataModule):
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
                               max_roi_scale=(1.0, 1.0, 0.8), 
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

        
    def forward(self, image3d: torch.Tensor, cameras: Type[CamerasBase]=None, 
        factor: float=None, weight: float=None, is_deterministic: bool=False,):
        
        pass
    

    def training_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='train'):
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        
        pass

    def evaluation_step(self, batch, batch_idx, stage: Optional[str]='evaluation'):   
        image3d = batch["image3d"]
        image2d = batch["image2d"]
        
        pass

    def validation_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, stage='validation')

    def test_step(self, batch, batch_idx):
        return self.evaluation_step(batch, batch_idx, stage='test')

    def evaluation_epoch_end(self, outputs, stage: Optional[str]='evaluation'):
        pass

    def train_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='validation')
    
    def test_epoch_end(self, outputs):
        return self.evaluation_epoch_end(outputs, stage='test')

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.parameters(), lr=1e0*(self.lr or self.learning_rate))
        return opt

def test_random_uniform_cameras(hparams, datamodule):
    # Set up the environment
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
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
    cam_ft = torch.Tensor(hparams.batch_size, 6).uniform_(-1, 1)
    print(cam_ft)
    cameras = init_random_cameras(cam_type=FoVPerspectiveCameras, 
                                  batch_size=hparams.batch_size, 
                                  cam_mu=cam_mu,
                                  cam_bw=cam_bw,
                                  cam_ft=None,
                                  random=True)
    cameras = cameras.to(device)

    #
    # Smoke test random cameras
    #
    screens = volume_model(cameras=cameras, volumes=volumes)

    for idx in range(hparams.batch_size):
        torchvision.utils.save_image(screens[idx,0,:,:].detach().cpu(), 
            f'test_camera_{idx}_features_{cam_ft[idx, 0]:.4f}_{cam_ft[idx, 1]:.4f}_{cam_ft[idx, 2]:.4f}_{cam_ft[idx, 3]:.4f}_{cam_ft[idx, 4]:.4f}_{cam_ft[idx, 5]:.4f}.png')

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
    test_random_uniform_cameras(hparams, datamodule)

    #############################################

    # model = NeRPLightningModule(
    #     hparams = hparams
    # )
    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model


    # trainer.fit(
    #     model, 
    #     datamodule,
    # )

    # test

    # serve
