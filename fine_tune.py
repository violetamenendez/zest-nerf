from pathlib import Path
import logging, coloredlogs
import random

# torch
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from train import MVSNeRFSystem
from opt import config_parser

# Enable logging
logging.captureWarnings(True)
coloredlogs.install(
   level=logging.WARNING,
   fmt="%(asctime)s %(name)s:%(module)s.%(funcName)s[%(lineno)d] %(levelname)s %(message)s",
   datefmt="%F %T"
)

def finetune():
    torch.set_default_dtype(torch.float32)
    hparams = config_parser()
    hparams.save_dir = Path(hparams.save_dir)
    print(hparams)
    # Override training parameters
    kwargs = {}
    kwargs['crossval'] = hparams.crossval
    kwargs['frame_jump'] = hparams.frame_jump
    # kwargs['configdir'] = hparams.configdir
    kwargs['datadir'] = hparams.datadir
    kwargs['expname'] = hparams.expname
    kwargs['save_dir'] = hparams.save_dir
    kwargs['finetune_scene'] = hparams.finetune_scene
    kwargs['batch_size'] = hparams.batch_size
    kwargs['num_extra_samples'] = 0
    # kwargs['dataset_name'] = hparams.dataset_name
    # kwargs['finetune_scene'] = hparams.finetune_scene
    # kwargs['batch_size'] = hparams.batch_size
    # kwargs['patch_size'] = hparams.patch_size
    # kwargs['chunk'] = hparams.chunk
    # kwargs['pts_embedder'] = hparams.pts_embedder
    # kwargs['depth_path'] = hparams.depth_path
    # kwargs['use_closest_views'] = hparams.use_closest_views

    save_dir = hparams.save_dir / hparams.expname
    save_dir_ckpts = save_dir / 'ckpts'
    save_dir_ckpts.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir_ckpts,
                                          filename='{epoch:02d}-{step}-{val_loss:.3f}',
                                          monitor='val_loss',
                                          mode='min',
                                          verbose=True,
                                          save_top_k=5,
                                          save_last=True)

    # Resume checkpoints if file exists
    # otherwise load specified checkpoints for fine-tuning
    resume_ckpt = None
    if Path(save_dir_ckpts / 'last.ckpt').exists():
        resume_ckpt = save_dir_ckpts / 'last.ckpt'
        print(f"Resuming found checkpoints from: {resume_ckpt}")
        system = MVSNeRFSystem(hparams, pts_embedder=hparams.pts_embedder, use_mvs=hparams.use_mvs, dir_embedder=hparams.dir_embedder)
    elif hparams.ckpt:
        print(f"Loading checkpoints from: {hparams.ckpt}")
        system = MVSNeRFSystem.load_from_checkpoint(hparams.ckpt, strict=False, **kwargs)

    print(system.hparams)

    # Resume logging?
    if Path(save_dir / 'wandb_id.txt').exists():
        with open(Path(save_dir / 'wandb_id.txt')) as f:
            wandb_id = int(f.readline())
            print(f"Resuming W&B job with id {wandb_id}")
    else:
        with open(Path(save_dir / 'wandb_id.txt'), 'w') as f:
            wandb_id = str(random.randint(0, 1000000))
            f.write(wandb_id)
            print(f"Starting W&B job with id {wandb_id}")

    logger = loggers.WandbLogger(
        project="SVS",
        save_dir=hparams.save_dir,
        name=hparams.expname,
        offline=False
    )

    hparams.num_gpus = 1
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      logger=logger,
                      callbacks=checkpoint_callback,
                      gpus=hparams.num_gpus,
                      num_sanity_val_steps=0,
                      check_val_every_n_epoch = max(hparams.num_epochs//system.hparams.N_vis,1),
                      benchmark=True,
                      precision=system.hparams.precision,
                      accumulate_grad_batches=hparams.acc_grad,
                      gradient_clip_val=1,
                      detect_anomaly=True)

    trainer.fit(system, ckpt_path=resume_ckpt)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    try:
        finetune()
    finally:
        print(torch.cuda.memory_summary(abbreviated=True))
