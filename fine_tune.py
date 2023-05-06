from pathlib import Path
import logging, coloredlogs

# torch
import torch
from pytorch_lightning import Trainer, loggers

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

    system = MVSNeRFSystem.load_from_checkpoint(hparams.ckpt, strict=False, **kwargs)
    print(system.hparams)

    logger = loggers.WandbLogger(
        project="SVS",
        save_dir=hparams.save_dir,
        name=hparams.expname,
        offline=False
    )

    hparams.num_gpus = 1
    trainer = Trainer(max_epochs=400,
                      logger=logger,
                      gpus=hparams.num_gpus,
                      num_sanity_val_steps=0,
                      check_val_every_n_epoch = max(system.hparams.num_epochs//system.hparams.N_vis,1),
                      benchmark=True,
                      precision=system.hparams.precision)

    trainer.fit(system)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    try:
        finetune()
    finally:
        print(torch.cuda.memory_summary(abbreviated=True))
