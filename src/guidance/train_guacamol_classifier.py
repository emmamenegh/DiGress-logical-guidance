# Rdkit import should be first, do not move it
from rdkit import Chem

import os
import torch
import torch.multiprocessing as mp
mp.set_start_method("fork", force=True)

import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import warnings

import src.utils as utils
import src.datasets.guacamol_dataset as guacamol_dataset
from src.metrics.molecular_metrics import SamplingMolecularMetrics, TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.guidance.guacamol_classifier_discrete import GuacamolClassifierDiscrete


warnings.filterwarnings("ignore", category=PossibleUserWarning)


# for logging messages to the output files
@rank_zero_only
def safe_print(*args, **kwargs):
    print(*args, **kwargs)

@rank_zero_only
def setup_wandb(run_name, cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': run_name, 'project': 'graph_ddm_classifier', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg

def get_checkpoint_callback(run_name, cfg):
    return ModelCheckpoint(
        dirpath=f"checkpoints/{run_name}",
        filename='{epoch}',
        monitor='val/bce',
        save_last=True,
        save_top_k=-1,
        mode='min',
        every_n_epochs=1
    )

@rank_zero_only
def create_run_folders(run_name):
    """Create output folders for a specific training run without modifying global utils."""
    for folder in ['checkpoints', 'graphs', 'chains']:
        os.makedirs(folder, exist_ok=True)
        os.makedirs(f"{folder}/{run_name}", exist_ok=True)


@hydra.main(version_base='1.1', config_path='../../configs', config_name='config')
def main(cfg: DictConfig):

    # Print configuration
    print("\n--- Config ---")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("--- End of config ---\n")

    # Setup dataset
    dataset_config = cfg["dataset"]
    safe_print(dataset_config)
    assert dataset_config["name"] == "guacamol", "Only GuacaMol dataset is supported for now"
    assert cfg.model.type == 'discrete'
    datamodule = guacamol_dataset.GuacamolDataModule(cfg, classifier=True)
    dataset_infos = guacamol_dataset.Guacamolinfos(datamodule=datamodule, cfg=cfg)
    datamodule.prepare_data()
    train_smiles = None

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)
    dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': 1} # binary classification with BCEWithLogitsLoss

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    
    visualization_tools = MolecularVisualization(cfg.general.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    run_name = f"{cfg.general.name}_target{cfg.general.guidance_target}"
    create_run_folders(run_name)
    setup_wandb(run_name, cfg)
    
    # Initialise model
    model = GuacamolClassifierDiscrete(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = get_checkpoint_callback(run_name, cfg)
        safe_print("Checkpoints will be logged to", checkpoint_callback.dirpath)
        callbacks.append(checkpoint_callback)

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                    accelerator='gpu' if cfg.general.gpus > 0 and torch.cuda.is_available() else 'cpu',
                    devices=1 if cfg.general.gpus > 0 and torch.cuda.is_available() else None,
                    num_nodes=4, # added
                    strategy=DDPStrategy(process_group_backend="nccl"), # added
                    limit_train_batches=20 if name == "debug" else 1.0,     # TODO: remove
                    limit_val_batches=20 if name == 'test' else None,
                    limit_test_batches=20 if name == 'test' else None,
                    max_epochs=cfg.train.n_epochs,
                    check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                    fast_dev_run=cfg.general.name == 'debug',
                    enable_progress_bar=cfg.train.progress_bar,
                    callbacks=callbacks)

    safe_print(f"[INFO] Detected {torch.cuda.device_count()} CUDA device(s)")
    safe_print(f"[INFO] Trainer is using devices: {trainer.num_devices} ({trainer.strategy})")

    # Train model
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)


if __name__ == '__main__':
    main()
