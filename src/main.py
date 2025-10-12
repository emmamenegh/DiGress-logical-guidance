""" 
This pipeline was simplified and tailored for this project's use case. In particular: 
    - It implements a more flexible configuration manager;
    - It only supports the GuacaMol dataset;
    - It only supports the discrete diffusion model;
    - It does not support resume training, it only supports testing and debugging.
"""
import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig, OmegaConf
import pprint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

from metrics.molecular_metrics import SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from analysis.visualization import MolecularVisualization

from datasets import guacamol_dataset


warnings.filterwarnings("ignore", category=PossibleUserWarning)



def get_resume(cfg):
    """ 
    This is the only place where the configuration can be changed.
    We must load the configuration before the dataset and model are loaded because we need to inject
    some keys that were added for the guidance experiments.
    Do not change model architecture parameters. 
    """
    saved_cfg = cfg.copy()

    name = saved_cfg.general.name + '_resume'
    wb = saved_cfg.general.wandb
    resume = saved_cfg.general.test_only
    sample_every_val = saved_cfg.general.sample_every_val
    samples_to_generate = saved_cfg.general.samples_to_generate
    samples_to_save = saved_cfg.general.samples_to_save
    chains_to_save = saved_cfg.general.chains_to_save
    check_val_every_n_epochs = saved_cfg.general.check_val_every_n_epochs
    final_model_samples_to_generate = saved_cfg.general.final_model_samples_to_generate
    final_model_samples_to_save = saved_cfg.general.final_model_samples_to_save
    final_model_chains_to_save = saved_cfg.general.final_model_chains_to_save
    save_model = saved_cfg.train.save_model
    n_epochs = saved_cfg.train.n_epochs
    batch_size = saved_cfg.train.batch_size
    
    #  manually load the checkpoint dictionary
    checkpoint = torch.load(resume, map_location='cpu')
    # extract the saved config
    old_cfg = checkpoint['hyper_parameters']['cfg']

    # inject new keys before model is constructed
    OmegaConf.set_struct(old_cfg, False)
    old_cfg.general.guidance_target = saved_cfg.general.guidance_target
    # old_cfg.general.satisfaction_rule = saved_cfg.general.satisfaction_rule

    old_cfg.general.name = name
    old_cfg.general.test_only = resume
    old_cfg.general.wandb = wb
    old_cfg.general.sample_every_val = sample_every_val
    old_cfg.general.samples_to_generate = samples_to_generate
    old_cfg.general.samples_to_save = samples_to_save
    old_cfg.general.chains_to_save = chains_to_save
    old_cfg.general.check_val_every_n_epochs = check_val_every_n_epochs
    old_cfg.general.final_model_samples_to_generate = final_model_samples_to_generate
    old_cfg.general.final_model_samples_to_save = final_model_samples_to_save
    old_cfg.general.final_model_chains_to_save = final_model_chains_to_save
    old_cfg.train.save_model = save_model
    old_cfg.train.n_epochs = n_epochs
    old_cfg.train.batch_size = batch_size

    print("\n--- Loaded config from checkpoint ---")
    for k, v in old_cfg.items():
        print(f"{k}: {v}")
    print("--- End of loaded config ---\n")
    
    return old_cfg


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    cfg = get_resume(cfg)

    # Load the dataset
    dataset_config = cfg["dataset"]
    assert dataset_config.name == "guacamol", "Only Guacamol dataset is supported for now"

    datamodule = guacamol_dataset.GuacamolDataModule(cfg)
    dataset_infos = guacamol_dataset.Guacamolinfos(datamodule=datamodule, cfg=cfg)
    datamodule.prepare_data()
    train_smiles = guacamol_dataset.get_train_smiles(cfg, datamodule, dataset_infos)

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    # Load the model
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(cfg.general.test_only, cfg=cfg, **model_kwargs)


    utils.create_folders(cfg)
    
    # Test or debug the model
    callbacks = []
    if cfg.general.name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      # strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if cfg.general.gpus > 0 and torch.cuda.is_available() else 'cpu',
                      devices=1,
                      max_epochs=cfg.train.n_epochs,
                      limit_test_batches=1 if cfg.general.name == "test_resume" else None,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=cfg.train.progress_bar,
                      callbacks=callbacks,
                      log_every_n_steps=50 if cfg.general.name != 'debug' else 1,
                      logger = [])

    trainer.test(model, datamodule=datamodule, ckpt_path=None) # no need to re-load the checkpoints
    # if cfg.general.evaluate_all_checkpoints:
    #     directory = pathlib.Path(cfg.general.test_only).parents[0]
    #     print("Directory:", directory)
    #     files_list = os.listdir(directory)
    #     for file in files_list:
    #         if '.ckpt' in file:
    #             ckpt_path = os.path.join(directory, file)
    #             if ckpt_path == cfg.general.test_only:
    #                 continue
    #             print("Loading checkpoint", ckpt_path)
    #             trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
