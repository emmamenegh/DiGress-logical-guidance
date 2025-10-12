""" 
Supports guidance towards the satisfaction of the Lipinski rules with a separate classifier for each property.
"""
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import wandb
import hydra
import os
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings


import src.utils as utils
from src.guidance.guidance_diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.datasets import guacamol_dataset
from src.metrics.molecular_metrics import SamplingMolecularMetrics, TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.utils import update_config_with_new_keys
from src.guidance.guacamol_classifier_discrete import GuacamolClassifierDiscrete


warnings.filterwarnings("ignore", category=PossibleUserWarning)



# --- Monkey-patch legacy module paths for checkpoint compatibility ---
patches = {
    'datasets': 'src.datasets',
    'analysis': 'src.analysis',
    'models': 'src.models',
    'metrics': 'src.metrics',
    'diffusion': 'src.diffusion',
    'utils': 'src.utils'
}

for legacy_name, actual_import in patches.items():
    __import__(actual_import)
    sys.modules[legacy_name] = sys.modules[actual_import]



# Mapping from target names to indices
TARGET_INDEX_MAP = {
    "logP": [0],
    "molW": [1],
    "HBD": [2],
    "HBA": [3],
    "all": [0, 1, 2, 3],      # all targets
    "COMP": [4],
    "LRO5": [5]
}

INDEX_TARGET_MAP = {
    0: "logP",
    1: "molW",
    2: "HBD",
    3: "HBA", 
    4: "COMP",
    5: "LRO5"
}



def get_resume(cfg):
    saved_cfg = cfg.copy()

    name = saved_cfg.general.name + '_resume'
    wb = saved_cfg.general.wandb
    resume = saved_cfg.general.test_only
    samples_to_save = saved_cfg.general.samples_to_save # graphs
    chains_to_save = saved_cfg.general.chains_to_save # chains
    number_chain_steps = saved_cfg.general.number_chain_steps # chain steps to visualise
    final_model_samples_to_save = saved_cfg.general.final_model_samples_to_save # SMILES to save
    batch_size = saved_cfg.train.batch_size
    save_model = saved_cfg.train.save_model
    
    #  manually load the checkpoint dictionary
    checkpoint = torch.load(resume, map_location='cpu')
    # extract the saved config
    old_cfg = checkpoint['hyper_parameters']['cfg']


    # inject new keys before model is constructed
    OmegaConf.set_struct(old_cfg, False)
    # guidance did not exist in the old configuration
    if "guidance" not in old_cfg:
        old_cfg.guidance = {}
        OmegaConf.set_struct(old_cfg.guidance, False)

    old_cfg.general.guidance_target = saved_cfg.general.guidance_target
    old_cfg.general.satisfaction_rule = saved_cfg.general.satisfaction_rule
    old_cfg.general.trained_classifier_dir = saved_cfg.general.trained_classifier_dir
    old_cfg.guidance.use_guidance = saved_cfg.guidance.use_guidance
    old_cfg.guidance.lambda_guidance = saved_cfg.guidance.lambda_guidance

    old_cfg.general.name = name
    old_cfg.general.test_only = resume
    old_cfg.general.wandb = wb
    old_cfg.general.samples_to_save = samples_to_save
    old_cfg.general.final_model_samples_to_save = final_model_samples_to_save
    old_cfg.general.chains_to_save = chains_to_save
    old_cfg.general.number_chain_steps = number_chain_steps
    old_cfg.train.batch_size = batch_size
    old_cfg.train.save_model = save_model

    print("\n--- Loaded config from checkpoint ---")
    for k, v in old_cfg.items():
        print(f"{k}: {v}")
    print("--- End of loaded config ---\n")
    
    return old_cfg


@rank_zero_only
def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': 'guidance', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):

    cfg = get_resume(cfg)

    dataset_config = cfg["dataset"]
    assert dataset_config.name == "guacamol", "Only Guacamol dataset is supported for now"
    datamodule = guacamol_dataset.GuacamolDataModule(cfg, classifier=False) # do not pass y to the diffusion model
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
                    'extra_features': extra_features, 'domain_features': domain_features,
                    'satisfaction_rule': cfg.general.satisfaction_rule}

    guidance_sampling_model = DiscreteDenoisingDiffusion.load_from_checkpoint(cfg.general.test_only, cfg=cfg, **model_kwargs)

    # Move to correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guidance_sampling_model.to(device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")

    # Load pretrained classifiers 
    guidance_models = {}
    if cfg.general.trained_classifier_dir is not None and cfg.general.guidance_target is not None:
        target_index = TARGET_INDEX_MAP[cfg.general.guidance_target]
        for idx, target in enumerate(target_index):
            model_path = os.path.join(cfg.general.trained_classifier_dir, f'{INDEX_TARGET_MAP[target]}.ckpt')

            # Manually load checkpoint to extract config
            checkpoint = torch.load(model_path, map_location='cpu')
            loaded_cfg = checkpoint['hyper_parameters']['cfg']

            # Print loaded config for debugging
            print(f"--- Loaded config for classifier {INDEX_TARGET_MAP[target]}: ---")
            for k, v in loaded_cfg.items():
                print(f"  {k}: {v}")
            print("--- End loaded config ---")

            guidance_models[idx] = GuacamolClassifierDiscrete.load_from_checkpoint(model_path,
                                                                                   train_metrics=train_metrics,
                                                                                   sampling_metrics=sampling_metrics)
            guidance_models[idx].to(device)
    else:
        print("Not enough information for guidance! Need classifiers' directory and target.")

    # Attach classifiers
    guidance_sampling_model.guidance_models = guidance_models

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    # Guided sampling
    guidance_sampling_model.generate_samples()



if __name__ == '__main__':
    main()
