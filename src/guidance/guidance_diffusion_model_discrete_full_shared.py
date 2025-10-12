import torch
import pytorch_lightning as pl
import time
import wandb
import os
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
import networkx as nx
from src.metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.metrics.train_metrics import TrainLossDiscrete
from src.datasets.guacamol_dataset_full_shared import LipinskiBinaryExtractor
import src.utils as utils

# packages for conditional generation with guidance
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit import Chem
import math
from src.analysis.rdkit_functions import build_molecule, mol2smiles, build_molecule_with_partial_charges
import pickle
import pandas as pd


# Map task names to indices
SINGLE_TASK_INDEX_MAP = {"logP": 0, "molW": 1, "HBD": 2, "HBA": 3}

def single_property_satisfaction(logit):
    return F.sigmoid(logit)

def composite_property_satisfaction(logits):
    return torch.prod(torch.sigmoid(logits), dim=1)

def LRo5_satisfaction(logits):
    probs = torch.sigmoid(logits)  # [batch_size, 4]

    # Compute all combinations efficiently
    all_four = torch.prod(probs, dim=1)  # [batch_size]

    triple_1 = probs[:, 0] * probs[:, 1] * probs[:, 2] * (1 - probs[:, 3])
    triple_2 = probs[:, 0] * probs[:, 1] * probs[:, 3] * (1 - probs[:, 2])
    triple_3 = probs[:, 0] * probs[:, 2] * probs[:, 3] * (1 - probs[:, 1])
    triple_4 = probs[:, 1] * probs[:, 2] * probs[:, 3] * (1 - probs[:, 0])

    total = all_four + triple_1 + triple_2 + triple_3 + triple_4  # [batch_size]

    return total


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, guidance_model=None, satisfaction_rule=None):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.num_classes = dataset_infos.num_classes
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)
        # Marginal noise schedule
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                          y_classes=self.ydim_output)
        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        # specific properties to generate molecules
        self.cond_val = nn.BCELoss(reduction='none') # input: probability added: reduction='none'
        self.num_total = 0
        self.num_valid_molecules = 0
        self.min_frags, self.max_frags, self.avg_frags = float('inf'), 0, 1
        self.num_logP, self.num_molW, self.num_HBD, self.num_HBA, self.num_comp, self.num_LRo5 = 0, 0, 0, 0, 0, 0

        self.guidance_target = cfg.general.guidance_target
        self.guidance_models = guidance_model if isinstance(guidance_model, dict) else {0: guidance_model}
        self.satisfaction_rule = satisfaction_rule

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)
    
    @torch.no_grad()
    def generate_samples(self):

        total_samples = self.cfg.general.final_model_samples_to_save
        bs = 2 * self.cfg.train.batch_size
        all_samples = 0
        for batch_id, batch_start in enumerate(range(0, total_samples, bs), start=1):
            current_batch_size = min(bs, total_samples - batch_start)
            
            print(f"Sampling of batch {batch_id} started")
            start = time.time()
            samples = self.sample_batch(
                batch_id=batch_id,
                batch_size=current_batch_size,
                num_nodes=None,
                save_final=self.cfg.general.samples_to_save,
                keep_chain=self.cfg.general.chains_to_save,
                number_chain_steps=self.number_chain_steps
            )
            print(f'Sampling of batch {batch_id} took {time.time() - start:.2f} seconds\n')
            all_samples += len(samples)
            print(f'Total number of samples after batch {batch_id}: {all_samples}')
            
            # Save current batch
            self.save_cond_samples(samples, file_path=os.path.join(os.getcwd(), f'cond_smiles{batch_id}.txt'))
            # Calculate benchmark and Lipinski metrics
            self.cond_sample_metric(samples)


    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y # y is an empty tensor at this point (bs,0)

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # print("Examples of generated graphs:")
        # for i in range(min(5, X.shape[0])):
        #     print("E: ", E[i])
        #     print("X: ", X[i])

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            print('Visualizing chains starts!')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            print('\nVisualizing chains Ends!')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)

        return molecule_list


    def cond_sample_metric(self, samples):
        
        num_components = []
        
        # Lipinski's properties satisfaction is logged here for monitoring (can be compued from final SMILES too)
        lipinski_extractor = LipinskiBinaryExtractor()

        for sample in samples:
            # follows relaxed validity principles
            mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)
            smiles = mol2smiles(mol)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except:
                num_components.append(0)

            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    self.num_valid_molecules += 1
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    continue
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    continue

                try:
                    L_properties = lipinski_extractor(largest_mol)
                    self.num_logP += L_properties[0]
                    self.num_molW += L_properties[1]
                    self.num_HBD += L_properties[2]
                    self.num_HBA += L_properties[3]
                    if sum(L_properties) == 4:
                        self.num_comp += 1
                    if sum(L_properties) >= 3:
                        self.num_LRo5 += 1
                except Exception as e:
                    print(f"Lipinski extraction failed: {e}")
        
        if max(num_components) > self.max_frags:
            self.max_frags = max(num_components)
        if min(num_components) < self.min_frags:
            self.min_frags = min(num_components)
        self.avg_frags = (self.avg_frags * self.num_total + sum(num_components)) / (self.num_total + len(samples))
        
        self.num_total += len(samples)
        
        validity_cumulative = self.num_valid_molecules/self.num_total # cumulative validity
        logP_cumulative, molW_cumulative, HBD_cumulative, HBA_cumulative, comp_cumulative, LRo5_cumulative = 0, 0, 0, 0, 0, 0
        if self.num_valid_molecules > 0:
            logP_cumulative = self.num_logP /self.num_valid_molecules
            molW_cumulative = self.num_molW / self.num_valid_molecules
            HBD_cumulative = self.num_HBD / self.num_valid_molecules
            HBA_cumulative = self.num_HBA / self.num_valid_molecules
            comp_cumulative = self.num_comp / self.num_valid_molecules
            LRo5_cumulative = self.num_LRo5 / self.num_valid_molecules

        print("Conditional generation metric:")
        print(f" Validity: {validity_cumulative * 100:.2f}%")
        print(f"Minimum n. fragments: {self.min_frags}")
        print(f"Maximum n. fragments: {self.max_frags}")
        print(f"Average n. fragments: {self.avg_frags:.2f}")
        print(f" logP: {logP_cumulative * 100:.2f}%")
        print(f" molW: {molW_cumulative * 100:.2f}%")
        print(f" HBD: {HBD_cumulative * 100:.2f}%")
        print(f" HBA: {HBA_cumulative * 100:.2f}%")
        print(f" Comp: {comp_cumulative * 100:.2f}%")
        print(f" LRo5: {LRo5_cumulative * 100:.2f}%")
        wandb.log({"val_epoch/valid molecules": validity_cumulative})


    def cond_fn(self, noisy_data, node_mask):
        #self.guidance_model.eval()

        t = noisy_data['t']

        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['t'] # y=t is passed to the classifier

        empty_extra = utils.PlaceHolder(
            X=torch.zeros_like(noisy_data['X_t'])[:, :, :0],
            E=torch.zeros_like(noisy_data['E_t'])[:, :, :, :0],
            y=y
            )

        with torch.enable_grad():
            x_in = X.float().detach().requires_grad_(True)
            e_in = E.float().detach().requires_grad_(True)

            # added to ensure that all parameters and variables are on the same device
            device = self.device

            classifier = list(self.guidance_models.values())[0]
            classifier = classifier.to(device)

            classifier_input = dict(noisy_data)                # keep shallow copy
            classifier_input['X_t'] = x_in                     # swap in grad-enabled X_t
            classifier_input['E_t'] = e_in                     # swap in grad-enabled E_t

            # forward pass through classifier
            pred = classifier.forward(classifier_input, empty_extra, node_mask)
            logits_all = pred.y  # shape: [B, 4]

            # Single-property guidance
            if self.guidance_target in SINGLE_TASK_INDEX_MAP:
                idx = SINGLE_TASK_INDEX_MAP[self.guidance_target]
                logits = logits_all[:, idx].unsqueeze(1)  # shape [B, 1]
                satisfaction_prob = single_property_satisfaction(logits)

            elif self.guidance_target == "all":
                logits = logits_all  # shape [B, 4]
                if self.satisfaction_rule == "composite":
                    satisfaction_prob = composite_property_satisfaction(logits)
                elif self.satisfaction_rule == "LRo5":
                    satisfaction_prob = LRo5_satisfaction(logits)
                else:
                    raise ValueError(f"Unsupported satisfaction_rule: {self.satisfaction_rule}")
            else:
                raise ValueError(f"Unsupported guidance_target: {self.guidance_target}")

            
            target = torch.ones_like(satisfaction_prob)
            
            bce = self.cond_val(satisfaction_prob, target)

            # compute the gradient independently for each graph
            grad_outputs = torch.ones_like(bce)
            grad_x, grad_e = torch.autograd.grad(
                outputs=bce,
                inputs=[x_in, e_in],
                grad_outputs=grad_outputs,
                create_graph=False, # do not create the computation graph for higher-order gradients
                retain_graph=False # do not keep the computation graph in memory (no reuse)
            )

            t_int = int(t.mean().item() * self.T) # was t[0]
            if t_int % 10 == 0:
                print(f'Classifier BCE at step {t_int}: {bce.mean().item():.4f}') # was .item()
            wandb.log({'Guidance BCE': bce.mean().item()})

            x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
            bs, n = x_mask.shape[0], x_mask.shape[1]

            e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
            diag_mask = torch.eye(n)
            diag_mask = ~diag_mask.type_as(e_mask1).bool()
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

            mask_grad_x = grad_x * x_mask
            mask_grad_e = grad_e * e_mask1 * e_mask2 * diag_mask

            mask_grad_e = 1 / 2 * (mask_grad_e + torch.transpose(mask_grad_e, 1, 2))
            return mask_grad_x, mask_grad_e

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        # # Guidance
        lamb = self.cfg.guidance.lambda_guidance

        if lamb > 0: # skip guidance block entirely if not using guidance
        
            grad_x, grad_e = self.cond_fn(noisy_data, node_mask)

            p_eta_x = torch.softmax(- lamb * grad_x, dim=-1)
            p_eta_e = torch.softmax(- lamb * grad_e, dim=-1)

            prob_X_unnormalized = p_eta_x * prob_X
            prob_X_unnormalized[torch.sum(prob_X_unnormalized, dim=-1) == 0] = 1e-7
            prob_X = prob_X_unnormalized / torch.sum(prob_X_unnormalized, dim=-1, keepdim=True)

            prob_E_unnormalized = p_eta_e * prob_E
            prob_E_unnormalized[torch.sum(prob_E_unnormalized, dim=-1) == 0] = 1e-7
            prob_E = prob_E_unnormalized / torch.sum(prob_E_unnormalized, dim=-1, keepdim=True)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0)) # y contains useful info for diffusion
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0)) # y contains useful info for diffusion

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def save_cond_samples(self, samples, file_path):
        cond_results = {'smiles': []} # saves also invalid molecules
        invalid = 0
        # num_components = []

        print("\tConverting conditionally generated molecules to SMILES ...")
        for sample in samples:
            # follows relaxed validity principles
            mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)
            smiles = mol2smiles(mol)
            # try:
            #     mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            #     num_components.append(len(mol_frags))
            # except:
            #     pass

            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    cond_results['smiles'].append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetMolFrags")
                    cond_results['smiles'].append(None)
                    invalid += 1
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    cond_results['smiles'].append(None)
                    invalid += 1
            else:
                print("Invalid molecule obtained (None).")
                cond_results['smiles'].append(None)
                invalid += 1

        print(f"\tDone. Total samples: {len(samples)}")
        print(f"\tValid molecules (after relaxed validity): {len(samples) - invalid}")
        print(f"\tInvalid molecules: {invalid}")
        # self.print(f"\tAverage number of fragments per molecule: {np.mean(num_components):.2f}")
        # self.print(f"\tDisconnected molecules (fragments > 1): {(np.sum(np.array(num_components) > 1))}")

        # save samples
        with open(file_path, 'w') as f:
            for smiles in cond_results['smiles']:
                line = smiles if isinstance(smiles, str) else "None"
                f.write(line + '\n')
