"""
This module does not filter the dataset. The filtered dataset must have been already created
with the code in the 'main' or 'fixed bug' branch, and be readily available in the appropriate folder.
This module adds Lipinski properties to the molecules in the input dataset (regardless of whether they
have been filtered) and saves them all to files. 
"""
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski # to compute logP, MolWt, HBD, HBA
from rdkit.Chem.rdchem import BondType as BT

import os
import os.path as osp
import pathlib
import hashlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url

from src import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges, compute_molecular_metrics
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule


TRAIN_HASH = '05ad85d871958a05c02ab51a4fde8530'
VALID_HASH = 'e53db4bff7dc4784123ae6df72e3b1f0'
TEST_HASH = '677b757ccec4809febd83850b43e1616'


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def compare_hash(output_file: str, correct_hash: str) -> bool:
    """
    Computes the md5 hash of a SMILES file and check it against a given one
    Returns false if hashes are different
    """
    output_hash = hashlib.md5(open(output_file, 'rb').read()).hexdigest()
    if output_hash != correct_hash:
        print(f'{output_file} file has different hash, {output_hash}, than expected, {correct_hash}!')
        return False

    return True


class LipinskiBinaryExtractor:
    def __init__(self):
        self.thresholds = {
            'logP': 5.0,
            'molW': 500.0,
            'HBD': 5.0,
            'HBA': 10.0
        }

    def __call__(self, mol):
        logp = Chem.Crippen.MolLogP(mol)
        molwt = Descriptors.MolWt(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        labels = [
            1 if logp <= self.thresholds['logP'] else 0,
            1 if molwt <= self.thresholds['molW'] else 0,
            1 if hbd <= self.thresholds['HBD'] else 0,
            1 if hba <= self.thresholds['HBA'] else 0
        ]

        return torch.tensor(labels, dtype=torch.long)  # shape: (4,)

class SelectLogPTransform:
    def __call__(self, data):
        data.y = data.y[..., 0:1]  # logP
        return data
    
class SelectMolWTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:2]  # MolWt
        return data
    
class SelectHBDTransform:
    def __call__(self, data):
        data.y = data.y[..., 2:3] # HBD
        return data

class SelectHBATransform:
    def __call__(self, data):
        data.y = data.y[..., -1:] # HBA
        return data

class LipinskiTransform:
    def __call__(self, data):
        # data.y: shape [batch_size, 4], binary values (0 or 1)
        passed = data.y.sum(dim=-1, keepdim=True) >= 3
        data.y = passed.float()  # shape [batch_size, 1]
        return data

class ConjunctiveTransform:
    def __call__(self, data):
        # Check if sum across the 4 properties is exactly 4
        passed = data.y.sum(dim=-1, keepdim=True) == 4
        data.y = passed.float()
        return data

class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class GuacamolDataset(InMemoryDataset):
    train_url = ('https://figshare.com/ndownloader/files/13612760')
    test_url = 'https://figshare.com/ndownloader/files/13612757'
    valid_url = 'https://figshare.com/ndownloader/files/13612766'
    all_url = 'https://figshare.com/ndownloader/files/13612745'

    def __init__(self, stage, root, filter: bool, transform=None, pre_transform=None, pre_filter=None):
        self.stage = stage
        self.filtered = filter
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        if self.filtered:
            return ['new_train.smiles', 'new_val.smiles', 'new_test.smiles']
        else:
            return ['guacamol_v1_train.smiles', 'guacamol_v1_valid.smiles', 'guacamol_v1_test.smiles']

    @property
    def split_file_name(self):
        if self.filtered:
            return ['new_train.smiles', 'new_val.smiles', 'new_test.smiles']
        else:
            return ['guacamol_v1_train.smiles', 'guacamol_v1_valid.smiles', 'guacamol_v1_test.smiles']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
            if self.filtered:
                return ['new_proc_tr.pt', 'new_proc_val.pt', 'new_proc_test.pt']
            else:
                return ['old_proc_tr.pt', 'old_proc_val.pt', 'old_proc_test.pt']

    def download(self):
        """
        Download raw guacamol files.
        """
        try:
            import rdkit  # noqa
            train_path = download_url(self.train_url, self.raw_dir)
            os.rename(train_path, osp.join(self.raw_dir, 'guacamol_v1_train.smiles'))
            train_path = osp.join(self.raw_dir, 'guacamol_v1_train.smiles')

            test_path = download_url(self.test_url, self.raw_dir)
            os.rename(test_path, osp.join(self.raw_dir, 'guacamol_v1_test.smiles'))
            test_path = osp.join(self.raw_dir, 'guacamol_v1_test.smiles')

            valid_path = download_url(self.valid_url, self.raw_dir)
            os.rename(valid_path, osp.join(self.raw_dir, 'guacamol_v1_valid.smiles'))
            valid_path = osp.join(self.raw_dir, 'guacamol_v1_valid.smiles')
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)

        # check the hashes
        # Check whether the md5-hashes of the generated smiles files match
        # the precomputed hashes, this ensures everyone works with the same splits.
        valid_hashes = [
            compare_hash(train_path, TRAIN_HASH),
            compare_hash(valid_path, VALID_HASH),
            compare_hash(test_path, TEST_HASH),
        ]

        if not all(valid_hashes):
            raise SystemExit('Invalid hashes for the dataset files')

        print('Dataset download successful. Hashes are correct.')

        if files_exist(self.split_paths):
            return

    def process(self):
        preprocess = False # do not filter by default

        RDLogger.DisableLog('rdApp.*')
        types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        smile_list = open(self.split_paths[self.file_idx]).readlines()

        # Initialize Lipinski property extractor
        lipinski_extractor = LipinskiBinaryExtractor()

        data_list = []
        smiles_kept = []
        for i, smile in enumerate(tqdm(smile_list)):
            mol = Chem.MolFromSmiles(smile)
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()

            # Extract Lipinski property indicators
            y = lipinski_extractor(mol)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if not preprocess:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data) # this is executed
                continue
            else:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']
                mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                        if len(mol_frags) == 1:
                            smiles_kept.append(smiles)
                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")

        if preprocess:
            smiles_save_path = osp.join(pathlib.Path(self.raw_paths[0]).parent, 'new_test.smiles')
            print(smiles_save_path)
            with open(smiles_save_path, 'w') as f:
                f.writelines('%s\n' % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smile_list)}")
            assert False, "This assert avoids overwriting train smiles with val or test"
        else:
            torch.save(self.collate(data_list), self.processed_paths[self.file_idx]) # save only molecular graphs with Lipinski properties


class GuacamolDataModule(MolecularDataModule):
    def __init__(self, cfg, classifier: bool = False):
        super().__init__(cfg)
        self.remove_h = True # does nothing
        self.datadir = cfg.dataset.datadir
        self.filtered = cfg.dataset.filter
        self.classifier = classifier
        self.train_smiles = []
        self.prepare_data()

    def prepare_data(self) -> None:
        target = self.cfg.general.guidance_target
        if self.classifier and target == "logP":
            transform = SelectLogPTransform()
        elif self.classifier and target == "molW":
            transform = SelectMolWTransform()
        elif self.classifier and target == "HBD":
            transform = SelectHBDTransform()
        elif self.classifier and target == "HBA":
            transform = SelectHBATransform()
        elif self.classifier and target == "LRO5":
            transform = LipinskiTransform()
        elif self.classifier and target == "COMP":
            transform = ConjunctiveTransform()
        else: # if we are not training the classifier
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': GuacamolDataset(stage='train', root=root_path, filter=self.filtered, 
                                             transform=transform if self.classifier else RemoveYTransform()),
                    'val': GuacamolDataset(stage='val', root=root_path, filter=self.filtered, 
                                           transform=transform if self.classifier else RemoveYTransform()),
                    'test': GuacamolDataset(stage='test', root=root_path, filter=self.filtered, 
                                            transform=transform if self.classifier else RemoveYTransform())}
        super().prepare_data(datasets)


class Guacamolinfos(AbstractDatasetInfos):
    atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5,
                    'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
    atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']

    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.name = 'Guacamol'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True # does nothing
        self.num_atom_types = 12
        self.max_weight = 1000

        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]

        self.atom_weights = {1: 12, 2: 14, 3: 16, 4: 19, 5: 10.81, 6: 79.9,
                             7: 35.45, 8: 126.9, 9: 30.97, 10: 30.07, 11: 78.97, 12: 28.09}

        self.node_types = torch.Tensor([7.4090e-01, 1.0693e-01, 1.1220e-01, 1.4213e-02, 6.0579e-05, 1.7171e-03,
        8.4113e-03, 2.2902e-04, 5.6947e-04, 1.4673e-02, 4.1532e-05, 5.3416e-05])

        self.edge_types = torch.Tensor([9.2526e-01, 3.6241e-02, 4.8489e-03, 1.6513e-04, 3.3489e-02])

        self.n_nodes = torch.Tensor([0, 0, 3.5760e-06, 2.7893e-05, 6.9374e-05, 1.6020e-04,
                                     2.8036e-04, 4.3484e-04, 7.3022e-04, 1.1722e-03, 1.7830e-03, 2.8129e-03,
                                     4.0981e-03, 5.5421e-03, 7.9645e-03, 1.0824e-02, 1.4459e-02, 1.8818e-02,
                                     2.3961e-02, 2.9558e-02, 3.6324e-02, 4.1931e-02, 4.8105e-02, 5.2316e-02,
                                     5.6601e-02, 5.7483e-02, 5.6685e-02, 5.2317e-02, 5.2107e-02, 4.9651e-02,
                                     4.8100e-02, 4.4363e-02, 4.0704e-02, 3.5719e-02, 3.1685e-02, 2.6821e-02,
                                     2.2542e-02, 1.8591e-02, 1.6114e-02, 1.3399e-02, 1.1543e-02, 9.6116e-03,
                                     8.4744e-03, 6.9532e-03, 6.2001e-03, 4.9921e-03, 4.4378e-03, 3.5803e-03,
                                     3.3078e-03, 2.7085e-03, 2.6784e-03, 2.2050e-03, 2.0533e-03, 1.5598e-03,
                                     1.5177e-03, 9.8626e-04, 8.6396e-04, 5.6429e-04, 5.0422e-04, 2.9323e-04,
                                     2.2243e-04, 9.8697e-05, 9.9413e-05, 6.0077e-05, 6.9374e-05, 3.0754e-05,
                                     3.5045e-05, 1.6450e-05, 2.1456e-05, 1.2874e-05, 1.2158e-05, 5.7216e-06,
                                     7.1520e-06, 2.8608e-06, 2.8608e-06, 7.1520e-07, 2.8608e-06, 1.4304e-06,
                                     7.1520e-07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 7.1520e-07, 0.0000e+00,
                                     1.4304e-06, 7.1520e-07, 7.1520e-07, 0.0000e+00, 1.4304e-06])

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = torch.zeros(self.max_n_nodes * 3 - 2)
        self.valency_distribution[0: 7] = torch.Tensor([0.0000, 0.1105, 0.2645, 0.3599, 0.2552, 0.0046, 0.0053])

        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

        if recompute_statistics:
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count()
            print("Distribution of the valencies", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies


def get_train_smiles(cfg, datamodule, dataset_infos, evaluate_dataset=False):
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file = "new_train.smiles"
    smiles_path = os.path.join(base_path, cfg.dataset.datadir, smiles_file)

    train_smiles = None
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.array(open(smiles_path).readlines())

    if evaluate_dataset:
        train_dataloader = datamodule.dataloaders['train']
        all_molecules = []
        for i, data in enumerate(tqdm(train_dataloader)):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])
        # all_molecules = all_molecules[:10]
        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles
