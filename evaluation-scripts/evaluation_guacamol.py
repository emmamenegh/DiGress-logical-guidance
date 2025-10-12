import os
import argparse
import json
import random
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, DataStructs

from guacamol.assess_distribution_learning import _assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator


# Dummy generator
class StaticGenerator(DistributionMatchingGenerator):
    def __init__(self, smiles):
        random.shuffle(smiles) # shuffle first
        self.smiles = smiles
        self.index = 0

    def generate(self, number_samples: int) -> list:
        end_index = self.index + number_samples
        if end_index <= len(self.smiles):
            batch = self.smiles[self.index:end_index]
            self.index = end_index
        else:
            remaining = end_index - len(self.smiles)
            batch = self.smiles[self.index:] + self.smiles[:remaining]
            self.index = remaining
        return batch


# Load SMILES from file
def load_smiles(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

    
# Lipinski Rule of Five
def extract_lipinski(mol):
    return {
        'molwt': Descriptors.MolWt(mol) <= 500,
        'logp': Descriptors.MolLogP(mol) <= 5,
        'hbd': Descriptors.NumHDonors(mol) <= 5,
        'hba': Descriptors.NumHAcceptors(mol) <= 10
    }


def compute_lipinski(smiles_list):

    valid = 0
    metrics = {'molwt': 0, 'logp': 0, 'hbd': 0, 'hba': 0, 'comp': 0, 'lro5': 0}

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles) # used in validity benchmark
            if mol is not None and mol.GetNumAtoms()>0:
                valid += 1
                try: 
                    result = extract_lipinski(mol)
                    for key in result:
                        metrics[key] += int(result[key])
                    if sum(result.values()) == 4:
                        metrics['comp'] += 1
                    if sum(result.values()) >= 3:
                        metrics['lro5'] += 1
                except Exception as e:
                    print(f"Couldn't evaluate Lipinski's properties: {e}")
                    continue
        except Exception as e:
            print(f"Invalid chemistry! {e}")
            continue

    metrics_perc = {key: value / valid * 100 for key, value in metrics.items()}
    valid_perc = valid / len(smiles_list) * 100

    return metrics_perc, valid_perc


def compute_internal_diversity(smiles_list):
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms() > 0:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
            fps.append(fp)
    
    n = len(fps)
    if n < 2:
        return 0.0  # Cannot compute diversity with <2 valid molecules

    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1 - sim)

    return sum(distances) / len(distances)

def compute_external_diversity(smiles_list):
    # Process training molecules
    with open("test-checkpoints-results/training-dataset/new_train_reduced.txt", "r") as f:
        training_list = [line.strip() for line in f if line.strip()]
    fps = []
    for smile in training_list:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None and mol.GetNumAtoms()>0:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
                fps.append(fp)
        except Exception as e:
            print(f"Sanitization failed: {e}")
            continue
    training_fps = fps

    # Process generated molecules
    fps = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and mol.GetNumAtoms()>0:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
                fps.append(fp)
        except Exception as e:
            print(f"Sanitization failed: {e}")
            continue
    n = len(fps)
    if n < 2:
        return 0.0  # Cannot compute diversity with <2 valid molecules
    
    generated_fps = fps

    # Compute max similarity to training set for each generated molecule
    max_similarities = []
    for gfp in generated_fps:
        sims = DataStructs.BulkTanimotoSimilarity(gfp, training_fps)
        if sims:
            max_similarities.append(max(sims))

    if len(max_similarities) == 0:
        return 0.0

    avg_max_similarity = sum(max_similarities) / len(max_similarities)
    external_diversity = 1.0 - avg_max_similarity
    return external_diversity

    

    

if __name__ == '__main__':

    SEED = 42  # or any integer
    random.seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='final_smiles.txt', help='Input file relative path')
    parser.add_argument('--output_file', type=str, help='Output file relative path')
    parser.add_argument('--benchmark_metrics', action='store_true', help='Whether to calculate benchmark metrics')
    parser.add_argument('--n_samples', type=int, help='Number of samples')

    args = parser.parse_args()

    output_filename = os.path.splitext(args.output_file)[0]


    # Run the full benchmark suite
    if args.benchmark_metrics:

        # Load generated SMILES
        generated = load_smiles(f'{args.input_file}')
        # Wrap your generator
        model = StaticGenerator(generated)

        _assess_distribution_learning(
            model=model,
            chembl_training_file='test-checkpoints-results/training-dataset/new_train_reduced.txt', # always reference
            json_output_file=f'{output_filename}.json',
            benchmark_version='v1',
            number_samples = args.n_samples
        )

        # Read benchmark results from JSON and save in txt file
        with open(f'{output_filename}.json') as f:
            results = json.load(f)['results']

        with open(f'{args.output_file}', 'w') as f_out:
            for item in results:
                name = item['benchmark_name']
                score = item['score']
                line = f"{name}: {score:.4f}"
                f_out.write(line + '\n')
    

    # Compute Lipinski metrics
    with open(args.input_file, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    random.shuffle(smiles_list)
    smiles_list = smiles_list[:args.n_samples]
    lipinski_metrics, validity = compute_lipinski(smiles_list)

    # Write Lipinski stats
    with open(f'{args.output_file}', 'a') as f_out:
        if not args.benchmark_metrics:
            f_out.write(f"Validity: {validity:.2f}")
        f_out.write("\nLipinski statistics:\n")
        for key, value in lipinski_metrics.items():
            f_out.write(f"{key.upper()}: {value:.2f}\n")

    # Compute diversity
    diversity_int = compute_internal_diversity(smiles_list)
    diversity_ext = compute_external_diversity(smiles_list)

    # Write diversity
    with open(f'{args.output_file}', 'a') as f_out:
        f_out.write(f"\nIntDiversity: {diversity_int:.4f}\n")
        if args.benchmark_metrics:
            f_out.write(f"ExtDiversity: {diversity_ext:.4f}")


    
