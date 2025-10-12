import argparse
import random
import umap
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from cddd.inference import InferenceModel


def lipinski_flags(smile):
    mol = Chem.MolFromSmiles(smile)
    
    molwt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    count = 0
    count += (molwt <= 500)
    count += (logp <= 5)
    count += (hbd <= 5)
    count += (hba <= 10)
    return count

def process_samples(file_path, n_samples):
    with open(file_path, "r") as f:
        samples = [line.strip() for line in f if line.strip()]
    valid_samples = []
    for smile in samples:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None and mol.GetNumAtoms()>0:
                new_smile = Chem.MolToSmiles(mol, isomericSmiles=False)
                valid_samples.append(new_smile)
        except Exception as e:
            print(f"Sanitization failed: {e}")
            continue
    random.shuffle(valid_samples)
    return valid_samples[:n_samples]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='final_smiles.txt', help='Input file relative path')
    parser.add_argument('--training_file', type=str, 
                        default='test-checkpoints-results/training-dataset/new_train_reduced.txt', 
                        help='Relative path of the training set samples.')
    parser.add_argument('--reference_file', type=str, 
                        default='test-checkpoints-results/unconditional-baseline/guidance-code/unconditional_relaxed_smiles.txt', 
                        help='Relative path of unconditional model samples.')
    parser.add_argument('--input_name', type=str, help='Type of input data.')
    parser.add_argument('--reference_name', type=str, help='Type of reference data.')
    parser.add_argument('--output_file', type=str, help='Output file relative path')
    parser.add_argument('--n_samples', type=int, help='Number of samples')

    args = parser.parse_args()

    # set random seed
    random.seed(123)

    # Load training SMILES (to define UMAP coordinates)
    training_list = process_samples(args.training_file, args.n_samples)

    # Load reference SMILES
    reference_list = process_samples(args.reference_file, args.n_samples)
    
    # Load generated SMILES
    generated_list = process_samples(args.input_file, args.n_samples)
    
    # Compute Lipinski rule satisfaction labels
    training_num_rules = [lipinski_flags(smile) for smile in training_list]
    reference_num_rules = [lipinski_flags(smile) for smile in reference_list]
    generated_num_rules = [lipinski_flags(smile) for smile in generated_list]
    all_num_rules = reference_num_rules + generated_num_rules

    # Compute UMAP embeddings
    model = InferenceModel(model_dir="../cddd/default_model")
    training_embeddings = model.seq_to_emb(training_list) # shape: (n_molecules, 512)
    reference_embeddings = model.seq_to_emb(reference_list)  # shape: (n_molecules, 512)
    generated_embeddings = model.seq_to_emb(generated_list)  # shape: (n_molecules, 512)

    embeddings = np.vstack([reference_embeddings, generated_embeddings])

    reference_labels = [args.reference_name] * len(reference_list)
    generated_labels = [args.input_name] * len(generated_list)
    labels = reference_labels + generated_labels

    # Fit and transform
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    umap_model.fit(training_embeddings) # fit the model on the training set for comparability of plots
    # training_coords = umap_model.transform(training_embeddings)
    umap_coords = umap_model.transform(embeddings)

    # # Step 1: Get global axis limits from the full training projection
    # x_min, x_max = training_coords[:, 0].min(), training_coords[:, 0].max()
    # y_min, y_max = training_coords[:, 1].min(), training_coords[:, 1].max()


    # Plot
    plt.figure(figsize=(10, 8))
    colors = {args.reference_name: 'blue', args.input_name: 'red'}

    for label in [args.reference_name, args.input_name]:
        idxs = [i for i, l in enumerate(labels) if l == label]

        # Split based on rule satisfaction
        idxs_good = [i for i in idxs if all_num_rules[i] >= 3]
        idxs_bad = [i for i in idxs if all_num_rules[i] < 3]

        plt.scatter(umap_coords[idxs_good, 0], umap_coords[idxs_good, 1],
                c=colors[label], alpha=0.5, label=f'{label} (≥3 rules)', marker='o')
    
        plt.scatter(umap_coords[idxs_bad, 0], umap_coords[idxs_bad, 1],
                    c=colors[label], alpha=0.5, label=f'{label} (<3 rules)', marker='x')
        
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    # plt.title("UMAP of CDDD Embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)