import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

# Function to compute Lipinski properties
def calculate_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        print("SMILES cannot be parsed!")
        return pd.Series([None, None, None, None])
    if mol is not None and mol.GetNumAtoms()>0:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
    else:
        return pd.Series([None, None, None, None])
    return pd.Series([mw, logp, hbd, hba])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SMILES file for Lipinski calculation")
    parser.add_argument('--input_file', type=str, help='Relative path to the input SMILES file')
    parser.add_argument('--output_file', type=str, help='Relative path to the output CSV file')

    # Parse arguments
    args = parser.parse_args()

    # Load your SMILES dataset
    df = pd.read_csv(args.input_file, header=None, names=['SMILES'])

    # Apply function
    df[['MolWt', 'LogP', 'HBD', 'HBA']] = df['SMILES'].apply(calculate_properties)

    # Remove invalid rows (e.g. invalid SMILES)
    num_na_rows = df[['MolWt', 'LogP', 'HBD', 'HBA']].isna().any(axis=1).sum()
    print(f"Number of molecules with at least one NA: {num_na_rows}")
    df_clean = df.dropna(subset=['MolWt', 'LogP', 'HBD', 'HBA'])

    # Save to CSV for reuse
    df_clean.to_csv(args.output_file, index=False)
