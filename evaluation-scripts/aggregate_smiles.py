import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, help='Folder relative path')
    parser.add_argument('--prefix', type=str, default='cond_smiles', help='Prefix of files containing SMILES')
    parser.add_argument('--output_file', type=str, help='Output file relative path')

    args = parser.parse_args()

    # Parameters
    folder_path = args.folder_path
    prefix = args.prefix
    output_file = args.output_file

    # Aggregate molecules
    molecules = []

    for filename in os.listdir(folder_path):
        if filename.startswith(prefix) and filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r") as file:
                lines = [line.strip() for line in file if line.strip()]
                molecules.extend(lines)

    # Write to output
    with open(output_file, "w") as out:
        for mol in molecules:
            out.write(f"{mol}\n")

