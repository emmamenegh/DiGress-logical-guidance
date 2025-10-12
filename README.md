# Guiding Diffusion with Logical Constraints: Molecular Graph Generation under Lipinski's Rules

This repository contains the code for running the experiments of the thesis **"Guiding Diffusion with Logical Constraints: Molecular Graph Generation under Lipinski's Rules"**, developed at **Universitat Politècnica de Catalunya (UPC), Facultat d’Informàtica de Barcelona (FIB)**, jointly with the **University of Padua, Department of Mathematics**, as part of the *Master’s in Data Science Double Degree Program*.  

## Repository Origin
This project is adapted from [DiGress](https://github.com/cvignac/DiGress), developed by **Clement Vignac, Igor Krawczuk, and Antoine Siraudin**, and distributed under the MIT License.  
Extensions and modifications © 2025 **Emma Meneghini**.  

## Code Structure
- `./src/` - Adapted code from DiGress, with extensive modifications and corrections.  
  The `datasets`, `guidance`, and `models` subfolders were significantly updated, and several issues in the original implementation were corrected.

- `./configs/` – Configurations for running the experiments, with major changes in the `experiment/` subfolder.  

- `./evaluation_scripts/` – Entirely new code for extracting useful statistics from the generated molecule sets.  

## Usage
To compile and run the programs, please follow the setup and execution instructions provided in the [original DiGress repository](https://github.com/cvignac/DiGress).  

## Acknowledgments
This work was carried out under the supervision of **Dr. Sergi Abadal Cavallé** (UPC-FIB) and **Nicolò Navarin** (University of Padua, Department of Mathematics).  

## License
This repository is distributed under the MIT License.  
See [LICENSE](./LICENSE) for details.