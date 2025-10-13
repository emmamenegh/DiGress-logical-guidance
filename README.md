# Guiding Diffusion with Logical Constraints: Molecular Graph Generation under Lipinski's Rules

This repository contains the code for running the experiments of the thesis **"Guiding Diffusion with Logical Constraints: Molecular Graph Generation under Lipinski's Rules"**, developed at **Universitat Politècnica de Catalunya (UPC), Facultat d’Informàtica de Barcelona (FIB)**, jointly with the **University of Padua, Department of Mathematics**, as part of the *Master’s in Data Science Double Degree Programme*.  

## Repository Origin
This project is adapted from [DiGress](https://github.com/cvignac/DiGress), developed by **Clement Vignac, Igor Krawczuk, and Antoine Siraudin**, and distributed under the MIT License.  
Extensions and modifications © 2025 **Emma Meneghini**.  

## Code Structure
The repository combines adapted components from the DiGress `guidance` branch with original contributions developed for this thesis:
- `./src/` – Adapted code from DiGress, with corrections and substantial extensions.  
  The `datasets`, `guidance`, and `models` subfolders were expanded with new functionality, and several issues from the original implementation were addressed.

- `./configs/` – Configurations for running the experiments, with significant additions in the `experiment/` subfolder.

- `./evaluation_scripts/` – Entirely new code for extracting detailed statistics from the generated molecule sets.  

## Usage
The setups and requirements provided in this repository should be used to build the environment. However, the compilation and execution procedures remain the same as those described in the [original DiGress repository](https://github.com/cvignac/DiGress).

## Acknowledgments
This work was carried out under the supervision of **Dr. Sergi Abadal Cavallé** (UPC-FIB) and **Dr. Nicolò Navarin** (University of Padua, Department of Mathematics).  

## License
This repository is distributed under the MIT License. See [LICENSE](./LICENSE) for details.