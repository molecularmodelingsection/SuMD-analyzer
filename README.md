# SuMD-analyzer
Analysis script for Supervised Molecular Dynamics (SuMD) trajectories  
**N.B.**: the Python code to run SuMD simulations can be found at github.com/molecularmodelingsection/SuMD

Reference publication:  
**"Investigating RNA-Protein Recognition Mechanisms through Supervised Molecular Dynamics (SuMD) Simulations."**  
Pavan M., Bassani D., Sturlese M., Moro S. (under peer review at *NAR Genomics and Bioinformatics*)

SuMD-analyzer is a Python code that can be utilized to analyze Supervised Molecular Dynamics (SuMD) trajectories and is an evolution of the original tool by V. Salmaso, described in Salmaso, V.; Sturlese, M.; Cuzzolin, A.; Moro, S. **Exploring Protein-Peptide Recognition Pathways Using a Supervised Molecular Dynamics Approach.** Structure. 2017, 25,655–662.e2. Although the script has been originally developed to analyze trajectories involving RNA-protein complexes, it can be easily repurposed for the scope of 

The script allows performing both **geometric and energetic analysis** and is split into **seven different analysis protocol**:
-**geometry**: perform different geometric analyses on trajectory
-**mmgbsa**: performs MMGBSA calculation of the receptor-ligand interaction energy and plots the related profiles (**AMBER >= 16 required**)
-**intEnergy**: performs force field calculation of the receptor-ligand interaction energy and plots the related profiles (**NAMD required**)
-**perResRec**: performs and plots a receptor-based per-residue decomposition of the receptor-ligand interaction energy (**NAMD required**)
-**perResLig**: performs and plots a ligand-based per-residue decomposition of the receptor-ligand interaction energy (**NAMD required**)
-**matrix**: performs and plots a 2D (receptor vs ligand) interaction energy matrix  (**NAMD required**)
-**video**: collects data from previous analyses and assembles a four-panel video. This script does NOT generate the trajectory snapshots for the movie generation: the user has to generate them separately using VMD (in the .tga format), put them in a folder called "frame" within the working directory.

Editable settings can be set at the beginning of the script. 
