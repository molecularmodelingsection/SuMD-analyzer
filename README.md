# SuMD-analyzer
Analysis script for Supervised Molecular Dynamics (SuMD) trajectories  
**N.B.**: the Python code to run SuMD simulations can be found at github.com/molecularmodelingsection/SuMD

Reference publication:  
**"Investigating RNA-Protein Recognition Mechanisms through Supervised Molecular Dynamics (SuMD) Simulations."**  
Pavan M., Bassani D., Sturlese M., Moro S. (under peer review at *NAR Genomics and Bioinformatics*)

SuMD-analyzer is a Python code that can be utilized to analyze Supervised Molecular Dynamics (SuMD) trajectories and is an evolution of the original tool by V. Salmaso, described in Salmaso, V.; Sturlese, M.; Cuzzolin, A.; Moro, S. **Exploring Protein-Peptide Recognition Pathways Using a Supervised Molecular Dynamics Approach.** Structure. 2017, 25,655–662.e2. Although the script has been originally developed to analyze trajectories involving RNA-protein complexes, it can be easily repurposed for the scope of analyzing trajectories containing peptide/small molecule ligands with small tweaks to the codebase. Generally speaking, we are actively working on making this script usable to analyze whatever trajectory involves the recognition between a receptor and a ligand, regardless of its origin.  

The script allows performing both **geometric and energetic analysis** and is split into **seven different analysis protocol**:  

-**geometry**: perform different geometric analyses on trajectory. This pool of analyses exploits the MDAnalysis (essential) and Barbaba Python package (only really required for working with nucleic acids).  

-**mmgbsa**: performs MMGBSA calculation of the receptor-ligand interaction energy and plots the related profiles (**AMBER >= 16 required**). By default, the script points out to the $AMBERHOME, so if multiple instances of AMBER are installed within your machine you will have to manually edit the script or your environment variables in order to use the intended version.   

-**intEnergy**: performs force field calculation of the receptor-ligand interaction energy and plots the related profiles. This protocol relies on **VMD**, the **NAMD Energy plugin for VMD** and **NAMD** as **external dependencies**. 

-**perResRec**: performs and plots a receptor-based per-residue decomposition of the receptor-ligand interaction energy. This protocol relies on **VMD**, the **NAMD Energy plugin for VMD** and **NAMD** as **external dependencies**.     

-**perResLig**: performs and plots a ligand-based per-residue decomposition of the receptor-ligand interaction energy. This protocol relies on **VMD**, the **NAMD Energy plugin for VMD** and **NAMD** as **external dependencies** (only useful if working with nucleic/peptide ligands, do not use when working with small molecules).  

-**matrix**: performs and plots a 2D (receptor vs ligand) interaction energy matrix. This protocol relies on **VMD**, the **NAMD Energy plugin for VMD** and **NAMD** as **external dependencies** (only useful if working with nucleic/peptide ligands, do not use when working with small molecules).  

-**video**: collects data from previous analyses and assembles a four-panel video. This script does NOT generate the trajectory snapshots for the movie generation: the user has to generate them separately using VMD (in the .tga format), put them in a folder called "frame" within the working directory.  

Editable settings can be set at the beginning of the script. A YAML file is provided in order to reconstitute the right Python environment to make the script work. To run:  
- **conda activate rna-sumd**
- **python3 RNASuMDAnalyzer.py [protocol]**
