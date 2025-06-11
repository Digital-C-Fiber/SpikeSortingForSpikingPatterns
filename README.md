# Supervised Spike Sorting Pipeline for Microneurography

This repository provides a **[Snakemake](https://snakemake.readthedocs.io/en/stable/)**-based pipeline for supervised spike sorting of **microneurography** recordings from human C-nociceptors. The pipeline is designed to address challenges such as single-electrode ambiguity, waveform variability, and high fiber similarity, which traditionally hinder the analysis of peripheral nociceptive activity.

Our method incorporates:
- **Spike detection via thresholding**
- Multiple **feature set extraction methods described [here](https://www.biorxiv.org/content/10.1101/2024.12.31.630860v2)**
- Supervised classification using:
  - **One-Class SVM**
  - **SVM**
  - **XGBoost**
- Validation using **experimental derived ground truth data from electrical stimulation**

---
## Spike Tracking via the marking method

During the experiment the **[marking method](https://pubmed.ncbi.nlm.nih.gov/7672025/)** is applied, a special electrical stimulation protocl to create spike tracks (vertical alignment of fiber responses). They can be extracted and analyzed post hoc. 

In our workflow, we use two the following tracking algorithm for microneurography data to identify and track spikes evoked by background stimuli:
- **Dapsys** – [www.dapsys.net](http://www.dapsys.net) based on [Turnquist et al., 2016](https://pubmed.ncbi.nlm.nih.gov/26778609/)

### The extracted spike times and track labels are essential inputs for running our supervised spike sorting pipeline. The setup instructions are provided below.

---


## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Digital-C-Fiber/SpikeSortingForSpikingPatterns.git
cd SpikeSortingForSpikingPatterns
```

### 2. Create the Conda environment
This pipeline uses `conda` and `snakemake` with `Python 3.11`. We provide an environment file.

```bash
conda env create -f environment.yml
conda activate Snakemake311
```

If you haven't already installed `snakemake`:
```bash
conda install -c conda-forge snakemake
```

---
## Snakemake Directory Overview

```
├── Snakefile                      # Main Snakemake workflow
├── config.yaml                    # Configuration for test dataset paths and parameters
├── environment.yml                # Conda environment file
├── scripts/                       # Core processing script
│ ├── create_nix.py
│ ├── detection_via_thresholding.py
│ ├── feature_extraction.py
│ ├── helper.py
│ ├── pre_processing.py
│ ├── pre_processing_classification.py
│ ├── read_in_data.py
│ └── templates_and_snr.py
├── model/                        # Classification models
│ ├── one_class_svm.py
│ ├── svm_model.py
│ └── xgboost_model.py
├── datasets_test/
│   ├── testset_1.dps              # Example Dapsys file
│   └── nix/                       # Output folder for NIX
├── Statistics/                    # R scripts for statistical analysis
```

---
##  Configuration (`config.yaml`)

Each dataset is defined under the `datasets` section. Example:

```yaml
datasets:
  ATest:
    path: "datasets_test/testset_1.dps"                # Path to raw Dapsys file
    path_nix: "datasets_test/nix/testset_1.nix"        # Output path for NIX format
    name: "ATest"                                      # Identifier for the dataset
    path_dapsys: "NI Puls Stimulator/Continuous Recording"  # Internal path in Dapsys hierarchy
    time1: 200                                         # Start time of analysis window
    time2: 922                                         # End time of analysis window
    track_of_interest: "Track3"                        # Track label for supervised training
    threshold: 3.00                                    # Spike detection threshold
    ground_truth_flag: False                           # Use ground truth flag for evaluation (True/False), in test file, no grount truth available
```
---
# Running the Pipeline

To execute the full spike sorting workflow, follow the steps below:

---

## 1. Configure Your Dataset

Open and edit the `config.yaml` file to specify:

- Dataset paths (`path`, `path_nix`)
- Processing window (`time1`, `time2`)
- Detection threshold (`threshold`)
- Track label for supervised classification (`track_of_interest`)
- Whether to use ground truth evaluation (`ground_truth_flag`)

---

## 2. Run the Pipeline

Execute the pipeline using Snakemake with the desired number of CPU cores:

```bash
snakemake --cores 8
```
---

## Contact

If you have any questions, issues, or suggestions, feel free to reach out:

Alina Troglio
Email: alina.troglio@rwth-aachen.de

---

## How to Cite

If you use this pipeline in your work, please cite our preprint:
Coming soon
