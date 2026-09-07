# STRATA-MS

STRATA-MS is a pipeline designed for processing and analyzing mass spectrometry data using **SNAP-MS**, **Molecular Networking**, and **MS2LDA**. Follow the instructions below to set up your environment, download necessary supporting files, and execute a workflow.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vdhooftcompmet/STRATA-MS.git
cd STRATA-MS
```

### 2. Set Up the Environment
Choose one of the following methods depending on your package manager preference:

#### Using Conda/Mamba (Recommended):
```bash
conda env create -f environment.yaml
conda activate strata-ms
```

#### Using Pip:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Supporting Files
Run the setup script to download required reference datasets, models, and demo files:
```bash
python setup.py
```

---

## Running the Pipeline
The pipeline uses **Snakemake** to manage execution, file validation, and logging.

### 1. Navigate to the Directory

Navigate into the snakemake directory before running any Snakemake commands:
```bash
cd snakemake
```

### 2. Configure the Run
Edit `snakemake/config.yaml` to specify your input `.mgf` file, output directories, model paths, and parameter sweeps (e.g., similarity types, motif numbers).

### 3. Configure tool-specific parameters
Edit the .yaml files in `snakemake/params/tools.yaml` to specify tool-specific settings if needed.  

### 4 Pre-Flight Dry Run
Verify that your input files, models, parameter files, and rules resolve correctly before running:
```bash
snakemake -n
```

### 5. Execute the Pipeline
Run the pipeline locally. Default execution options (CPU cores, logging preferences, failed-job handling) are pre-configured in `profiles/default/config.yaml`:
```bash
snakemake
```

To override default cores or run with custom parameters on the command line:
```bash
snakemake --cores 8
```

---

## Outputs & Debugging

* **Results:** Outputs (molecular networks, LDA models, motifs, SnapMS annotations, and plots) are saved in structured subdirectories within your designated results folder (`results/networks/`, `results/ms2lda/`, `results/snapms/`).
* **Logs:** Execution logs for every rule are automatically written to `results/logs/<rule_name>/` for simple step-by-step debugging.
