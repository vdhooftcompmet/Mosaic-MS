# STRATA-MS

STRATA-MS is a pipeline designed for processing and analyzing mass spectrometry data using SNAP-MS and MS2LDA. Follow the instructions below to set up your environment, download necessary supporting files, and execute a demo run.

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vdhooftcompmet/STRATA-MS.git
cd STRATA-MS
```

### 2. Set Up the Environment
Choose **one** of the following methods depending on your package manager preference:

* **Using Conda/Mamba (Recommended):**
  ```bash
  conda env create -f environment.yaml
  conda activate strata-ms
  ```

* **Using Pip:**
  ```bash
  py -3.11 -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
  pip install -r requirements.txt
  ```

### 3. Download Supporting Files
Run the setup script to download required reference datasets and demo files:
```bash
python setup.py
```

---

## Running the Demo

Once setup is complete, execute the demo workflow using Snakemake:

Navigate to the workflow directory
```bash
cd snakemake
```

Run the pipeline locally using 1 CPU core
```bash
snakemake --cores 1
```

