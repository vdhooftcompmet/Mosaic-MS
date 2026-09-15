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

-### 3. Download Supporting Files
Run the setup script to download required reference datasets, models, and demo files (type `y` when prompted):
```bash
python setup.py
```

---

## Running the Pipeline

Navigate into the `src` directory to run the CLI commands:

```bash
cd src
```

### CLI Workflow Commands

1. **Run MS2LDA**
   ```bash
   python cli.py run-ms2lda --mgf <path_to_mgf>
   ```

2. **Run Molecular Networking**
   ```bash
   python cli.py run-mn --mgf ../data/test_file.mgf
   ```

3. **Run SNAP-MS**
   ```bash
   python cli.py run-snapms --graph ../results/base.cx
   ```

4. **Add MS2LDA Results to Graph**
   ```bash
   python cli.py add-ms2lda --graph ../results/base.cx --model ../results/model.lda
   ```

5. **Add SNAP-MS Results to Graph**
   ```bash
   python cli.py add-snapms --graph ../results/base.cx --snapms ../results/snapms
   ```