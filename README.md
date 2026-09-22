# Mosaic-MS

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

## Result preview

To get an idea of the type of graphs this pipeline gives, open the recently just example files in Cytoscape. 

#### 1. download cytoscape
To download cytoscape, go to: `https://cytoscape.org/download.html`

#### 2. open example files
`../data/mn_example.cx` shows the metadata added to typical graph dervied from an MS2LDA andsnap-ns analysis

`../data/snapms_example.cx` shows a the results of a SNAP-MS analysis for cluster 1 of the molecular networking example. 

#### 3. download the ChemViz extension
For the full experience, also download the ChemViz Cytoscape extension. 
This extension visualizes chemical structures in SNAP-MS results. 

1. Open Cytoscape. 
2. Go to the top menu bar and click Apps $\rightarrow$ App Store $\rightarrow$ Show App Store (or open the App Manager).  
3. In the search box, type chemViz2. 
4. Select chemViz2 from the list and click the Install button.

## Running the Pipeline

Navigate into the `src` directory to run the CLI commands:

```bash
cd src
```

### CLI Workflow Commands

1. **Run MS2LDA**
   ```bash
   python cli.py run-ms2lda --mgf ../data/test_file.mgf
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