<div align="center">


<img src="https://github.com/KehanGuo2/MolPuzzle/blob/main/image/MolPuzzle_logo.png" width="100%">

# Can LLMs Solve Molecule Puzzles? A Multimodal Benchmark for Molecular Structure Elucidation


[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%8D-blue?style=for-the-badge&logoWidth=40)](https://github.com/KehanGuo2/MolPuzzle/edit/main/README.md)
[![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightgrey?style=for-the-badge&logoWidth=40)](https://github.com/KehanGuo2/MolPuzzle/edit/main/README.md)
[![Dataset](https://img.shields.io/badge/Dataset-%F0%9F%92%BE-green?style=for-the-badge&logoWidth=40)](https://github.com/KehanGuo2/MolPuzzle/edit/main/README.md)
</div>

<div align="center">




</div>


## Updates & News
- [09/29/2024] 🥂 **MolePuzze has been accepted by NeurIPS 2024 Datasets and Benchmarks Track as a spotlight.! See you in Vancouver!**

## 🙋 **About MolePuzzle**
We introduce a new challenge: molecular structure elucidation, which involves deducing a molecule’s structure from various types of spectral data. Solving such a molecular puzzle, akin to solving crossword puzzles, poses reasoning challenges that require integrating clues from diverse sources and engaging in iterative hypothesis testing. To address this challenging problem with LLMs, we present **MolPuzzle**, a benchmark comprising 234 instances of structure elucidation, which feature over 18,000 QA samples presented in a sequential puzzle-solving process, involving three interlinked sub-tasks: **molecule understanding**, **spectrum interpretation**, and **molecule construction**.


<div align="center">
<img width="700" alt="Screenshot 2024-07-11 at 17 58 31" src="https://github.com/user-attachments/assets/bbf3fae0-aa8f-4cd5-a274-55a1e42285e9">
</div>

The figure illustrates the problem of molecular structure elucidation alongside its analogical counterpart, the crossword puzzle, highlighting the parallels in strategy and complexity between these two intellectual challenges


## 📖 **Dataset Usage**
<img width="746" alt="Screenshot 2024-07-11 at 18 19 17" src="https://github.com/user-attachments/assets/1253bda0-c894-47f1-ae35-93864377afbf">


## 🧹 **Before Evaluation**
### **Installation**

## Data source
The initial molecules were selected by referencing the textbook Organic Structures from Spectra, 4th Edition, available as an online PDF on ResearchGate. We chose 234 molecules based on spectrum tasks involving IR, MS, 1H-NMR, and 13C-NMR to reflect a difficulty level suitable for graduate students. To address copyright concerns, we excluded molecules with publicly available mass spectrometry (MS) spectra in open-source databases from our study. The remaining spectra were sourced from public resources, notably the PubChem database. For additional spectra that were unavailable, we used simulation methods and provided a Jupyter notebook to generate these data, ensuring high-quality spectra for analysis. 

You can download the dataset at [data](https://github.com/KehanGuo2/MolPuzzle/tree/main/Data)

## Usage Demos

We offer demo examples for tasks in each Stage, the notebook can be found here [Demos](https://github.com/KehanGuo2/MolPuzzle/tree/main/demos)

