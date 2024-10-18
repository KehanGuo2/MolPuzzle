<div align="center">

<img src="/image/MolPuzzle_logo2.png" width="100%">



# MolPuzzle: Can LLMs Solve Molecule Puzzles? A Multimodal Benchmark for Molecular Structure Elucidation


[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%8D-blue?style=for-the-badge&logoWidth=40)](https://kehanguo2.github.io/Molpuzzle.io/)
[![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightgrey?style=for-the-badge&logoWidth=40)](https://kehanguo2.github.io/Molpuzzle.io/)
[![Dataset](https://img.shields.io/badge/Dataset-%F0%9F%92%BE-green?style=for-the-badge&logoWidth=40)](https://kehanguo2.github.io/Molpuzzle.io/)


</div>



## 🔥 Updates & News
- 2024.09: 🎉🎉 MolPuzzle has been accepted by NeurIPS 2024 Dataset and Benchmark Track as a spotlight!


## 💡Overview

We present MolPuzzle, a benchmark comprising **234** instances of structure elucidation, which feature over **18,000 QA samples** presented in a sequential puzzle-solving process, involving three interlinked subtasks: **molecule understanding**, **spectrum interpretation**, and **molecule construction**.

<div align="center">
<img width="530" alt="Screenshot 2024-07-11 at 17 58 31" src="https://github.com/user-attachments/assets/bbf3fae0-aa8f-4cd5-a274-55a1e42285e9">
</div>

The figure illustrates the problem of molecular structure elucidation alongside its analogical counterpart, the crossword puzzle, highlighting the parallels in strategy and complexity between these two intellectual challenges




## 📕Model Summary

| Model                | Stage 1 | Stage 2 | Stage 3 |
|:---------------------|:--------|:--------|:--------|
| GPT-4o               | ✅      | ✅      | ✅      |
| Claude-3             | ✅      | ❌      | ✅      |
| Gemini-pro           | ✅      | ❌      | ✅      |
| GPT-3.5              | ✅      | ❌      | ✅      |
| Gemini-3-pro-vision  | ❌      | ✅      | ❌      |
| LLava1.5-8b          | ❌      | ✅      | ❌      |
| Qwen-VL-Chat         | ❌      | ✅      | ❌      |
| InstructBLIP-7b      | ❌      | ✅      | ❌      |
| InstructBLIP-13b     | ❌      | ✅      | ❌      |
| Llama3-8b            | ✅      | ❌      | ❌      |
| Vicuna-7b            | ✅      | ❌      | ❌      |
| Llama2-7b            | ✅      | ❌      | ❌      |
| Llama2-13b           | ✅      | ❌      | ❌      |
| Mistral-7b           | ✅      | ❌      | ❌      |



## 📊Dataset Statistics
<img width="746" alt="Screenshot 2024-07-11 at 18 19 17" src="https://github.com/user-attachments/assets/1253bda0-c894-47f1-ae35-93864377afbf">


The initial molecules were selected by referencing the textbook Organic Structures from Spectra, 4th Edition, available as an online PDF on ResearchGate. We chose 234 molecules based on spectrum tasks involving IR, MS, 1H-NMR, and 13C-NMR to reflect a difficulty level suitable for graduate students. To address copyright concerns, we excluded molecules with publicly available mass spectrometry (MS) spectra in open-source databases from our study. The remaining spectra were sourced from public resources, notably the PubChem database. For additional spectra that were unavailable, we used simulation methods and provided a Jupyter notebook to generate these data, ensuring high-quality spectra for analysis. 

You can download the dataset at [data](https://github.com/KehanGuo2/MolPuzzle/tree/main/Data)

## 🔧Usage Demos

1. **Install Required Packages**  
   Install the necessary Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Setup**  
   - Add API keys for **OpenAI**, **Claude**, and **Gemini** models 
   - Example Commands (Stage 2)
   - #### Generate Responses for IR Using Multiple Models
```bash
      python stage2.py --task IR --action generate_responses --models instructBlip-7B instructBlip-13B llava gpt-4 claude-v1 --iterations 3
```
   - #### Evaluate Responses for IR
```bash
      python stage2.py --task IR --action evaluate --models instructBlip-7B instructBlip-13B llava gpt-4 claude-v1 --iterations 3
```










## ☎️ Contact us
Kehan Guo: kguo2@nd.edu