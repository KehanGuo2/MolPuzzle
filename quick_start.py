import os
import sys
import json
import pandas as pd
import argparse

# Add src to path so we can import molpuzzle without installation
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from molpuzzle.spectrum_analysis import sample_data

def main():
    print("Welcome to MolPuzzle Quick Start Session!")
    print("=========================================")

    # Define paths
    data_dir = "Data"
    sample_dir = "sample_data"
    os.makedirs(sample_dir, exist_ok=True)
    
    json_path = os.path.join(data_dir, "Stage1.json")
    csv_path = os.path.join(sample_dir, "stage1_sample.csv")

    # 1. Load Data Sample
    print(f"\n[1] Loading data from {json_path}...")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Display first few entries
    print(f"    Loaded {len(data)} entries.")
    print("    Here is a sample entry:")
    print(json.dumps(data[0], indent=4))

    # 2. Convert to CSV for processing
    print(f"\n[2] Converting to CSV format for processing -> {csv_path}...")
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print("    Conversion complete.")

    # 3. Run Sample Data Session
    print("\n[3] Running Data Sampling Session...")
    print("    This demonstrates how we sample questions for evaluation.")
    
    # Define ratios for sampling (example ratios based on 'cls' column)
    # Let's inspect unique classes first
    unique_classes = df['cls'].unique()
    print(f"    Available classes: {unique_classes}")
    
    # Create a simple ratio dictionary (equal distribution for demo)
    ratios = {cls: 1.0/len(unique_classes) for cls in unique_classes}
    
    output_prefix = os.path.join(sample_dir, "sampled_output")
    
    try:
        sample_data(csv_path, output_prefix, ratios, total_samples=10, iterations=1)
        print(f"    Sampling complete. Output saved to {output_prefix}_0.csv")
        
        # Show sampled result
        sampled_df = pd.read_csv(f"{output_prefix}_0.csv")
        print("\n    Sampled Data Preview:")
        print(sampled_df[['Molecule Index', 'Question', 'Answer']].head().to_markdown(index=False))
        
    except Exception as e:
        print(f"    Error during sampling: {e}")

    print("\n[4] Next Steps")
    print("    You can now run the full analysis using the command line interface:")
    print("    $ python src/molpuzzle/spectrum_analysis.py --task H-NMR --action evaluate --models gpt-4 --input_csv sample_data/sampled_output_0.csv")
    print("\nThank you for using MolPuzzle!")

if __name__ == "__main__":
    main()

