import sys
import os
import pandas as pd
import pytest
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from molpuzzle.spectrum_analysis import sample_data

def test_sample_data(tmp_path):
    # Create dummy data
    data = {
        'Molecule Index': [1, 2, 3, 4],
        'cls': ['A', 'A', 'B', 'B'],
        'Question': ['Q1', 'Q2', 'Q3', 'Q4'],
        'Answer': ['A1', 'A2', 'A3', 'A4']
    }
    df = pd.DataFrame(data)
    input_csv = tmp_path / "test_input.csv"
    df.to_csv(input_csv, index=False)
    
    output_prefix = tmp_path / "test_output"
    ratios = {'A': 0.5, 'B': 0.5}
    
    sample_data(str(input_csv), str(output_prefix), ratios, total_samples=4, iterations=1)
    
    output_file = tmp_path / "test_output_0.csv"
    assert output_file.exists()
    
    sampled_df = pd.read_csv(output_file)
    assert len(sampled_df) == 4
    assert 'cls' in sampled_df.columns

