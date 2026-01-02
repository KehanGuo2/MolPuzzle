from setuptools import setup, find_packages

setup(
    name="molpuzzle",
    version="0.1.0",
    description="Spectrum Analysis using LLMs",
    author="MolPuzzle Contributors",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "transformers",
        "Pillow",
        "scikit-learn",
        "openai",
        "anthropic",
        "fcd",
        "nltk",
        "Levenshtein",
        "rdkit",
    ],
    entry_points={
        "console_scripts": [
            "molpuzzle=molpuzzle.spectrum_analysis:main",
        ],
    },
)

