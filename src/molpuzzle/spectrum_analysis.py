# Import Required Libraries
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    AutoModelForCausalLM,
    LlavaForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from sklearn.metrics import accuracy_score, f1_score
import openai
import anthropic
import base64
from pathlib import Path

# Constants
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = ROOT_DIR / "Data"
CACHE_DIR = ROOT_DIR / "cache"

# Set API Keys
# Best practice: Load from environment variables, no hardcoded placeholders
# os.environ['OPENAI_API_KEY'] = ... (Expected to be set in environment)

# Define Model Paths
def get_model_path(model_name):
    # Allow overriding paths via environment variables
    env_var_name = f"{model_name.upper().replace('-', '_')}_PATH"
    path = os.getenv(env_var_name)
    if path:
        return path
    
    # Defaults or placeholders - raise error if critical and missing
    paths = {
        'llava': 'llava-hf/llava-1.5-7b-hf', # Example HF path, user should override
        'instructBlip-7B': 'Salesforce/instructblip-vicuna-7b',
        'instructBlip-13B': 'Salesforce/instructblip-vicuna-13b',
        'Qwen-VL-Chat': 'Qwen/Qwen-VL-Chat',
        'InternVL': 'OpenGVLab/InternVL-Chat-V1-5',
    }
    return paths.get(model_name)

# Define Functional Groups
functional_groups = [
    "Alcohol", "Phenol", "Ketone", "Acid", "Aldehyde", "Ester", "Nitrile",
    "Isocyanate", "Amine", "Ether", "Sulfide", "Halide"
]

def get_image_path(task, molecule_index):
    """
    Resolves the image path based on task and molecule index.
    """
    index = str(molecule_index).strip()
    
    # Handle index format (e.g., '5_99' or '99')
    # Assuming filenames match the index provided in the JSON/CSV.
    # If the index in CSV is "5_99", file is "5_99.png".
    
    filename = f"{index}.png"
    
    task_folders = {
        'IR': 'ir_figures',
        'MASS': 'ms_figures',
        'H-NMR': 'nmr_H_complet',
        'C-NMR': 'nmr_C_figures'
    }
    
    folder = task_folders.get(task)
    if not folder:
        raise ValueError(f"Unknown task for image path: {task}")
        
    return DATA_DIR / "Spectrum_images" / folder / filename

# Data Preprocessing Functions
def generate_ir_questions(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    columns = ["Molecule Index", "SMILES", "cls", "Formula", "Question", "Answer"]
    results_df = pd.DataFrame(columns=columns)

    for _, row in data.iterrows():
        # Q31: O-H Stretching question
        oh_question = "Does the IR spectrum contain a broad absorption peak of O-H stretching around 3300 cm⁻¹?"
        oh_answer = "Yes" if row['O-H Stretching'] == 1 else "No"
        qclass = "IR spectrum peak analysis"
        results_df.loc[len(results_df)] = [row['Molecule Index'], row['SMILES'], qclass, row['Formula'], oh_question, oh_answer]

        # Q32: Alkyl question
        alkyl_question = "Does the IR spectrum contain a sharp absorption peak of Alkyl stretching around 2900 cm⁻¹?"
        alkyl_answer = "Yes" if row['Alkyl'] == 1 else "No"
        qclass = "IR spectrum peak analysis"
        results_df.loc[len(results_df)] = [row['Molecule Index'], row['SMILES'], qclass, row['Formula'], alkyl_question, alkyl_answer]

        # Q33: C=O Stretching question
        co_question = "Does the IR spectrum contain a strong, sharp peak of C=O stretching around 1700 cm⁻¹?"
        co_answer = "Yes" if row['C=O Stretching'] == 1 else "No"
        qclass = "IR spectrum peak analysis"
        results_df.loc[len(results_df)] = [row['Molecule Index'], row['SMILES'], qclass, row['Formula'], co_question, co_answer]

        # Q34: N-H Stretching question
        nh_question = "Does the IR spectrum contain a broad absorption peak of N-H stretching around 3200-3600 cm⁻¹?"
        nh_answer = "Yes" if row['N-H Stretching'] == 1 else "No"
        qclass = "IR spectrum peak analysis"
        results_df.loc[len(results_df)] = [row['Molecule Index'], row['SMILES'], qclass, row['Formula'], nh_question, nh_answer]

        # Q35: Triple bond C-H Stretching question
        tbch_question = "Does the IR spectrum contain a weak absorption peak of triple bond C-H stretching around 2260-2100 cm⁻¹?"
        tbch_answer = "Yes" if row['triple bond C-H Stretching:'] == 1 else "No"
        qclass = "IR spectrum peak analysis"
        results_df.loc[len(results_df)] = [row['Molecule Index'], row['SMILES'], qclass, row['Formula'], tbch_question, tbch_answer]

        # Functional group presence questions
        for group in functional_groups:
            question = f"Examine the IR spectrum to determine if the molecule could potentially contain specific functional groups: {group}? Look for the presence of characteristic absorption bands and analyze the wavenumbers and intensities of these peaks. This analysis will help identify the functional groups and key structural features within the molecule."
            answer = "Yes" if row[group] == 1 else "No"
            qclass = "IR spectrum structure elucidation"
            results_df.loc[len(results_df)] = [row['Molecule Index'], row['SMILES'], qclass, row['Formula'], question, answer]

    results_df.to_csv(output_csv, index=False)
    print(f"IR questions generated and saved to '{output_csv}'")

def generate_mass_questions(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    columns = ["Molecule Index", "SMILES", "cls", "Formula", "Question", "Answer"]
    results_df = pd.DataFrame(columns=columns)

    for _, row in data.iterrows():
        for group in functional_groups:
            question = f"Examine the MS spectrum to determine if the molecule could potentially contain specific fragments: {group}. Look into the number of fragments observed and analyze the differences between the larger fragments. This analysis will help identify the presence of key structural features within the molecule."
            answer = "Yes" if row[group] == 1 else "No"
            qclass = "MASS spectrum structure elucidation"
            results_df.loc[len(results_df)] = [row['Molecule Index'], row['SMILES'], qclass, row['Formula'], question, answer]

    results_df.to_csv(output_csv, index=False)
    print(f"MASS questions generated and saved to '{output_csv}'")

def generate_h_nmr_questions(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    # Placeholder for logic
    results_df = pd.DataFrame(columns=["Molecule Index", "SMILES", "cls", "Formula", "Question", "Answer"])
    results_df.to_csv(output_csv, index=False)
    print(f"H-NMR questions generated and saved to '{output_csv}'")

def generate_c_nmr_questions(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    # Placeholder for logic
    results_df = pd.DataFrame(columns=["Molecule Index", "SMILES", "cls", "Formula", "Question", "Answer"])
    results_df.to_csv(output_csv, index=False)
    print(f"C-NMR questions generated and saved to '{output_csv}'")

# Data Sampling Function
def sample_data(input_csv, output_prefix, ratios, total_samples=100, iterations=3):
    data = pd.read_csv(input_csv)

    for i in range(iterations):
        samples_per_class = {clss: int(total_samples * ratio) for clss, ratio in ratios.items()}
        sampled_data = pd.DataFrame()

        for clss, n_samples in samples_per_class.items():
            if clss in data['cls'].unique():
                sampled_class_data = data[data['cls'] == clss].sample(n=min(n_samples, len(data[data['cls'] == clss])))
                sampled_data = pd.concat([sampled_data, sampled_class_data])

        if len(sampled_data) < total_samples:
            remaining_needed = total_samples - len(sampled_data)
            remaining_data = data[~data.index.isin(sampled_data.index)]
            if not remaining_data.empty:
                additional_samples = remaining_data.sample(n=min(remaining_needed, len(remaining_data)), random_state=42)
                sampled_data = pd.concat([sampled_data, additional_samples])

        output_csv = f'{output_prefix}_{i}.csv'
        sampled_data.to_csv(output_csv, index=False)
        print(f"Sampled data saved to '{output_csv}'")

# Model Interaction Functions
def is_open_source(model_name):
    return not any(x in model_name.lower() for x in ['claude', 'gemini', 'gpt'])

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_prompt(model_name, image_path, prompt_text, sys_prompt, model=None, processor=None):
    if not os.path.exists(image_path):
         raise FileNotFoundError(f"Image not found at {image_path}")

    if 'instructBlip' in model_name:
        return analyze_image_with_prompt_instructBlip(model, processor, image_path, prompt_text, sys_prompt)
    elif 'llava' in model_name:
        return analyze_image_with_prompt_llava(model, processor, image_path, prompt_text, sys_prompt)
    elif 'Qwen' in model_name:
        return analyze_image_with_prompt_qwen(model, processor, image_path, prompt_text, sys_prompt)
    elif 'InternVL' in model_name:
        return analyze_image_with_prompt_internvl(model, processor, image_path, prompt_text, sys_prompt)
    elif 'gpt' in model_name.lower():
        return analyze_image_with_prompt_gpt(model_name, image_path, prompt_text, sys_prompt)
    elif 'claude' in model_name.lower():
        return analyze_image_with_prompt_claude(model_name, image_path, prompt_text, sys_prompt)
    elif 'gemini' in model_name.lower():
        raise NotImplementedError("Gemini model handler not implemented.")
    else:
        raise ValueError(f"Model '{model_name}' handler not implemented.")

def analyze_image_with_prompt_instructBlip(model, processor, image_path, prompt_text, sys_prompt):
    image = Image.open(image_path).convert("RGB")
    prompt = sys_prompt + prompt_text
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        do_sample=True,
        num_beams=5,
        max_length=512,
        min_length=4,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    generated_text = processor.decode(outputs[0], skip_special_tokens=True).strip()
    return generated_text

def analyze_image_with_prompt_llava(model, processor, image_path, prompt_text, sys_prompt):
    image = Image.open(image_path)
    prompt = sys_prompt + prompt_text
    inputs = processor(images=image, text=prompt, return_tensors='pt').to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=5,
        do_sample=True,
        temperature=0.7,
    )
    generated_text = processor.decode(outputs[0], skip_special_tokens=True).strip()
    return generated_text

def analyze_image_with_prompt_qwen(model, tokenizer, image_path, prompt_text, sys_prompt):
    raise NotImplementedError("Qwen model handler not implemented.")

def analyze_image_with_prompt_internvl(model, tokenizer, image_path, prompt_text, sys_prompt):
    raise NotImplementedError("InternVL model handler not implemented.")

def analyze_image_with_prompt_gpt(model_name, image_path, prompt_text, sys_prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
        
    base64_image = encode_image_to_base64(image_path)
    prompt = f"{sys_prompt}\n{prompt_text}\n[Image: data:image/png;base64,{base64_image}]"
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert organic chemist."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7,
    )
    generated_text = response['choices'][0]['message']['content'].strip()
    return generated_text

def analyze_image_with_prompt_claude(model_name, image_path, prompt_text, sys_prompt):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        
    base64_image = encode_image_to_base64(image_path)
    client = anthropic.Client(api_key)
    prompt = f"{anthropic.HUMAN_PROMPT}{sys_prompt}\n{prompt_text}\n[Image: data:image/png;base64,{base64_image}]{anthropic.AI_PROMPT}"
    response = client.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        max_tokens_to_sample=512,
        model=model_name,
        temperature=0.7,
    )
    generated_text = response['completion'].strip()
    return generated_text

# Evaluation Functions
def evaluate_responses(df, categories):
    results = {}
    for category in categories:
        category_data = df[df['cls'] == category]
        if category_data.empty:
            continue

        # Prepare answers
        answer = [1 if str(ans).strip().lower() == 'yes' else 0 for ans in category_data['Answer']]
        generated_response = [
            1 if 'yes' in str(ans).lower() else 0 if 'no' in str(ans).lower() else -1
            for ans in category_data['Generated Response']
        ]

        # Compute metrics
        accuracy = accuracy_score(answer, generated_response)
        f1 = f1_score(answer, generated_response, average='macro')
        results[category] = {'Accuracy': accuracy, 'F1 Score': f1}
    return results

def perform_evaluation(models, categories, data_prefix, iteration=3):
    for model in models:
        print(f"\nEvaluating model: {model}")
        f1_scores = {category: [] for category in categories}
        accuracies = {category: [] for category in categories}

        for i in range(iteration):
            input_csv = f'{data_prefix}_{i}.csv' # This assumes the prefix points to ground truth?
            # Actually, typically evaluation compares generated vs ground truth.
            # The generated CSV usually contains both.
            generated_file = f'{data_prefix}_{model}_generated_responses_{i}.csv'
            if not os.path.exists(generated_file):
                print(f"File not found: {generated_file}")
                continue
                
            generated_answers = pd.read_csv(generated_file)
            results = evaluate_responses(generated_answers, categories)

            for category in categories:
                if category in results:
                    f1_scores[category].append(results[category]['F1 Score'])
                    accuracies[category].append(results[category]['Accuracy'])

        for category in categories:
            f1_scores_arr = np.array(f1_scores[category])
            accuracies_arr = np.array(accuracies[category])
            if len(f1_scores_arr) > 0:
                print(f"\nCategory: {category}")
                print(f"F1 Score Mean: {f1_scores_arr.mean():.4f}")
                print(f"F1 Score Std: {f1_scores_arr.std():.4f}")
                print(f"Accuracy Mean: {accuracies_arr.mean():.4f}")
                print(f"Accuracy Std: {accuracies_arr.std():.4f}")

# Main Function with Argument Parsing
def main():
    parser = argparse.ArgumentParser(description='Spectrum Analysis using LLMs')
    parser.add_argument('--task', choices=['H-NMR', 'C-NMR', 'IR', 'MASS'], required=True, help='Task to perform')
    parser.add_argument('--action', choices=['generate_questions', 'sample_data', 'generate_responses', 'evaluate'], required=True, help='Action to perform')
    parser.add_argument('--input_csv', type=str, help='Input CSV file')
    parser.add_argument('--output_csv', type=str, help='Output CSV file')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations')
    parser.add_argument('--models', nargs='+', help='List of model names to use')
    args = parser.parse_args()

    # Define System Prompts
    SYS_PROMPTS = {
        'H-NMR': """As an expert organic chemist, your task is to analyze and determine the potential structures that can be derived from a given molecule's H-NMR spectrum image.
Utilize your knowledge to systematically explore and identify plausible structural configurations based on these spectrum images provided and answer the question.
Identify and list possible molecular fragments that match the spectral data and ensure the fragments are chemically feasible and consistent with the H-NMR data.
Analyze the problem step-by-step internally, but do not include the analysis in your output.
Respond with ONLY 'Yes' or 'No' to indicate whether the molecule could potentially contain the functional group.
Example output: Yes.""",
        'C-NMR': """As an expert organic chemist, your task is to analyze and determine the potential structures that can be derived from a given molecule's C-NMR spectrum image.
Utilize your knowledge to systematically explore and identify plausible structural configurations based on these spectrum images provided and answer the question.
Identify and list possible molecular fragments that match the spectral data and ensure the fragments are chemically feasible and consistent with the C-NMR data.
Analyze the problem step-by-step internally, but do not include the analysis in your output.
Respond with ONLY 'Yes' or 'No' to indicate whether the molecule could potentially contain the functional group.
Example output: Yes.""",
        'IR': """As an expert organic chemist, your task is to analyze and determine the potential structures that can be derived from a given molecular spectrum image.
Utilize your knowledge to systematically explore and identify plausible structural configurations based on the spectrum image provided and answer the question.
Please think step by step and provide the answer.""",
        'MASS': """As an expert organic chemist, your task is to analyze and determine the potential structures that can be derived from a given molecular MASS spectrum image.
Utilize your knowledge to systematically explore and identify plausible structural configurations based on the MASS spectrum image provided and answer the question.
Analyze the problem step-by-step internally, but do not include the analysis in your output. Provide only a very short answer with the exact result. Respond with 'Yes' or 'No' to the question.
Example output: Yes."""
    }

    if args.action == 'generate_questions':
        if not args.input_csv or not args.output_csv:
             print("Error: --input_csv and --output_csv are required for generate_questions")
             return
             
        if args.task == 'IR':
            generate_ir_questions(args.input_csv, args.output_csv)
        elif args.task == 'MASS':
            generate_mass_questions(args.input_csv, args.output_csv)
        elif args.task == 'H-NMR':
            generate_h_nmr_questions(args.input_csv, args.output_csv)
        elif args.task == 'C-NMR':
            generate_c_nmr_questions(args.input_csv, args.output_csv)
        else:
            print(f"Question generation not implemented for task '{args.task}'")
            
    elif args.action == 'sample_data':
        # Default logic if input_csv is not provided?
        # Ideally, user should provide input_csv.
        # But legacy code had hardcoded paths. We'll support args.
        
        if not args.input_csv:
             # Try to find default file based on task?
             # But better to enforce explicit input.
             print("Error: --input_csv is required for sample_data")
             return

        input_csv = args.input_csv
        # If output_csv is provided, use it as prefix (removing extension if needed)
        if args.output_csv:
             output_prefix = str(Path(args.output_csv).with_suffix(''))
        else:
             output_prefix = str(Path(args.input_csv).with_suffix('')) + "_sampled"
             
        ratios = {}
        # Define default ratios based on task if needed, or assume uniform if not specified.
        # The original code hardcoded specific ratios for specific class names.
        # We can try to infer or just use the passed file.
        # For this refactor, let's keep it simple: assume the user knows what they are doing or use uniform.
        # But to match legacy behavior, let's look at the task.
        
        if args.task == 'IR':
             ratios = {'IR spectrum peak analysis': 0.8, 'IR spectrum structure elucidation': 0.2}
        elif args.task == 'MASS':
             ratios = {'MASS spectrum structure elucidation': 1}
        elif args.task == 'H-NMR':
             ratios = {'H-NMR spectrum structure elucidation': 1}
        elif args.task == 'C-NMR':
             ratios = {'C-NMR spectrum structure elucidation': 1}
        else:
             # Default fallback: uniform over existing classes in input
             df = pd.read_csv(input_csv)
             classes = df['cls'].unique()
             ratios = {c: 1.0/len(classes) for c in classes}
             
        sample_data(input_csv, output_prefix, ratios, total_samples=100, iterations=args.iterations)
        
    elif args.action == 'generate_responses':
        if not args.models:
            print("Please provide a list of models using --models")
            return
        if not args.input_csv:
            print("Error: --input_csv is required (base path for iteration files, e.g. sampled_0.csv)")
            return

        # Expect input_csv to be the prefix or the exact file?
        # The loop iterates 0..iterations.
        # If input_csv is 'data_0.csv', we might deduce prefix 'data'.
        # Let's assume input_csv is the prefix used in sample_data output.
        # But typical usage: user passes one file. 
        # The original code iterated: input_csv = f'{data_prefix}_{i}.csv'
        
        # We will assume args.input_csv is the PREFIX.
        data_prefix = str(Path(args.input_csv).with_suffix(''))
        if data_prefix.endswith("_0"):
            data_prefix = data_prefix[:-2] # Strip _0

        sys_prompt = SYS_PROMPTS[args.task]
        
        for model_name in args.models:
            print(f"Processing model: {model_name}")
            model = processor = None
            
            # Load model (moved logic here or keep it)
            if is_open_source(model_name):
                model_path = get_model_path(model_name)
                if not model_path:
                    print(f"Warning: No path configured for {model_name}. Skipping.")
                    continue
                    
                if 'instructBlip' in model_name:
                    model = InstructBlipForConditionalGeneration.from_pretrained(
                        model_path, cache_dir=CACHE_DIR
                    ).to("cuda")
                    processor = InstructBlipProcessor.from_pretrained(
                        model_path, cache_dir=CACHE_DIR
                    )
                elif 'llava' in model_name:
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        cache_dir=CACHE_DIR
                    ).to("cuda")
                    processor = AutoProcessor.from_pretrained(model_path, cache_dir=CACHE_DIR)
                # ... Add other models ...
            
            for i in range(args.iterations):
                input_file = f'{data_prefix}_{i}.csv'
                if not os.path.exists(input_file):
                    print(f"Input file not found: {input_file}. Skipping iteration {i}")
                    continue
                    
                data = pd.read_csv(input_file)
                data_frame = pd.DataFrame(columns=["Question", "cls", "Answer", "Generated Response"])

                for _, row in data.iterrows():
                    try:
                        image_path = get_image_path(args.task, row['Molecule Index'])
                        
                        generated_response = analyze_image_with_prompt(
                            model_name,
                            image_path,
                            row['Question'],
                            sys_prompt=sys_prompt,
                            model=model,
                            processor=processor,
                        )
                        print(f"Generated Response: {generated_response}")
                        data_frame.loc[len(data_frame)] = [
                            row['Question'],
                            row['cls'],
                            row['Answer'],
                            generated_response,
                        ]

                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue

                output_file = f'{data_prefix}_{model_name}_generated_responses_{i}.csv'
                data_frame.to_csv(output_file, index=False)
                print(f"Generated responses saved to '{output_file}'")

            # Clean up
            if model is not None:
                del model
            if processor is not None:
                del processor
            torch.cuda.empty_cache()
            
    elif args.action == 'evaluate':
        if not args.input_csv:
             print("Error: --input_csv is required (prefix)")
             return
             
        data_prefix = str(Path(args.input_csv).with_suffix(''))
        if data_prefix.endswith("_0"):
            data_prefix = data_prefix[:-2]

        categories = []
        if args.task == 'IR':
            categories = ['IR spectrum peak analysis', 'IR spectrum structure elucidation']
        elif args.task == 'MASS':
            categories = ['MASS spectrum structure elucidation']
        elif args.task == 'H-NMR':
            categories = ['H-NMR spectrum structure elucidation']
        elif args.task == 'C-NMR':
            categories = ['C-NMR spectrum structure elucidation']
        else:
            print(f"Evaluation not implemented for task '{args.task}'")
            return

        perform_evaluation(
            models=args.models,
            categories=categories,
            data_prefix=data_prefix,
            iteration=args.iterations
        )
    else:
        print(f"Action '{args.action}' not recognized")

if __name__ == "__main__":
    main()
