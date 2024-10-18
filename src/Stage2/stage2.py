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

# Set Cache Directory for Models
cache_dir = "./cache"

# Set API Keys (Replace with your actual API keys)
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
os.environ['ANTHROPIC_API_KEY'] = 'your_anthropic_api_key'

# Define Model Paths (Replace with your actual model paths)
model_paths = {
    'llava': '/path/to/llava-model',
    'instructBlip-7B': '/path/to/instructBlip-7B-model',
    'instructBlip-13B': '/path/to/instructBlip-13B-model',
    'Qwen-VL-Chat': '/path/to/Qwen-VL-Chat-model',
    'InternVL': '/path/to/InternVL-model',
    # API-based models don't require local paths
    'gpt-4': 'gpt-4',
    'gpt-3.5-turbo': 'gpt-3.5-turbo',
    'claude-v1': 'claude-v1',
    'gemini': 'gemini',
    # Add other models as needed
}

# Define Functional Groups
functional_groups = [
    "Alcohol", "Phenol", "Ketone", "Acid", "Aldehyde", "Ester", "Nitrile",
    "Isocyanate", "Amine", "Ether", "Sulfide", "Halide"
]

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
    columns = ["Molecule Index", "SMILES", "cls", "Formula", "Question", "Answer"]
    results_df = pd.DataFrame(columns=columns)

    # Assuming H-NMR specific questions are generated here
    # Implement your H-NMR question generation logic

    results_df.to_csv(output_csv, index=False)
    print(f"H-NMR questions generated and saved to '{output_csv}'")

def generate_c_nmr_questions(input_csv, output_csv):
    data = pd.read_csv(input_csv)
    columns = ["Molecule Index", "SMILES", "cls", "Formula", "Question", "Answer"]
    results_df = pd.DataFrame(columns=columns)

    # Assuming C-NMR specific questions are generated here
    # Implement your C-NMR question generation logic

    results_df.to_csv(output_csv, index=False)
    print(f"C-NMR questions generated and saved to '{output_csv}'")

# Data Sampling Function
def sample_data(input_csv, output_prefix, ratios, total_samples=100, iterations=3):
    data = pd.read_csv(input_csv)

    for i in range(iterations):
        samples_per_class = {clss: int(total_samples * ratio) for clss, ratio in ratios.items()}
        sampled_data = pd.DataFrame()

        for clss, n_samples in samples_per_class.items():
            sampled_class_data = data[data['cls'] == clss].sample(n=n_samples)
            sampled_data = pd.concat([sampled_data, sampled_class_data])

        if len(sampled_data) < total_samples:
            additional_samples = data[~data.index.isin(sampled_data.index)].sample(n=total_samples - len(sampled_data), random_state=42)
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
        # Implement analyze_image_with_prompt_gemini if needed
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
    # Implement Qwen model handler if applicable
    raise NotImplementedError("Qwen model handler not implemented.")

def analyze_image_with_prompt_internvl(model, tokenizer, image_path, prompt_text, sys_prompt):
    # Implement InternVL model handler if applicable
    raise NotImplementedError("InternVL model handler not implemented.")

def analyze_image_with_prompt_gpt(model_name, image_path, prompt_text, sys_prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
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
    base64_image = encode_image_to_base64(image_path)
    client = anthropic.Client(os.getenv("ANTHROPIC_API_KEY"))
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

        # Prepare answers
        answer = [1 if ans.strip().lower() == 'yes' else 0 for ans in category_data['Answer']]
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
            input_csv = f'{data_prefix}_{i}.csv'
            generated_answers = pd.read_csv(f'{data_prefix}_{model}_generated_responses_{i}.csv')
            results = evaluate_responses(generated_answers, categories)

            for category in categories:
                f1_scores[category].append(results[category]['F1 Score'])
                accuracies[category].append(results[category]['Accuracy'])

        for category in categories:
            f1_scores_arr = np.array(f1_scores[category])
            accuracies_arr = np.array(accuracies[category])
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

    sys_prompt = SYS_PROMPTS[args.task]

    # Paths
    data_dir = "./data/mol_figures/step2/"
    os.makedirs(data_dir, exist_ok=True)

    if args.action == 'generate_questions':
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
        ratios = {}
        if args.task == 'IR':
            ratios = {
                'IR spectrum peak analysis': 0.8,
                'IR spectrum structure elucidation': 0.2,
            }
            input_csv = f"{data_dir}IR_questions.csv"
            output_prefix = f"{data_dir}IR_sampled_questions_answers"
        elif args.task == 'MASS':
            ratios = {
                'MASS spectrum structure elucidation': 1,
            }
            input_csv = f"{data_dir}MASS_questions.csv"
            output_prefix = f"{data_dir}MASS_sampled_questions_answers"
        elif args.task == 'H-NMR':
            ratios = {
                'H-NMR spectrum structure elucidation': 1,
            }
            input_csv = f"{data_dir}H-NMR_questions.csv"
            output_prefix = f"{data_dir}H-NMR_sampled_questions_answers"
        elif args.task == 'C-NMR':
            ratios = {
                'C-NMR spectrum structure elucidation': 1,
            }
            input_csv = f"{data_dir}C-NMR_questions.csv"
            output_prefix = f"{data_dir}C-NMR_sampled_questions_answers"
        else:
            print(f"Data sampling not implemented for task '{args.task}'")
            return

        sample_data(input_csv, output_prefix, ratios, total_samples=100, iterations=args.iterations)
    elif args.action == 'generate_responses':
        if not args.models:
            print("Please provide a list of models using --models")
            return
        data_prefix = f"{data_dir}{args.task}_sampled_questions_answers"
        sys_prompt = SYS_PROMPTS[args.task]
        for model_name in args.models:
            print(f"Processing model: {model_name}")
            model = processor = None

            if is_open_source(model_name):
                if 'instructBlip' in model_name:
                    model = InstructBlipForConditionalGeneration.from_pretrained(
                        model_paths[model_name], cache_dir=cache_dir
                    ).to("cuda")
                    processor = InstructBlipProcessor.from_pretrained(
                        model_paths[model_name], cache_dir=cache_dir
                    )
                elif 'llava' in model_name:
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_paths[model_name],
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        cache_dir=cache_dir
                    ).to("cuda")
                    processor = AutoProcessor.from_pretrained(model_paths[model_name], cache_dir=cache_dir)
                elif 'Qwen' in model_name:
                    # Implement Qwen model loading if applicable
                    raise NotImplementedError("Qwen model loading not implemented.")
                elif 'InternVL' in model_name:
                    # Implement InternVL model loading if applicable
                    raise NotImplementedError("InternVL model loading not implemented.")
                else:
                    raise ValueError(f"Open-source model '{model_name}' handler not implemented.")
            else:
                # For API-based models, no local model loading is required
                pass

            for i in range(args.iterations):
                data_frame = pd.DataFrame(columns=["Question", "cls", "Answer", "Generated Response"])
                input_csv = f'{data_prefix}_{i}.csv'
                data = pd.read_csv(input_csv)

                for _, row in data.iterrows():
                    try:
                        index_parts = row['Molecule Index'].split('_')
                        index_parts = [part.strip() for part in index_parts]
                        image_suffix = {
                            'IR': '_1.png',
                            'MASS': '_2.png',
                            'C-NMR': '_3.png',
                            'H-NMR': '_4.png'
                        }[args.task]
                        image_path = f'./data/mol_figures/{index_parts[0]}th_specs/Problem {index_parts[1]}{image_suffix}'

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

                output_csv = f'{data_prefix}_{model_name}_generated_responses_{i}.csv'
                data_frame.to_csv(output_csv, index=False)
                print(f"Generated responses saved to '{output_csv}'")

            # Clean up
            if model is not None:
                del model
            if processor is not None:
                del processor
            torch.cuda.empty_cache()
    elif args.action == 'evaluate':
        categories = []
        if args.task == 'IR':
            categories = ['IR spectrum peak analysis', 'IR spectrum structure elucidation']
            data_prefix = f"{data_dir}IR_sampled_questions_answers"
        elif args.task == 'MASS':
            categories = ['MASS spectrum structure elucidation']
            data_prefix = f"{data_dir}MASS_sampled_questions_answers"
        elif args.task == 'H-NMR':
            categories = ['H-NMR spectrum structure elucidation']
            data_prefix = f"{data_dir}H-NMR_sampled_questions_answers"
        elif args.task == 'C-NMR':
            categories = ['C-NMR spectrum structure elucidation']
            data_prefix = f"{data_dir}C-NMR_sampled_questions_answers"
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