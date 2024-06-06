import os
os.environ['HF_HOME'] = '/data/farima/huggingface_hub/'

import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from utils import get_llama_activations_bau, get_gpt2_activations_bau
from utils import load_triviaqa, tokenized_trivia_qa, split_dataset
from utils import tokenized_nq, tokenized_trivia_qa, tokenized_sciq, tokenized_truthfulqa
import llama
import gpt2
import argparse


import random
seed = 42
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, default='llama2_chat_7B')
parser.add_argument('dataset_name', type=str, default='nq')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--min_positive_samples', type=int, default=2)
args = parser.parse_args()

directory_path = {
    'nq': 'NQ',
    'trivia_qa': 'TriviaQA',
    'sciq': 'SciQ',
    'tqa_mc2' : 'TQA'
}

tokenization_function = {
    'nq': tokenized_nq,
    'trivia_qa': tokenized_trivia_qa,
    'sciq': tokenized_sciq,
    'tqa_mc2' : tokenized_truthfulqa
}

HF_NAMES = {
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'gpt2_xl': 'gpt2-xl',
    'gpt2_large': 'gpt2-large'
}


def main(data_dir = None): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "nq", "trivia_qa", "sciq". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for 5 LMs. 
    """
    os.mkdir(f'{data_dir}/features') if not os.path.exists(f'{data_dir}/features') else print(f"Folder '{data_dir}/features' already exists.")
    MODEL = HF_NAMES[args.model_name]  
    if args.model_name in ['vicuna_7B', 'llama2_chat_7B', 'llama2_chat_13B']:
        # llama family
        tokenizer = llama.LlamaTokenizer.from_pretrained(MODEL, token="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")
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        model = llama.LlamaForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float32, device_map="auto", token="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")
        # model.to(device)
        get_activations_func = get_llama_activations_bau

    elif args.model_name in ['gpt2_large', 'gpt2_xl']:
        # gpt family: 
        tokenizer = gpt2.GPT2Tokenizer.from_pretrained(MODEL)
        model = gpt2.GPT2LMHeadModel.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float32, device_map="auto")
        # model.to(device)
        get_activations_func = get_gpt2_activations_bau
    
    else: 
        print("model is not yet implemented.")
    
    
    formatter = tokenization_function[args.dataset_name]
    train_data_dir = os.path.join(data_dir, "splits")
    train_data_path = os.path.join(train_data_dir, f"train.csv")
    train_data = pd.read_csv(train_data_path)
    print(train_data_path, len(train_data))

    print("Tokenizing prompts")
    prompts, labels = formatter(train_data, tokenizer, seed=seed, min_pos_samples=args.min_positive_samples)

    all_head_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        head_wise_activations = get_activations_func(model, prompt, device)
        all_head_wise_activations.append(head_wise_activations[:, -1, :])

    print("Saving labels")
    np.save(f'{data_dir}/features/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving head wise activations")
    np.save(f'{data_dir}/features/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)
        


if __name__ == '__main__':

    data_dir = f"datasets_results/{directory_path[args.dataset_name]}"

    # dataset_creation - run once / studied dataset splits are available.
    # dataset = load_triviaqa(seed=seed)
    # split_dataset(dataset, data_dir, num_fold=3)
    
    main(data_dir)