import torch
from einops import rearrange
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = 'path/to/huggingface_hub'
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset

import sys
sys.path.append('../')
from utils import alt_evaluate, get_interventions_dict, get_top_heads, get_separated_activations_sa, get_com_directions
import llama
import gpt2
import warnings

warnings.filterwarnings("ignore")

HF_NAMES = {
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'gpt2_xl': 'gpt2-xl',
    'gpt2_large': 'gpt2-large'
}

directory_path = {
    'nq': 'NQ',
    'trivia_qa': 'TriviaQA',
    'sciq': 'SciQ',
    'tqa_mc2' : 'TQA'
}

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, default='llama2_chat_7B', choices=HF_NAMES.keys(), help='model name')
parser.add_argument('--dataset_name', type=str, default='nq', help='feature bank for training probes')
parser.add_argument('--activations_dataset', type=str, help='feature bank for calculating std along direction')
parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
parser.add_argument("--num_fold", type=int, default=3, help="number of folds")
parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
parser.add_argument('--device', type=int, default=0, help='device')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--num_samples', type=int, default=3000, help='number of samples for info collection')
parser.add_argument('--no_intervention', type=bool, default=False)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--create_training_data', type=bool, default=False)
args = parser.parse_args()

torch.cuda.empty_cache()
# set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_interventions(train_path, train_dataset, model, llm_family):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    features_path = os.path.join(train_path, "features")
    head_wise_activations = np.load(os.path.join(features_path, f"{args.model_name}_{args.dataset_name}_head_wise.npy"))
    labels = np.load(os.path.join(features_path, f"{args.model_name}_{args.dataset_name}_labels.npy"))
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h=num_heads)

    # tuning dataset: no labels used, just to get std of activations along the direction
    print(f"activation dataset: {args.activations_dataset}")
    tuning_activations = head_wise_activations

    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations_sa(train_dataset, labels, head_wise_activations, dataset_name=args.dataset_name)

    train_idxs = np.array([i for i in range(0, len(train_dataset))])
    train_set_idxs = train_idxs[:int(len(train_dataset)*(1-args.val_ratio))]
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

    # get directions
    if args.use_center_of_mass:
        print(f"num of hidden layers: {num_layers}, num of attention heads: {num_heads}")
        com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
    else:
        com_directions = None
    top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)
    print("Heads intervened: ", sorted(top_heads))

    interventions = get_interventions_dict(top_heads, probes, tuning_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions, family=llm_family)
    return interventions


def main(model, tokenizer, dir_path, llm_family):
    data_splits = os.path.join(dir_path, 'splits')
    dataset = pd.read_csv(os.path.join(data_splits, "train.csv"))
    
    num_heads = model.config.num_attention_heads
    interventions = get_interventions(dir_path, dataset, model, llm_family)
    def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'):
        head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
        for head, direction, proj_val_std in interventions[layer_name]:
            if args.model_name == "llama2_chat_13B":
                direction_to_add = torch.tensor(direction).to(model.device).half()
            else: 
                direction_to_add = torch.tensor(direction).to(model.device)
            if start_edit_location == 'lt':
                head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
            else:
                head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
        head_output = rearrange(head_output, 'b s h d -> b s (h d)')
        return head_output
    
    if args.no_intervention:
        filename = f'baseline'
        interventions = {}
        lt_modulated_vector_add = None

    else: 
        filename = f'alpha_{int(args.alpha)}'

    if args.use_center_of_mass:
        filename += '_com'
    
    if args.mode == 'train':
        input_test_file = f'{dir_path}/classification_dataset/classification_dataset.csv'
    elif args.mode == 'test':
        input_test_file = f'{dir_path}/splits/test.csv'
    
    
    alt_evaluate(
        {args.model_name: model}, 
        input_test_file,
        f'{dir_path}/results/{args.model_name}/{args.mode}/files/{filename}.csv',
        mean_hidden_states_path=f'{dir_path}/results/{args.model_name}/{args.mode}/hidden_states/{filename}_mean_hidden_states.pt',
        tokenizer=tokenizer,
        device=device, 
        preset=args.dataset_name,
        interventions=interventions, 
        intervention_fn=lt_modulated_vector_add, 
        instruction_prompt = args.instruction_prompt,
        # verbose=True
    )

    
def build_dataset_for_classififcation(base_dir, dataset_length):
    # os.mkdir(f'{dir_path}') if not os.path.exists(f'{dir_path}') else print(f"Folder '{dir_path}' already exists.")
    # dir_path = os.path.join(path, 'short_prompt')
    if args.dataset_name == 'nq':
        dir_path = f'{base_dir}/NQ'
        dir_path = os.path.join(dir_path, 'classification_dataset')
        dataset = load_dataset("nq_open", split="train")
        sampled_daataset = dataset.shuffle(seed=args.seed)
        sampled_daataset = sampled_daataset.select([i for i in range (0, dataset_length)])
        sampled_df = sampled_daataset.to_pandas()
        sampled_df['answer'] = sampled_df['answer'].apply(lambda x: "; ".join(x.astype(str)))
        sampled_df = sampled_df.rename(columns={'question': 'Question', 'answer': 'Answer'})
    
    if args.dataset_name == 'trivia_qa':
        dir_path = f'{base_dir}/TriviaQA'
        dir_path = os.path.join(dir_path, 'classification_dataset')
        dataset = load_dataset("trivia_qa", "rc", split="train")
        dataset = dataset.select_columns(['question', 'answer'])
        sampled_daataset = dataset.shuffle(seed=args.seed)
        sampled_daataset = sampled_daataset.select([i for i in range (0, dataset_length)])
        sampled_df = sampled_daataset.to_pandas()
        sampled_df['answer'] = sampled_df['answer'].apply(lambda x: "; ".join([ans.strip() for ans in x['aliases']]))
        sampled_df = sampled_df.rename(columns={'question': 'Question', 'answer': 'Answer'})
    
    if args.dataset_name == 'sciq':
        dir_path = f'{base_dir}/SciQ'
        dir_path = os.path.join(dir_path, 'classification_dataset')
        dataset = load_dataset("sciq", split="train")
        sampled_daataset = dataset.shuffle(seed=args.seed)
        sampled_daataset = sampled_daataset.select([i for i in range (0, dataset_length)])
        sampled_df = sampled_daataset.to_pandas()
        func = lambda row: "; ".join([ans.strip() for ans in [row["distractor1"], row["distractor2"], row["distractor3"]]])
        sampled_df['false_answer'] = sampled_df.apply(func, axis=1)
        sampled_df = sampled_df.rename(columns={'question': 'Question', 'correct_answer': 'Answer'})
        sampled_df = sampled_df[['Question', 'Answer', 'false_answer']]
    sampled_df.to_csv(os.path.join(dir_path, f'classification_dataset.csv'), index=False)

    # for truthfulQA, we use the same data used for training iti.
    if args.dataset_name == 'tqa_mc2':
        dir_path = f'{base_dir}/TQA'
        input_path = os.path.join(dir_path, 'splits')
        output_path = os.path.join(dir_path, 'classification_dataset')

        dataset = pd.read_csv(os.path.join(input_path, "train.csv"))
        dataset.to_csv(output_path, index=False)


if __name__ == "__main__":
    dir_path = "datasets_results"
    
    # create dataset - run once
    if args.create_training_data:
        build_dataset_for_classififcation(dir_path, args.num_samples)
    
    ## create model
    model_name = HF_NAMES[args.model_name]  
    if args.model_name in ['vicuna_7B', 'llama2_chat_7B', 'llama2_chat_13B']:
        print(f"loading {args.model_name}...")
        llm_family = 'llama'
        # llama family
        tokenizer = llama.LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        model = llama.LlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float32, device_map="auto")
        # model.to(device)

    elif args.model_name in ['gpt2_large', 'gpt2_xl']:
        print(f"loading {args.model_name}...")
        llm_family = 'gpt'
        # gpt family: 
        tokenizer = gpt2.GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        model = gpt2.GPT2LMHeadModel.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float32, device_map="auto")
        model.to(device)

    dir_path += directory_path[args.dataset_name] # e.g., 'TriviaQA'
    main(model, tokenizer, dir_path, llm_family)
        
    
