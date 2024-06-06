import os

import numpy as np

from sentence_transformers import CrossEncoder
os.environ['TRANSFORMERS_CACHE'] = '/data/farima/huggingface_hub/'
import argparse
import pandas as pd
import json

directory_path = {
    'nq': 'NQ',
    'trivia_qa': 'TriviaQA',
    'sciq': 'SciQ',
    'tqa_mc2' : 'TQA'
}

HF_NAMES = {
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'gpt2_xl': 'gpt2-xl',
    'gpt2_large': 'gpt2-large'
}


parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, default='llama2_chat_7B', choices=HF_NAMES.keys(), help='model name')
parser.add_argument('--dataset_name', type=str, default='trivia_qa', help='feature bank for training probes')
parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
parser.add_argument('--num_samples', type=int, default=3000, help='number of samples for info collection')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
args = parser.parse_args()


# NLI
model = CrossEncoder('cross-encoder/nli-deberta-v3-large')

def entailment_evaluation(row):
	# print(row)
	correct_answers = str(row['Answer']).split('; ')
	prediction = row[args.model_name]
	question = row['Question']
	refs = [f'Q: {question}? A: {ref}' for ref in correct_answers]
	pred = f'Q: {question}? A: {prediction}'
	scores = model.predict([(ref, pred) for ref in refs])
	
	# Convert scores to labels
	label_mapping = ['contradiction', 'entailment', 'neutral'] # future note: neutral can be interpreted as irrelevant
	labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
	if 'entailment' in labels: 
		return 1
	else: 
		return 0
	
# Part1: evaluate
def annotate_(alpha, mode, model_name, dataset_name):
	print(f"annotating at alpha {alpha}...")
	data_dir = directory_path[dataset_name]
	data_dir = f'{data_dir}/results/{model_name}/{mode}/files'
	filename = f"alpha_{int(alpha)}" if alpha != 0 else "baseline"
	data_file_path = f"{data_dir}/{filename}.csv"
	df = pd.read_csv(data_file_path)
	column_name = "entailment"
	df[column_name] = df.apply(entailment_evaluation, axis=1)
	print(f"accuracy at alpha {alpha}: {df[column_name].mean()}")
	df.to_csv(f'{data_dir}/{filename}_annotated.csv', index=False)

def prepare_data_(mode, dataset_name):
	data_dir = directory_path[dataset_name]
	# data_dir = f'{data_dir}/results/{args.model_name}/{args.mode}/files/'
	data_dir = f'{data_dir}/results/{args.model_name}/{mode}/files/'


	data_file_paths = [data_dir + "alpha_5_annotated.csv",
					data_dir + "alpha_10_annotated.csv",
					data_dir + "alpha_15_annotated.csv",
					data_dir + "alpha_20_annotated.csv",
					data_dir + "alpha_25_annotated.csv",
					data_dir + "alpha_30_annotated.csv",
					data_dir + "alpha_35_annotated.csv"]


	# df = pd.read_csv(data_file_paths[0])
	# df = df[['Question', 'Answer']]
	# for i, path in enumerate(data_file_paths):
	# 	df[f'alpha_{(i+1)*5}'] = pd.read_csv(path)[args.model_name] 
	# 	df[f'conf_{(i+1)*5}'] = pd.read_csv(path)['mean_probs'] 
	# 	df[f'label_{(i+1)*5}'] = pd.read_csv(path)['entailment']
	# 	df[f'tokens_{(i+1)*5}'] = pd.read_csv(path)['tokens']
	# 	df[f'token_probs_{(i+1)*5}'] = pd.read_csv(path)['token_probs']
	
	raw_file_paths = [data_dir + "alpha_5.csv",
					data_dir + "alpha_10.csv",
					data_dir + "alpha_15.csv",
					data_dir + "alpha_20.csv",
					data_dir + "alpha_25.csv",
					data_dir + "alpha_30.csv",
					data_dir + "alpha_35.csv"]
	df = pd.read_csv(data_file_paths[0])
	df = df[['Question', 'Answer']]
	for i, (raw_path, path) in enumerate(zip(raw_file_paths, data_file_paths)):
		df[f'alpha_{(i+1)*5}'] = pd.read_csv(path)[args.model_name] 
		df[f'conf_{(i+1)*5}'] = pd.read_csv(raw_path)['mean_probs'] 
		df[f'label_{(i+1)*5}'] = pd.read_csv(path)['entailment']
		df[f'tokens_{(i+1)*5}'] = pd.read_csv(raw_path)['tokens']
		df[f'token_probs_{(i+1)*5}'] = pd.read_csv(raw_path)['token_probs']

	df = df[
		# ['Question', 'answer', 'false_answer'] +
		['Question', 'Answer'] +
		[f'alpha_{i*5}' for i in range(1,8)] + 
		[f'conf_{i*5}' for i in range(1,8)] +
		[f'label_{i*5}' for i in range(1,8)] +
		[f'tokens_{i*5}' for i in range(1,8)] + 
		[f'token_probs_{i*5}' for i in range(1,8)]
		] 

	df.to_csv(f"{data_dir}classification_{mode}_data_all.csv", index=False)

	

# Part1: evaluate
def annotate():
	print(f"annotating at alpha {args.alpha}...")
	data_dir = directory_path[args.dataset_name]
	data_dir = f'{data_dir}/results/{args.model_name}/{args.mode}/min_files' #files
	filename = f"alpha_{int(args.alpha)}" if args.alpha != 0 else "baseline"
	data_file_path = f"{data_dir}/{filename}.csv"
	df = pd.read_csv(data_file_path)
	column_name = "entailment"
	df[column_name] = df.apply(entailment_evaluation, axis=1)
	print(f"accuracy at alpha {args.alpha}: {df[column_name].mean()}")
	df.to_csv(f'{data_dir}/{filename}_annotated.csv', index=False)

def prepare_data():
	data_dir = directory_path[args.dataset_name]
	# data_dir = f'{data_dir}/results/{args.model_name}/{args.mode}/files/'
	data_dir = f'{data_dir}/results/{args.model_name}/{args.mode}/min_files/'


	data_file_paths = [data_dir + "alpha_5_annotated.csv",
					data_dir + "alpha_10_annotated.csv",
					data_dir + "alpha_15_annotated.csv",
					data_dir + "alpha_20_annotated.csv",
					data_dir + "alpha_25_annotated.csv"]


	df = pd.read_csv(data_file_paths[0])
	df = df[['Question', 'Answer']]
	for i, path in enumerate(data_file_paths):
		df[f'alpha_{(i+1)*5}'] = pd.read_csv(path)[args.model_name] 
		df[f'conf_{(i+1)*5}'] = pd.read_csv(path)['mean_probs'] 
		df[f'label_{(i+1)*5}'] = pd.read_csv(path)['entailment']
		df[f'tokens_{(i+1)*5}'] = pd.read_csv(path)['tokens']
		df[f'token_probs_{(i+1)*5}'] = pd.read_csv(path)['token_probs']

	df = df[
		# ['Question', 'answer', 'false_answer'] +
		['Question', 'Answer'] +
		[f'alpha_{i*5}' for i in range(1,6)] + 
		[f'conf_{i*5}' for i in range(1,6)] +
		[f'label_{i*5}' for i in range(1,6)] +
		[f'tokens_{i*5}' for i in range(1,6)] + 
		[f'token_probs_{i*5}' for i in range(1,6)]
		] 

	df.to_csv(f"{data_dir}classification_{args.mode}_data.csv", index=False)

if __name__ == '__main__':
	# annotate()
	# for dataset in ['nq', 'sciq', 'trivia_qa']:
	# 	for mode in ['test', 'train']:
	# 		for alpha in [30, 35]:
	# 			if dataset == "nq" and alpha == 35 and mode == "test":
	# 				continue
	# 			annotate_(alpha, mode, args.model_name, dataset)
	# prepare_data()
	# for dataset in ['nq', 'sciq', 'trivia_qa', 'tqa_mc2']:
	for dataset in ['nq']:
		for mode in ['test']: #, 'train']:
			annotate_(35, mode, args.model_name, dataset)
			prepare_data_(mode, dataset)
