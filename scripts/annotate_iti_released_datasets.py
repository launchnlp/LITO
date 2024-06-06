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
parser.add_argument('--dataset_name', type=str, default='nq', help='from trivia_qa and nq')
args = parser.parse_args()


# NLI
model = CrossEncoder('cross-encoder/nli-deberta-v3-large')

def entailment_evaluation(row):
	# print(row)
	correct_answers = str(row['Answer']).split('; ')
	false_answer = str(row['false_answer'])
	question = row['Question']
	refs = [f'Q: {question}? A: {ref}' for ref in correct_answers]
	pred = f'Q: {question}? A: {false_answer}'
	scores = model.predict([(ref, pred) for ref in refs])
	
	# Convert scores to labels
	label_mapping = ['contradiction', 'entailment', 'neutral'] # future note: neutral can be interpreted as irrelevant
	labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
	if 'entailment' in labels: 
		return 1
	else: 
		return 0
	
def is_sentence(false_answer):
	# print(row)
	false_answer = str(false_answer).strip()
	false_answer = false_answer.replace('"', '').replace('"""', '')
	# print(false_answer, false_answer.endswith("."))
	if false_answer.endswith("."):
		return 1
	else: 
		return 0
	

# Part1: evaluate
def annotate():
	print(f"annotating...")
	data_dir = directory_path[args.dataset_name]
	filename = "train"
	data_file_path = f"{data_dir}/splits/{filename}.csv"
	df = pd.read_csv(data_file_path)
	
	column_name = "entailment"
	df[column_name] = df.apply(entailment_evaluation, axis=1)
	print(f"overlap ratio: {df[column_name].mean()}") 

	column_name = "is_sentence"
	df[column_name] = df["false_answer"].apply(is_sentence)
	print(f"sentence ratio: {df[column_name].mean()}") 
	df.to_csv(f"check_{args.dataset_name}.csv", index=False)
 

if __name__ == '__main__':
	annotate()