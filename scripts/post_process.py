import argparse
from math import sqrt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
device = torch.device('cpu')

seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)

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
parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'val'])
parser.add_argument('--pre_train_epochs', type=int, default=20)
parser.add_argument('--patiance', type=int, default=10)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lstm_direction', type=str, default="uni", choices=["uni", "bi"])
parser.add_argument('--k', default=5, type=int)
args = parser.parse_args()

directory_path = {
    'nq': 'NQ',
    'trivia_qa': 'TriviaQA',
    'sciq': 'SciQ',
    'tqa_mc2' : 'TQA'
}
DATA_DIR = f'datasets_results/{directory_path[args.dataset_name]}'


def f1_calc(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def calculate_final_accuracy():
    data_dir = f'../{DATA_DIR}/results/{args.model_name}'
    for direction in ["uni", "bi"]:
        print(direction)
        try:
            test_file_path = f'{data_dir}/test_outputs.csv'
            dataset = pd.read_csv(test_file_path)
        except:
            continue
        final_label = 0
        truth_label = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        num_have_correct_answer = len(dataset)
        print("num_have_correct_answer: ", num_have_correct_answer, len(dataset))
        for i, row in dataset.iterrows():
            if "I have no comment." in str(row["Answer"]).strip(): # special case of TruthfulQA'
                num_have_correct_answer -= 1
            
            # convert to np arrays
            alpha_values = [i*5 for i in range(1, args.k+1)]
            alpha_values = np.array(alpha_values)
            confidence_values = [round(row[f'conf_{i*5}'], 4) for i in range(1, args.k+1)]
            confidence_values = np.array(confidence_values)
            tokens = [str(row[f'alpha_{i*5}']) for i in range(1, args.k+1)]
            tokens = np.array(tokens)
            
            preds = [row[f'pred_{(i)*5}'] for i in range(1, args.k+1)]
            preds_array = np.array(preds)

            has_correct = True if any([row[f'label_{i*5}'] for i in range(1, args.k+1)]) == 1 else False

            if np.all(np.equal(preds_array, 0)):
                truth_label += 1
                if not has_correct:
                    final_label += 1
                    tn += 1
                else:
                    fn += 1
            else: 
                correct_preds_indices = np.where(preds_array==1)[0]
                
                max_confidence_index = np.argmax(confidence_values[correct_preds_indices])
                selected_alpha = alpha_values[correct_preds_indices[max_confidence_index]]
                
                if "I have no comment" in str(tokens[int(selected_alpha/5)-1]).strip():
                    truth_label += 1
                elif row[f'label_{selected_alpha}'] == 1:
                    tp += 1
                    final_label += 1
                    truth_label += 1
                else: 
                    fp += 1
        p,r,f1 = f1_calc(tp, fp, fn)
        p,r,f1 = round(p*100, 2), round(r*100, 2), round(f1*100, 2)
        accuracy = 100*tp/num_have_correct_answer
        truthfulness = 100*truth_label/len(dataset)
        print("Accuracy (Factuality): ", accuracy)
        print("Truthfulness: ", round(100*truth_label/len(dataset), 2))  
        print("TA: ", round(sqrt(accuracy*truthfulness), 2)) 
        break
 
if __name__ == "__main__":
    calculate_final_accuracy()