# baseline implementation
from math import sqrt
import numpy as np
import argparse
import pandas as pd

HF_NAMES = {
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'gpt2_xl': 'gpt2-xl',
    'gpt2_large': 'gpt2-large'
}

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, default='llama2_chat_7B', choices=HF_NAMES.keys(), help='model name')
parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--pre_train_epochs', type=int, default=10)
parser.add_argument('--patiance', type=int, default=10)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

directory_path = {
    'nq': 'NQ',
    'trivia_qa': 'TriviaQA',
    'sciq': 'SciQ',
    'tqa_mc2' : 'TQA',
    'xin': 'XIN'
}

data_dir = directory_path[args.dataset_name]
data_file = f'../{data_dir}/results/{args.model_name}/test/files/classification_test_data.csv'


def f1_calc(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def get_base_results():
    # no intervention:
    base_df = pd.read_csv(f'../{data_dir}/results/{args.model_name}/test/files/baseline_annotated.csv')
    num_have_correct_answer = len(base_df)
    column_name = "entailment"
    final_label = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, row in base_df.iterrows():
        if "I have no comment." in str(row["Answer"]).strip(): # special case of TruthfulQA'
            num_have_correct_answer -= 1
        if "I have no comment" in str(row[args.model_name]).strip():
            final_label += 1
            tn += 1 
        else:
            if row[column_name] == 1:
                tp += 1
                final_label += 1
            else:
                fp += 1
    p,r,f1 = f1_calc(tp, fp, fn)
    p,r,f1 = round(p*100, 2), round(r*100, 2), round(f1*100, 2)
    print(f"no intervention: Truth: {round(final_label/len(base_df)*100, 2)}, Acc: {round(100*tp/num_have_correct_answer, 2)}")


    df = pd.read_csv(data_file)
    for i in range(1 , 6): 
        column_name = f"alpha_{i*5}"
        label_column = f"label_{i*5}"
        final_label = 0
        tp, fp, tn, fn = 0, 0, 0, 0
        num_have_correct_answer = len(df)
        for idx, row in df.iterrows():
            if "I have no comment." in str(row["Answer"]).strip(): # special case of TruthfulQA'
                num_have_correct_answer -= 1
            if "I have no comment" in str(row[column_name]).strip():
                final_label += 1
                tn += 1 
            else:
                if row[label_column] == 1:
                    tp += 1
                    final_label += 1
                else:
                    fp += 1
        p,r,f1 = f1_calc(tp, fp, fn)
        p,r,f1 = round(p*100, 2), round(r*100, 2), round(f1*100, 2)
        print(f"alpha {i*5}: Truth: {round(final_label/len(df)*100, 2)}, Acc: {round(100*tp/num_have_correct_answer, 2)}")


def majority_vote():
    dataset = pd.read_csv(data_file)
    truth_label = 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0
    num_have_correct_answer = len(dataset)
    for i, row in dataset.iterrows():
        if "I have no comment." in str(row["Answer"]).strip(): # special case of TruthfulQA'
            num_have_correct_answer -= 1
        # convert to np arrays
        alpha_values = [i*5 for i in range(1, 6)]
        alpha_values = np.array(alpha_values)
        confidence_values = [round(row[f'conf_{i*5}'], 4) for i in range(1, 6)]
        confidence_values = np.array(confidence_values)
        tokens = [str(row[f'alpha_{i*5}']) for i in range(1, 6)]
        tokens = np.array(tokens)
        
        majority_vote = {}
        for token in tokens:
            if token in majority_vote:
                majority_vote[token] += 1
            else:
                majority_vote[token] = 1
        
        majority_vote = {k: v for k, v in sorted(majority_vote.items(), key=lambda item: item[1], reverse=True)}
        selected_token = list(majority_vote.keys())[0]
        

        max_token_index = np.where(tokens==selected_token)[0]
        if len(max_token_index) > 1:
            max_confidence_index = np.argmax(confidence_values[max_token_index])
            selected_alpha = alpha_values[max_token_index[max_confidence_index]]
        else:
            selected_alpha = alpha_values[max_token_index[0]]


        if "I have no comment" in str(tokens[int(selected_alpha/5)-1]).strip():
                truth_label += 1
        elif row[f'label_{selected_alpha}'] == 1:
                tp += 1
                truth_label += 1

    print(f"majority vote: Truth: {round(truth_label/len(dataset)*100, 2)}")
    print(f"majority vote: Acc: {round(100*tp/num_have_correct_answer, 2)}")



def max_confidence():
    dataset = pd.read_csv(data_file)
    num_have_correct_answer = len(dataset)
    truth_label = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, row in dataset.iterrows():
        if "I have no comment." in str(row["Answer"]).strip(): # special case of TruthfulQA'
            num_have_correct_answer -= 1
        alpha_values = [i*5 for i in range(1, 6)]
        alpha_values = np.array(alpha_values)
        confidence_values = [round(row[f'conf_{i*5}'], 4) for i in range(1, 6)]
        confidence_values = np.array(confidence_values)
        tokens = [str(row[f'alpha_{i*5}']) for i in range(1, 6)]
        tokens = np.array(tokens)
        
        max_confidence_index = np.argmax(confidence_values)
        selected_alpha = alpha_values[max_confidence_index]
        
        has_correct = True if any([row[f'label_{i*5}'] for i in range(1, 6)]) == 1 else False
        
        if "I have no comment" in str(tokens[int(selected_alpha/5)-1]).strip():
                truth_label += 1
        elif row[f'label_{selected_alpha}'] == 1:
                tp += 1
                truth_label += 1
    print(f"maximum confidence: Truth: {round(truth_label/len(dataset)*100, 2)}")
    print(f"maximum confidence: Acc: {round(100*tp/num_have_correct_answer, 2)}")


def max_confidence_greater_than_t(threshold=0.6):
    dataset = pd.read_csv(data_file)
    num_have_correct_answer = len(dataset)
    final_label = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, row in dataset.iterrows():
        if "I have no comment." in str(row["Answer"]).strip(): # special case of TruthfulQA'
            num_have_correct_answer -= 1
        alpha_values = [i*5 for i in range(1, 6)]
        alpha_values = np.array(alpha_values)
        confidence_values = [round(row[f'conf_{i*5}'], 4) for i in range(1, 6)]
        confidence_values = np.array(confidence_values)
        tokens = [str(row[f'alpha_{i*5}']) for i in range(1, 6)]
        tokens = np.array(tokens)

        max_confidence_index = np.argmax(confidence_values)
        has_correct = True if any([row[f'label_{i*5}'] for i in range(1, 6)]) == 1 else False
        if confidence_values[max_confidence_index] > threshold:
            selected_alpha = alpha_values[max_confidence_index]
            # final_label += row[f'label_{selected_alpha}']
            if "I have no comment" in str(tokens[int(selected_alpha/5)-1]).strip():
                final_label += 1
            elif row[f'label_{selected_alpha}'] == 1:
                tp += 1
                final_label += 1
            else:
                fp += 1
        else:
            final_label += 1
            if not has_correct:
                tn += 1
            else:
                fn += 1
    p,r,f1 = f1_calc(tp, fp, fn)
    p,r,f1 = round(p*100, 2), round(r*100, 2), round(f1*100, 2)
    print(f"maximum confidence > {threshold}: Truth: {round(final_label/len(dataset)*100, 2)}")
    print(f"maximum confidence > {threshold}: Acc: {round(100*tp/num_have_correct_answer, 2)}")
    

if __name__ == "__main__":
    get_base_results()
    print('\n')
    majority_vote()
    print('\n')
    max_confidence()
    print('\n')
    max_confidence_greater_than_t(threshold=0.6)
