import argparse
from math import sqrt
from collections import defaultdict
import random
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from classification_utils import f1_calc, conf_matrix_calc
from classification.dataset import build_dataloader, load_input_data, load_labels, data_balancing
from classification.answer_selection_models import AnswerSelectionUni
device = torch.device('cpu')
import logging
logging.basicConfig(
    filename=f'out.log', 
    filemode='w',            
    level=logging.INFO,   
    format='%(asctime)s - %(message)s'
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, default='llama2_chat_7B', help='model name')
parser.add_argument('--dataset_name', type=str, default='nq', help='dataset name')
parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
parser.add_argument('--pre_train_epochs', type=int, default=20)
parser.add_argument('--patiance', type=int, default=10)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--k', type=int, default=5, help='number of interventions and responses')
parser.add_argument('--weight_decay', type=float, default=1e-3)
args = parser.parse_args()

HF_NAMES = {
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'gpt2_xl': 'gpt2-xl',
    'gpt2_large': 'gpt2-large'
}

MODEL_DIMS = {
    'vicuna_7B': [4096, 512], 
    'llama2_chat_7B': [4096, 512], 
    'llama2_chat_13B': [5120, 640], 
    'gpt2_xl': [1600, 200],
    'gpt2_large': [1280, 160] 
}

directory_path = {
    'nq': 'NQ',
    'trivia_qa': 'TriviaQA',
    'sciq': 'SciQ',
    'tqa_mc2' : 'TQA'
}

DATA_DIR = f"results/{directory_path[args.dataset_name]}"


def calculate_accuracy_tqa(dataset):
    final_label = 0
    truth_label = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    num_have_correct_answer = len(dataset)
    for i, row in dataset.iterrows():
        if "I have no comment." in str(row["Answer"]).strip():
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
    ta_score = round(sqrt((100*tp/num_have_correct_answer) * (100*truth_label/len(dataset))), 2)
    # logging.info(f"Accuracy: {round(100*tp/num_have_correct_answer,2)}, Truthfulness: {round(100*truth_label/len(dataset), 2)}, TA: {ta_score}")
    return ta_score


def calculate_accuracy(dataset):
    final_label = 0
    truth_label = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, row in dataset.iterrows():
        # convert to np arrays
        alpha_values = [i*5 for i in range(1, args.k+1)]
        alpha_values = np.array(alpha_values)
        confidence_values = [round(row[f'conf_{i*5}'], 4) for i in range(1, args.k+1)]
        confidence_values = np.array(confidence_values)
        tokens = [str(row[f'alpha_{i*5}']) for i in range(1, args.k+1)]
        tokens = np.array(tokens)
        
        preds = [row[f'pred_{(i*5)}'] for i in range(1, args.k+1)]
        preds_array = np.array(preds)

        has_correct = True if any([row[f'label_{i*5}'] for i in range(1, args.k+1)]) == 1 else False

        if np.all(np.equal(preds_array, 0)):
            truth_label += 1
            if "I have no comment." in str(row["Answer"]).strip(): # special case of TruthfulQA'
                tp += 1 
            if not has_correct:
                final_label += 1
                tn += 1
            else:
                fn += 1
        else: 
            correct_preds_indices = np.where(preds_array==1)[0]
            
            max_confidence_index = np.argmax(confidence_values[correct_preds_indices])
            selected_alpha = alpha_values[correct_preds_indices[max_confidence_index]]
            if row[f'label_{selected_alpha}'] == 1:
                tp += 1
                final_label += 1
                truth_label += 1
            else: 
                fp += 1
    ta_score = round(sqrt((100*tp/len(dataset)) * (100*truth_label/len(dataset))), 2)
    # logging.info("Accuracy: ", round(100*tp/len(dataset),2))
    # logging.info("Truthfulness: ", round(100*truth_label/len(dataset), 2))  
    # logging.info("FINAL TA: ", ta_score)
    return ta_score


def train(dataloader, val_dataloader, model):
    # retain the information regarding acc and loss:
    val_info = defaultdict(list)
    train_info = defaultdict(list)

    # loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=args.weight_decay) 
    model.train()
    best_acc_dev, best_f1_dev = 0, 0
    early_stop_cnt = 0
    for epoch in tqdm(range(args.epochs)):
        total_loss = 0
        for batch in dataloader:
            # access data for the current batch
            input_data = batch['data'] # to_device
            optimizer.zero_grad()
            output = model(input_data) 
            loss = 0
            for i in range(0, args.k):
                label = batch[f'label{i+1}']
                logits = output[:,i,:]
                loss += criterion(logits, label)
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        average_loss = total_loss/len(dataloader)
        train_info['loss'].append(average_loss)
        
        if epoch >= args.pre_train_epochs:
            dev_accuracy, dev_f1, val_info, _ = eval(val_dataloader, model, val_info)
            if dev_accuracy + dev_f1 > best_acc_dev + best_f1_dev:
                best_acc_dev = dev_accuracy
                best_f1_dev = dev_f1
                model_path = f"../{DATA_DIR}/results/{args.model_name}/models/best_model"
                torch.save(model.state_dict(), open(model_path, "wb"))
                early_stop_cnt = 0
            
            early_stop_cnt += 1
            if early_stop_cnt > args.patiance:
                logging.info("Early Stop: best accuracy: {:6.2f}%".format(100 * best_acc_dev))
                break
    model_path = f"../{DATA_DIR}/results/{args.model_name}/last_model"
    torch.save(model.state_dict(), open(model_path, "wb")) 


def eval(dataloader, model, info, mode="val"):
    model.eval()

    all_outputs = [[] for i in range(args.k)]
    corrects = [0]*args.k
    totals = [0]*args.k
    total_tp, total_fp, total_fn, total_tn = [0]*args.k, [0]*args.k, [0]*args.k, [0]*args.k
    with torch.no_grad(): 
        for batch in dataloader:
            input_data = batch['data']

            output = model(input_data) 
            for i in range(0, args.k):
                label = batch[f'label{i+1}']
                logits = output[:,i,:]
                preds = (logits > 0.5).float()
                all_outputs[i].extend(preds.squeeze().tolist())
                corrects[i] += (label == preds).sum().item()
                totals[i] += label.size(0)
                tp, fp, tn, fn = conf_matrix_calc(label, preds)
            
                total_tp[i] += tp
                total_fp[i] += fp
                total_fn[i] += fn
                total_tn[i] += tn
    accuracy = sum([c/t for (c,t) in zip(corrects, totals)]) / args.k
    total_f1 = sum([f1_calc(tp, fp, fn) for (tp, fp, fn) in zip(total_tp, total_fp, total_fn)])/args.k
    info['total_acc'].append(accuracy)
    info['total_f1'].append(total_f1)
    return accuracy, total_f1, info, all_outputs

def test(model, last=False):
    batch_size = 32
    test_files = [f'{data_dir}/test/hidden_states/alpha_{i*5}_mean_hidden_states.pt' for i in range(1,args.k+1)]
    test_data= load_input_data(test_files)
    test_labels_path = f'{data_dir}/test/files/classification_test_data.csv'
    test_labels = load_labels(test_labels_path)
    
    test_dataloader = build_dataloader(test_data, test_labels, batch_size, shuffle=False)
    test_info = defaultdict(list)
    output_path = f"../{DATA_DIR}/results/{args.model_name}"
    model_path = f"{output_path}/last_model_{args.lstm_direction}" if last else f"{output_path}/best_model_{args.lstm_direction}"
    state_dict = torch.load(open(model_path, 'rb'), map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    _, _, _, all_outputs = eval(test_dataloader, model, test_info, mode="test")
    
    # write output to file:
    df = pd.read_csv(f'{output_path}/test/files/classification_test_data.csv')
    for i in range(0,args.k):
        df[f"pred_{(i+1)*5}"] = all_outputs[i]
    df.to_csv(f"{output_path}/test_outputs_{args.lstm_direction}.csv", index=False)
    
    if args.dataset_name == "tqa_mc2":
        return calculate_accuracy_tqa(df)
    else: 
        return calculate_accuracy(df)


def main(data_inputs, labels, batch_size=32):
    ntrain = int(len(data_inputs[0]) * (1-args.val_ratio))
    
    train_data = [input_data[:ntrain] for input_data in data_inputs]
    train_labels = [label[:ntrain] for label in labels]
    
    # balance the combinations of correct and incorrect responses
    train_data, train_labels = data_balancing(train_data, train_labels, args.k)

    val_data = [input_data[ntrain:] for input_data in data_inputs]
    val_labels = [label[ntrain:] for label in labels]

    # Prepare dataloaders
    train_dataloader = build_dataloader(train_data, train_labels, batch_size)
    val_dataloader = build_dataloader(val_data, val_labels, batch_size)

    input_size, hidden_size = MODEL_DIMS[args.model_name]
    model = AnswerSelectionUni(input_size, hidden_size)  # Initialize model for each fold

    # Train model
    if args.mode == "train":
        train(train_dataloader, val_dataloader, model)

    # test
    ta_score = test(model)
    logging.info('*'*80)
    logging.info(f'Final TA score: {ta_score}')
    logging.info('*'*80)



if __name__ == "__main__":
    # get base results:
    data_dir = f'../{DATA_DIR}/results/{args.model_name}/'
    
    # train
    train_files = [f'{data_dir}/train/hidden_states/alpha_{i*5}_mean_hidden_states.pt' for i in range(1,args.k+1)]
    data_inputs = load_input_data(train_files)


    # load labels:
    batch_size = 32
    labels_path = f'{data_dir}/train/files/classification_train_data.csv'
    labels = load_labels(labels_path, args.k)
    
    assert len(data_inputs[0]) == len(labels[0])


    main(data_inputs, labels, batch_size=batch_size)
