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
from dataset_k import build_dataloader, load_input_data, load_labels
from classification.answer_selection_models import AnswerSelectionUni
device = torch.device('cpu')
import logging




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

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


parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, default='llama2_chat_7B', choices=HF_NAMES.keys(), help='model name')
parser.add_argument('--dataset_name', type=str, default='sciq', help='name of the dataset')
parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
parser.add_argument('--pre_train_epochs', type=int, default=20)
parser.add_argument('--patiance', type=int, default=10)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--num_folds', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=1e-3)
args = parser.parse_args()

directory_path = {
    'nq': 'NQ',
    'trivia_qa': 'TriviaQA',
    'sciq': 'SciQ',
    'tqa_mc2' : 'TQA'
}

DATA_DIR = directory_path[args.dataset_name]

logging.basicConfig(
    filename=f'out_k_{args.k}.log',  # File where logs will be written
    filemode='w',            # Append mode, use 'w' for overwriting
    level=logging.INFO,      # Minimum level of logs to save
    format='%(asctime)s - %(message)s'  # Log message format
)

def calculate_accuracy_tqa(fold):
    data_dir = f'../{DATA_DIR}/results/{args.model_name}'
    # test_file_path = f'{data_dir}/test_outputs_{direction}_min.csv'
    test_file_path = f'{data_dir}/val/val_outputs_k_{args.k}_fold_{fold}.csv'
    dataset = pd.read_csv(test_file_path)
    
    final_label = 0
    truth_label = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    num_have_correct_answer = len(dataset)
    # print("num_have_correct_answer: ", num_have_correct_answer, len(dataset))
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
    logging.info(f"Accuracy: {round(100*tp/num_have_correct_answer,2)}, Truthfulness: {round(100*truth_label/len(dataset), 2)}, TA: {ta_score}")
    return ta_score


def calculate_accuracy(fold):
    data_dir = f'../{DATA_DIR}/results/{args.model_name}'
    test_file_path = f'{data_dir}/val/val_outputs_k_{args.k}_fold_{fold}.csv'
    dataset = pd.read_csv(test_file_path)
    
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
                # print("I come here")
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
    logging.info("Accuracy: ", round(100*tp/len(dataset),2))
    logging.info("Truthfulness: ", round(100*truth_label/len(dataset), 2))  
    logging.info("FINAL TA: ", ta_score)
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
                model_path = f"../{DATA_DIR}/results/{args.model_name}/models/best_model_{args.k}"
                torch.save(model.state_dict(), open(model_path, "wb"))
                early_stop_cnt = 0
            
            early_stop_cnt += 1
            if early_stop_cnt > args.patiance:
                logging.info("Early Stop: best accuracy: {:6.2f}%".format(100 * best_acc_dev))
                break


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


def val(val_dataset, val_dataloader, model, fold=0):
    val_info = defaultdict(list)
    output_path = f"../{DATA_DIR}/results/{args.model_name}"
    model_path = f"{output_path}/models/best_model_{args.k}"
    state_dict = torch.load(open(model_path, 'rb'), map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    _, _, _, all_outputs  = eval(val_dataloader, model, val_info)
    
    for i in range(0,args.k):
        val_dataset[f"pred_{(i+1)*5}"] = all_outputs[i]
    val_dataset.to_csv(f"{output_path}/val/val_outputs_k_{args.k}_fold_{fold}.csv", index=False)


def cross_validate(data_inputs, labels, n_folds=5, batch_size=32):
    kf = KFold(n_splits=n_folds)
    fold = 0
    model_performances = []

    for train_index, val_index in kf.split(data_inputs[0]):  # Assuming all inputs have the same length
        train_index = train_index.tolist()
        val_index = val_index.tolist()
      
        logging.info(f"Training fold {fold + 1}/{n_folds}")
        train_data = [[input_data[i] for i in train_index] for input_data in data_inputs]
        train_labels = [[label[i] for i in train_index] for label in labels]
        # train_data, train_labels = data_balancing(train_data, train_labels, args.k)

        val_data = [[input_data[i] for i in val_index] for input_data in data_inputs]
        val_dataset = pd.read_csv(labels_path).iloc[val_index]
        val_labels = [[label[i] for i in val_index] for label in labels]

        # Prepare dataloaders
        train_dataloader = build_dataloader(train_data, train_labels, batch_size)
        val_dataloader = build_dataloader(val_data, val_labels, batch_size)

        # Train model
        input_size, hidden_size = MODEL_DIMS[args.model_name]
        model = AnswerSelectionUni(input_size, hidden_size)  # Initialize model for each fold
        train(train_dataloader, val_dataloader, model)
        
        # Evaluate model
        val(val_dataset, val_dataloader, model, fold)
        model_performances.append(calculate_accuracy_tqa(fold))
        # else: 
        #     model_performances.append(calculate_accuracy(fold))
        fold += 1
        
        

    return model_performances


if __name__ == "__main__":
    # get base results:
    data_dir = f'../{DATA_DIR}/results/{args.model_name}/'
    
    # train
    train_files = [f'{data_dir}/train/hidden_states/alpha_{i*5}_mean_hidden_states.pt' for i in range(1,args.k+1)]
    data_inputs = load_input_data(train_files)


    # load labels:
    batch_size = 32
    labels_path = f'{data_dir}/train/files/classification_train_data_all.csv'
    labels = load_labels(labels_path, args.k)
    
    assert len(data_inputs[0]) == len(labels[0])

    results = cross_validate(data_inputs, labels, args.num_folds, batch_size=batch_size)
    logging.info('*'*80)
    logging.info(f'k: {args.k} FINAL RESULT: {np.mean(results)}')
    logging.info('*'*80)
