import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.k = len(labels)
        self.data = data
        self.labels = [torch.tensor(labels[i], dtype=torch.float32).view(len(labels[i]), 1) for i in range(len(labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        labels = {f'label{i+1}': self.labels[i][index] for i in range(self.k)}

        # Create the dictionary to return, combining data and labels
        result = {'data': data}
        result.update(labels)

        return result


def build_dataloader(answers, labels, batch_size=32, shuffle=True):
    data = []
    for idx, group in enumerate(zip(*answers)):
        answers = [a.to(torch.float32) for a in group]
        answers = torch.stack(answers)
        data.append(answers)
    dataset = CustomDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader   

def load_input_data(data_files):
    answers = []
    for data_file in data_files:
        answers.append(torch.load(data_file))
    return answers


def load_labels(labels_path, k):
    dataset = pd.read_csv(labels_path)
    label_columns = [f'label_{i*5}' for i in range(1,k+1)]
    labels = dataset[label_columns].values.T.tolist()
    return labels


def data_balancing(answers, labels, k=5):
    minority_threshold = 300
    majority_threshold = 30

    min_occurance = 1 # do not include very rare combs, can omit in future
    combo_labels = []
    for i in range(len(labels[0])):
        combo = tuple([labels[j][i] for j in range(7)])
        combo_labels.append(combo) 
    
    combo_counts = Counter(combo_labels)
    minority_combos = [combo for combo, count in combo_counts.items() if min_occurance <= count < minority_threshold]
    print("combo count: ", combo_counts)
    for combo in minority_combos:
    
        # Get combo indexes
        combo_idxs = [i for i, label in enumerate(combo_labels) if label == combo]

        # Calculate oversampling needed
        n_to_duplicate = minority_threshold - combo_counts[combo]

        # Oversample
        oversample_idxs = np.random.choice(combo_idxs, size=n_to_duplicate)

        # Concatenate data
        answers = [a+[a[i] for i in oversample_idxs] for a in answers]
        
        # Concatenate labels
        for i in range(k):
            labels[i] = labels[i] + [labels[i][j] for j in oversample_idxs]
    
    majority_combos = [c for c, cnt in combo_counts.items() if cnt > majority_threshold] 
    remove_idxs = []
    for combo in majority_combos:
        combo_idxs = [i for i, label in enumerate(combo_labels) if label==combo]
        n_remove = combo_counts[combo] -  majority_threshold
        remove_idxs += np.random.choice(combo_idxs, size=n_remove, replace=False).tolist()
    
    remove_idxs = list(set(remove_idxs))
    for i in range(k): 
        answers[i] = [answers[i][j] for j in range(len(answers[i])) if j not in remove_idxs]
        labels[i] = [labels[i][j] for j in range(len(labels[i])) if j not in remove_idxs]
    
    return answers, labels