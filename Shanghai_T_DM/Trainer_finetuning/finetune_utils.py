from datetime import datetime
import json

import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from shanghai_t_dm_utils import read_dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length, prediction_length, tokenizer):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.tokenizer = tokenizer
        
    def __len__(self):
        # Each dataframe contributes (len(df) - context_length - prediction_length + 1) samples
        return sum(len(df) - self.context_length - self.prediction_length + 1 for df in self.data)

    def __getitem__(self, idx):
        # Find the appropriate dataframe and index within that dataframe
        cumulative_length = 0
        for df in self.data:
            current_length = len(df) - self.context_length - self.prediction_length + 1
            if idx < cumulative_length + current_length:
                local_idx = idx - cumulative_length
                context = df['CGM'].values[local_idx:local_idx + self.context_length]

                past_target = torch.tensor(context).unsqueeze(0)
                input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
                    past_target
                )
                
                target = df['CGM'].values[local_idx + self.context_length:local_idx + self.context_length + self.prediction_length]
                future_target = torch.tensor(target).unsqueeze(0)
                labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
                labels[labels_mask == 0] = -100                
                return {
                    "input_ids": input_ids.squeeze(0),
                    "attention_mask": attention_mask.squeeze(0),
                    "labels": labels.squeeze(0),
                }
            cumulative_length += current_length
        raise IndexError("Index out of range in dataset")

def save_log_history(log_history,timestamp):
    output_path = f"log_history_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(log_history, f, indent=4)

def plot_learning_rate(log_history, args,timestamp):
    lr = []
    steps = []

    for entry in log_history:
        if 'learning_rate' in entry.keys():
            lr.append(entry['learning_rate'])
            if 'epoch' in entry.keys():
                steps.append(f"{entry['epoch']:.2f}")
            else:
                steps.append(entry['step'])

    plt.figure(figsize=(10, 5))
    plt.plot(steps, lr, label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Over Time - Chronos Model: {args.chronos_model}, {args.dataset} Dataset\n'
              f'Train Batch Size: {args.per_device_train_batch_size}, Initial LR: {args.learning_rate}, '
              f'Context Hours: {args.context_hours}, Prediction Hours: {args.prediction_hours}')

    plt.legend()
    plt.grid(True)

    # Set xticks based on the number of steps
    if len(steps) > 5:
        xticks = steps[::len(steps)//5]
    else:
        xticks = steps

    plt.xticks(ticks=xticks, labels=xticks)

    plot_filename = f"lr_{timestamp}.png"
    plt.savefig(plot_filename)

def plot_loss(log_history, args, timestamp):
    train_loss = []
    eval_loss = []
    steps = []

    for entry in log_history:
        if 'loss' in entry.keys():
            train_loss.append(entry['loss'])
            if 'epoch' in entry.keys():
                steps.append(f"{entry['epoch']:.2f}")
            else:
                steps.append(entry['step'])
        if 'eval_loss' in entry.keys():
            eval_loss.append(entry['eval_loss'])

    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_loss, label='Training Loss')
    plt.plot(steps, eval_loss, label='Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Evaluation Loss Over Time - Chronos Model: {args.chronos_model}, {args.dataset} Dataset\n'
              f'Train Batch Size: {args.per_device_train_batch_size}, Initial LR: {args.learning_rate}, '
              f'Context Hours: {args.context_hours}, Prediction Hours: {args.prediction_hours}')

    plt.legend()
    plt.grid(True)

    # Set xticks based on the number of steps
    if len(steps) > 5:
        xticks = steps[::len(steps)//5]
    else:
        xticks = steps

    plt.xticks(ticks=xticks, labels=xticks)

    plot_filename = f"loss_{timestamp}.png"
    plt.savefig(plot_filename)


def create_dataloaders(dataset_name, data, split_ratios,context_window,prediction_window, chronos_config):

    # Load the datasets
    print(f"Loading {dataset_name} datasets...")

    train_data, val_data, test_data = [], [], []
    train_ratio,val_ratio,test_ratio = split_ratios

    assert(train_ratio+val_ratio+test_ratio==1)

    for i in range(len(data)):
        num_rows = len(data[i])
        sample_size = context_window + prediction_window
        if num_rows>sample_size:
            if num_rows < 3*sample_size + 1:
                train_data.append(data[i])
            else:
                available_samples = (num_rows - 3*sample_size - 1)
                training_samples = int(available_samples * train_ratio)
                validation_samples = int(available_samples * val_ratio)
                testing_samples = int(available_samples * test_ratio)
            
                if validation_samples == 0:
                    training_samples -= 1
                    validation_samples += 1
                if testing_samples == 0:
                    training_samples -= 1
                    testing_samples += 1
                
                train_data.append(data[i].iloc[:training_samples + sample_size + 1])
                val_data.append(data[i].iloc[training_samples + sample_size + 1:training_samples + validation_samples + 2*sample_size + 2])
                test_data.append(data[i].iloc[training_samples + validation_samples + 2*sample_size + 2:])

    # TimeSeries Dataloaders
    train_dataset = TimeSeriesDataset(train_data, context_window, prediction_window, tokenizer=chronos_config.create_tokenizer())
    val_dataset = TimeSeriesDataset(val_data, context_window, prediction_window, tokenizer=chronos_config.create_tokenizer())
    test_dataset = TimeSeriesDataset(test_data, context_window, prediction_window, tokenizer=chronos_config.create_tokenizer())

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")
    print(f"Number of test examples: {len(test_dataset)}")

    return train_dataset,val_dataset,test_dataset