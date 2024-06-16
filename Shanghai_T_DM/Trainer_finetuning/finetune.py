import argparse
import warnings
from datetime import datetime
from finetune_utils import create_dataloaders, plot_learning_rate, plot_loss, save_log_history
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import yaml
import torch
import random
from pathlib import Path
import transformers
from train import ChronosConfig, get_next_path, load_model
import ast
import torch._dynamo
from huggingface_hub import login
import sys
import pickle
from peft import LoraConfig, get_peft_model
import evaluate

# Ignore all warnings
warnings.filterwarnings("ignore")

torch._dynamo.config.suppress_errors = True

# Create a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the log file name
log_filename = f"finetuning_{timestamp}.log"

# Open the log file
log_file = open(log_filename, "a")

# Logger class to write to both terminal and log file
class Logger:
    def __init__(self):
        self.terminal = sys.__stdout__  # Keep a reference to the original standard output
        self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()  # Ensure the message is immediately displayed on the terminal
        self.log.flush()  # Ensure the message is immediately written to the log file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout and stderr to Logger
sys.stdout = Logger()
sys.stderr = Logger()

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def configure_and_train(
    chronos_model='small',
    context_hours=48,
    prediction_hours=6,
    train_ratio=0.98,
    val_ratio=0.01,
    test_ratio=0.01,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=32,
    learning_rate=0.0001,
    lr_scheduler='polynomial',
    log_steps=100,
    new_model_name=None,
    push_to_hub=False,
    early_stopping_patience=30,
    num_epochs=100,
    dataset='ShanghaiT1DM',
    train_with_lora=False,
    lora_option=1,
):
    
    print(f'train with lora: {train_with_lora}')
    print(f'dataset: {dataset}')
    assert(dataset in ['ShanghaiT1DM','ShanghaiT2DM','Patient0','Dinamo_Shanghai_T1DM'])

    model_dict = {
        'tiny': ('amazon/chronos-t5-tiny', 'chronos_config/chronos-t5-tiny.yaml'),
        'mini': ('amazon/chronos-t5-mini', 'chronos_config/chronos-t5-mini.yaml'),
        'small': ('amazon/chronos-t5-small', 'chronos_config/chronos-t5-small.yaml'),
        'base': ('amazon/chronos-t5-base', 'chronos_config/chronos-t5-base.yaml'),
        'large': ('amazon/chronos-t5-large', 'chronos_config/chronos-t5-large.yaml'),
        'finetuned_tiny_1': ('moschouChry/finetuned-chronos-tiny-type-1', 'chronos_config/chronos-t5-tiny.yaml'),
        'finetuned_tiny_2': ('moschouChry/finetuned-chronos-tiny-type-2', 'chronos_config/chronos-t5-tiny.yaml'),
        'finetuned_small_1': ('moschouChry/finetuned-chronos-small-type-1', 'chronos_config/chronos-t5-small.yaml'),
        'finetuned_small_2': ('moschouChry/finetuned-chronos-small-type-2', 'chronos_config/chronos-t5-small.yaml'),
    }
    
    model_name, file_path = model_dict[chronos_model]
    
    if push_to_hub:
        if new_model_name is None:
            new_model_name = f'moschouChry/chronos-t5-{chronos_model}-{dataset}-fine-tuned_{timestamp}'
        # Login to Hugging Face Hub
        access_token = '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'
        login(token=access_token)

    context_window = context_hours * 4  # Assuming 4 samples per hour
    prediction_window = prediction_hours * 4

    # Read the YAML configuration file
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    config.update({
        'random_init': False,
        'model_id': model_name,
        'context_length': context_window,
        'min_past': context_window,
        'prediction_length': prediction_window,
        'lr_scheduler_type': lr_scheduler,
        'output_dir': './finetuned_model/',
        'probability': [1.0]
    })

    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    if config['tf32'] and not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8):
        config['tf32'] = False


    seed = random.randint(0, 2**32)
    transformers.set_seed(seed=seed)

    output_dir = Path(config['output_dir'])
    tokenizer_kwargs = config['tokenizer_kwargs']
    
    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)

    output_dir = get_next_path("run", base_dir=output_dir, file_type="")

    model = load_model(
        model_id=config['model_id'],
        model_type=config['model_type'],
        vocab_size=config['n_tokens'],
        random_init=config['random_init'],
        tie_embeddings=config['tie_embeddings'],
        pad_token_id=0,
        eos_token_id=1
    )

    chronos_config = ChronosConfig(
        tokenizer_class=config['tokenizer_class'],
        tokenizer_kwargs=tokenizer_kwargs,
        n_tokens=config['n_tokens'],
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=config['use_eos_token'],
        model_type=config['model_type'],
        context_length=config['context_length'],
        prediction_length=config['prediction_length'],
        num_samples=config['num_samples'],
        temperature=1,
        top_k=50,
        top_p=1
    )

    model.config.chronos_config = chronos_config.__dict__

    # Load the dictionary from the file
    with open('dataset_dictionary.pkl', 'rb') as file:
        loaded_data_dict = pickle.load(file)

    data = loaded_data_dict[dataset]
    dataset_name = dataset

    split_ratios = [train_ratio,val_ratio,test_ratio]
    train_dataset,val_dataset,test_dataset = create_dataloaders(dataset_name,data,split_ratios,context_window,prediction_window, chronos_config)

    print(f"Trainable parameters without lora")
    print_trainable_parameters(model)
    
    if train_with_lora:
        if lora_option == 1:
            print(f"Trainable parameters with lora 1")
            lora_config = LoraConfig(
                task_type='seq2seq',
                r=4,
                lora_alpha=8,
                lora_dropout=0.1,
            )
        elif lora_option == 2:
            print(f"Trainable parameters with lora 2")
            lora_config = LoraConfig(
                task_type='seq2seq',
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
            )
        elif lora_option == 3:
            print(f"Trainable parameters with lora 3")
            lora_config = LoraConfig(
                task_type='seq2seq',
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
            )
        elif lora_option == 4:
            print(f"Trainable parameters with lora 4")
            lora_config = LoraConfig(
                task_type='seq2seq',
                r=128,
                lora_alpha=256,
                lora_dropout=0.1,
            )
        lora_model = get_peft_model(model, lora_config)
        print_trainable_parameters(lora_model)
        model = lora_model


    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,
        optim='adamw_torch_fused',
        logging_dir=str(output_dir / "logs"),
        logging_strategy="epoch",
        logging_steps=log_steps,
        save_strategy="epoch",
        report_to=["tensorboard"],
        tf32=config['tf32'],
        torch_compile=config['torch_compile'],
        evaluation_strategy="epoch",
        metric_for_best_model='loss',
        load_best_model_at_end=True,
        save_total_limit=10,
        label_names=['labels'],
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping_callback],
    )

    print("Starting training...")
    trainer.train()

    plot_loss(trainer.state.log_history, args, timestamp)
    plot_learning_rate(trainer.state.log_history, args, timestamp)
    save_log_history(trainer.state.log_history,timestamp)
    print('Training completed')
    print(f'Training and evaluation loss plot saved')

    if push_to_hub:
        print("Pushing model to hub...")
        if train_with_lora:
            merged_model = model.merge_and_unload()
            merged_model.push_to_hub(new_model_name)
        else:
            model.push_to_hub(new_model_name)
        print(f"Model pushed to hub. New model name: {new_model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Chronos model with user-defined parameters.")
    parser.add_argument('--chronos_model', type=str, default='tiny', choices=['tiny', 'mini', 'small', 'base', 'large', 'finetuned_tiny_1','finetuned_tiny_2','finetuned_small_1','finetuned_small_2'])
    parser.add_argument('--context_hours', type=int, default=24)
    parser.add_argument('--prediction_hours', type=int, default=3)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.05)
    parser.add_argument('--per_device_train_batch_size', type=int, default=128)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=str, default='polynomial')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--new_model_name', type=str, default=None)
    parser.add_argument('--push_to_hub', type=bool, default=False)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='ShanghaiT1DM')
    parser.add_argument('--train_with_lora', type=bool, default=False)
    parser.add_argument('--lora_option', type=int, default=1)
    
    args = parser.parse_args()

    # Log the arguments
    print(f"Arguments: {args}")
    
    configure_and_train(
        chronos_model=args.chronos_model,
        context_hours=args.context_hours,
        prediction_hours=args.prediction_hours,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        log_steps=args.log_steps,
        new_model_name=args.new_model_name,
        push_to_hub=args.push_to_hub,
        early_stopping_patience=args.early_stopping_patience,
        num_epochs=args.num_epochs,
        dataset=args.dataset,
        train_with_lora=args.train_with_lora,
        lora_option=args.lora_option,
    )
