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

def configure_and_train(
    chronos_model='small',
    context_hours=48,
    prediction_hours=6,
    train_ratio=0.9,
    val_ratio=0.05,
    test_ratio=0.05,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=0.0001,
    lr_scheduler='linear',
    log_steps=200,
    new_model_name=None,
    push_to_hub=False,
    early_stopping_patience=10,
    num_epochs=5
):
    model_dict = {
        'tiny': ('amazon/chronos-t5-tiny', 'chronos_config/chronos-t5-tiny.yaml'),
        'mini': ('amazon/chronos-t5-mini', 'chronos_config/chronos-t5-mini.yaml'),
        'small': ('amazon/chronos-t5-small', 'chronos_config/chronos-t5-small.yaml'),
        'base': ('amazon/chronos-t5-base', 'chronos_config/chronos-t5-base.yaml'),
        'large': ('amazon/chronos-t5-large', 'chronos_config/chronos-t5-large.yaml')
    }
    
    model_name, file_path = model_dict[chronos_model]
    
    if push_to_hub:
        if new_model_name is None:
            new_model_name = f'moschouChry/chronos-t5-{chronos_model}-fine-tuned_{timestamp}'
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

    train_dataset,val_dataset,test_dataset = create_dataloaders(train_ratio,val_ratio,test_ratio,context_window,prediction_window, chronos_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=0.1,
        optim='adamw_torch_fused',
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=log_steps,
        report_to=["tensorboard"],
        tf32=config['tf32'],
        torch_compile=config['torch_compile'],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        eval_strategy="steps",
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True
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
        callbacks=[early_stopping_callback]
    )

    print("Starting training...")
    trainer.train()

    plot_loss(trainer.state.log_history, args)
    plot_learning_rate(trainer.state.log_history, args)
    save_log_history(trainer.state.log_history)
    print('Training completed')
    print(f'Training and evaluation loss plot saved')

    if push_to_hub:
        print("Pushing model to hub...")
        model.push_to_hub(new_model_name)
        print(f"Model pushed to hub. New model name: {new_model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Chronos model with user-defined parameters.")
    parser.add_argument('--chronos_model', type=str, default='small', choices=['tiny', 'mini', 'small', 'base', 'large'])
    parser.add_argument('--context_hours', type=int, default=48)
    parser.add_argument('--prediction_hours', type=int, default=6)
    parser.add_argument('--train_ratio', type=float, default=0.98)
    parser.add_argument('--val_ratio', type=float, default=0.01)
    parser.add_argument('--test_ratio', type=float, default=0.01)
    parser.add_argument('--per_device_train_batch_size', type=int, default=128)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--new_model_name', type=str, default=None)
    parser.add_argument('--push_to_hub', type=bool, default=False)
    parser.add_argument('--early_stopping_patience', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=1000)
    
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
        num_epochs=args.num_epochs
    )