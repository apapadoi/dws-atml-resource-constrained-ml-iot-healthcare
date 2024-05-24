import pandas as pd
import torch

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import root_mean_squared_error

from shanghai_t_dm_utils import read_dataset
from gptq_chronos import GPTQChronosPipeline


RESULT_OUTPUT_DIR = "./results"
MODEL_OUTPUT_FOLDER = "../models/shanghai_t_dm/8bits"
Path(RESULT_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


pipeline = GPTQChronosPipeline.from_pretrained(
    "distilbert/distilgpt2",
    MODEL_OUTPUT_FOLDER,
    device_map="cuda:1",
    load_in_4bit=True,
    chronos_config={
        "context_length": 512,
        "eos_token_id": 1,
        "model_type": "causal",
        "n_special_tokens": 2,
        "n_tokens": 4096,
        "num_samples": 20,
        "pad_token_id": 0,
        "prediction_length": 64,
        "temperature": 1.0,
        "tokenizer_class": "MeanScaleUniformBins",
        "tokenizer_kwargs": {
        "high_limit": 15.0,
        "low_limit": -15.0
        },
        "top_k": 50,
        "top_p": 1.0,
        "use_eos_token": True
    }
    # torch_dtype=torch.bfloat16,
)

# see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5478034/ for acceptable deviation (below 15%, in case of CGM values less than 100 then errors below 15 units)
# TODO causalLM model fine-tuning on validation set (create pairs of num_samples + prediction length for fine-tuning) and then gptq quantization?

t1_dataset_data, t1_dataset_file_names = read_dataset()
t2_dataset_data, t2_dataset_file_names = read_dataset("T2")
data = t1_dataset_data + t2_dataset_data
file_names = t1_dataset_file_names + t2_dataset_file_names
rmses = []

for df, file_name in zip(
   data,
   file_names
):
    forecast = pipeline.predict(
        context=torch.tensor(df["CGM"].iloc[:-24]),
        prediction_length=24, # 6 hours ahead because sample rate is 15 minutes
        num_samples=96, # last 24 hours
    )
    print('df shape: ', df.shape)
    # print(ChronosPipeline.predict.__doc__)

    forecast_index = range(len(df) - 24, len(df))
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    print('median shape: ', median.shape)

    actual = np.array(df[-24:]['CGM'].tolist())

    print('actual shape: ', actual.shape)

    rmse = root_mean_squared_error(actual, median)

    rmses.append(rmse)
    
    plt.figure(figsize=(8, 4))

    plt.plot([], [], ' ', label=f'RMSE = {rmse}')
    plt.plot(df["CGM"], color="royalblue", label="historical data")
    plt.plot(forecast_index, median, color="tomato", label="median forecast")
    plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
    plt.legend()
    plt.grid()
    plt.title(file_name)

    plt.savefig(f'{RESULT_OUTPUT_DIR}/{file_name}_distilgpt2.png')
    break

print(f'Average RMSE: {sum(rmses) / len(rmses)}')
