#!/bin/bash

extract_numbers() {
    local output="$1"

    local model_size=$(echo "$output" | grep -oP 'The input model file size \(MB\): \K[0-9.]+')
    local init=$(echo "$output" | grep -oP 'Init: \K\d+')
    local first=$(echo "$output" | grep -oP 'First inference: \K\d+')
    local warmup_avg=$(echo "$output" | grep -oP 'Warmup \(avg\): \K[0-9.]+')
    local inference_avg=$(echo "$output" | grep -oP 'Inference \(avg\): \K[0-9.]+')
    local init_memory=$(echo "$output" | grep -oP 'Memory footprint delta from the start of the tool \(MB\): init=\K[0-9.]+')
    local overall_memory=$(echo "$output" | grep -oP 'Memory footprint delta from the start of the tool \(MB\): init=[0-9.]+ overall=\K[0-9.]+')

    local count=$(echo "${init[@]}" | wc -l)
    local median=$(( count / 2 ))

    echo "{\"Init\": ${init[$median]}, \"First Inference\": ${first[$median]}, \"Warmup (avg)\": ${warmup_avg[$median]}, \"Inference (avg)\": ${inference_avg[$median]}, \"Model Size (MB)\": ${model_size[$median]}, \"Init Memory (MB)\": ${init_memory[$median]}, \"Overall Memory (MB)\": ${overall_memory[$median]}}"
}

declare -A results

iterations=10

while IFS= read -r -d '' file; do
        output=$(./linux_x86-64_benchmark_model --graph="$file" --num_threads=1 --allow_fp16=true 2>&1)
        if [ $? -eq 0 ]; then
            numbers=$(extract_numbers "$output")
            results["$file"]=$numbers
        else
            echo "Error executing benchmark for file: $file"
            echo "Error output: $output"
        fi
done < <(find . -type f -name '*quant*' -print0)

# calculate results for the original model
file="post_training_float16_quantization/initial_model.tflite"
output=$(./linux_x86-64_benchmark_model --graph="$file" --num_threads=1 --allow_fp16=true 2>&1)
if [ $? -eq 0 ]; then
    numbers=$(extract_numbers "$output")
    results["$file"]=$numbers
else
    echo "Error executing benchmark for file: $file"
    echo "Error output: $output"
fi

for file in "${!results[@]}"; do
    echo "Results for $file:"
    echo "${results[$file]}"
done