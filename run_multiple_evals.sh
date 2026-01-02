#!/bin/bash

# OpenRouter APIキーの設定（環境変数から読み込むか、ここで直接設定してください）
# export OPENROUTER_API_KEY="sk-or-..." 

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable is not set."
    echo "Please export it: export OPENROUTER_API_KEY=your_key_here"
    exit 1
fi

# 評価対象のモデルリスト
# OpenRouterのモデル名を指定してください
MODELS=(
    "openai/gpt-4o"
    "anthropic/claude-3.5-sonnet"
    "google/gemini-2.5-flash"
    "meta-llama/llama-3-70b-instruct"
)

# OpenRouterのBase URL
BASE_URL="https://openrouter.ai/api/v1"

echo "=== Starting Batch Evaluation ==="
echo "Models to evaluate: ${MODELS[*]}"

for model in "${MODELS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Evaluating model: $model"
    echo "----------------------------------------------------------------"
    
    python3 jbc_eval.py \
        --model "$model" \
        --base-url "$BASE_URL" \
        --api-key "$OPENROUTER_API_KEY" \
        --n-threads 5
        
    if [ $? -eq 0 ]; then
        echo "Successfully finished evaluation for $model"
    else
        echo "Error occurred during evaluation for $model"
    fi
    
    echo ""
done

echo "=== All evaluations completed ==="
