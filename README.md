# Japanese_BreastCancer_Benchmark

日本の乳がん診療ガイドラインに対するLLMの認識を評価するベンチマーク。

## 概要

本ベンチマークは、日本における医薬品の承認状況を踏まえてLLMが日本の乳がん診療をどの程度正確に理解しているかを測定します。**LLM-as-a-judge**（LLMによる評価）と**ルーブリック評価**を採用し、各問題に対してカスタマイズされた評価基準を使用します。

### 主な特徴

- **ルーブリック評価**: 問題ごとにカスタマイズされた評価基準と重み付けスコアリング
- **減点方式**: 日本で未承認の薬剤を推奨した場合に減点
- **日本語評価**: 医療情報を正確に評価するための日本語プロンプト
- **OpenAI互換**: OpenAI APIおよびOpenRouterに対応

## Install

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用方法（OpenAI）

```bash
export OPENAI_API_KEY=your_api_key
python jbc_eval.py --model gpt-4o
```

### OpenRouterの使用

```bash
python jbc_eval.py \
    --model anthropic/claude-3.5-sonnet \
    --base-url https://openrouter.ai/api/v1 \
    --api-key $OPENROUTER_API_KEY
```

### 複数モデルの一括評価

複数のモデルをまとめて評価するための補助スクリプトを用意しています。OpenRouterを使用して、主要なモデル（GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, Llama 3など）を一括で評価できます。

1. `run_multiple_evals.sh`内の`OPENROUTER_API_KEY`を設定するか、環境変数としてエクスポートします。
2. 必要に応じて、スクリプト内の`MODELS`配列を編集して評価対象モデルを変更します。
3. スクリプトを実行します：

```bash
chmod +x run_multiple_evals.sh
./run_multiple_evals.sh
```

### CLIオプション

| オプション | 説明 | デフォルト |
|--------|-------------|---------|
| `--model` | 評価対象モデル（必須） | - |
| `--data` | JSONLデータファイルのパス | `jbc_data.jsonl` |
| `--grader` | 評価に使用するモデル | `gpt-4.1` |
| `--base-url` | カスタムAPIベースURL | OpenAI |
| `--api-key` | APIキー | `OPENAI_API_KEY`環境変数 |
| `--num-examples` | 評価する問題数 | 全件 |
| `--n-threads` | 並列評価スレッド数 | 10 |

## データ形式

問題は`jbc_data.jsonl`に保存されます：

```json
{
  "prompt_id": "jbc_001_cdk46",
  "prompt": [
    {"role": "user", "content": "閉経後ホルモン受容体陽性HER2陰性進行転移再発乳がんの一次治療として..."}
  ],
  "rubrics": [
    {"criterion": "アベマシクリブまたはパルボシクリブを含めている", "points": 2.0, "tags": ["approved_drug"]},
    {"criterion": "日本未承認のリボシクリブを推奨している", "points": -1.0, "tags": ["unapproved_trap"]}
  ],
  "example_tags": ["breast_cancer", "1st_line", "cdk4/6"]
}
```

### スコアリング

- **加点**: 望ましい回答に対して付与（例：承認薬の言及）
- **減点**: 望ましくない回答に対して適用（例：未承認薬の推奨）
- **最終スコア**: `獲得点数 / 総加点可能点数`

## 結果

結果は`results/{model_name}/`に保存されます：

```
results/
├── gpt-4o/
│   ├── jbc_results_gpt-4o_20260102_220233.json
│   └── jbc_report_gpt-4o_20260102_220233.html
└── claude-3.5-sonnet/
    ├── jbc_results_claude-3.5-sonnet_20260102_220233.json
    └── jbc_report_claude-3.5-sonnet_20260102_220233.html
```

## ライセンス

MIT License
