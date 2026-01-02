#!/usr/bin/env python3
"""
Japanese Breast Cancer Benchmark Evaluation Script

LLM-as-a-judgeとルーブリック評価を使用して、
日本の乳がん診療ガイドラインに対するLLMの認識を評価します。
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from jinja2 import Template


GRADER_SYSTEM_PROMPT = """あなたは日本の乳がん診療ガイドラインの専門家です。
LLMの回答を評価基準に基づいて評価してください。

評価の際は以下の点に注意してください：
1. 回答が評価基準を満たしているかどうかを判断してください
2. 薬剤名は一般名、商品名のどちらで言及されていても認めてください
3. 日本での承認状況に関する正確性を重視してください

出力形式：
各評価基準について、以下のJSON形式で回答してください：
{
  "evaluations": [
    {
      "criterion": "評価基準のテキスト",
      "satisfied": true または false,
      "reasoning": "判断の理由"
    }
  ]
}
"""


def create_grader_prompt(question: str, response: str, rubrics: list) -> str:
    """評価用プロンプトを生成する"""
    rubric_text = "\n".join([
        f"- 基準{i+1}: {r['criterion']} (配点: {r['points']}点)"
        for i, r in enumerate(rubrics)
    ])

    return f"""以下の質問に対するLLMの回答を評価してください。

## 質問
{question}

## LLMの回答
{response}

## 評価基準
{rubric_text}

各評価基準について、回答が基準を満たしているかどうかを判断し、JSON形式で出力してください。
"""


def load_data(data_path: str, num_examples: int | None = None) -> list:
    """JSONLファイルからデータを読み込む"""
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    if num_examples is not None:
        examples = examples[:num_examples]

    return examples


def get_model_response(client: OpenAI, model: str, messages: list) -> str:
    """モデルから回答を取得する"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content


def grade_response(
    client: OpenAI,
    grader_model: str,
    question: str,
    response: str,
    rubrics: list
) -> dict:
    """LLM-as-a-judgeで回答を評価する"""
    grader_prompt = create_grader_prompt(question, response, rubrics)

    grader_response = client.chat.completions.create(
        model=grader_model,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": grader_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(grader_response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        return {"evaluations": [], "error": "Failed to parse grader response"}


def calculate_score(rubrics: list, evaluations: list) -> dict:
    """スコアを計算する"""
    earned_points = 0.0
    max_positive_points = 0.0

    eval_map = {}
    for e in evaluations:
        c = e.get("criterion", "")
        # Remove "基準(number): " prefix if present
        clean_c = re.sub(r"^基準\d+:\s*", "", c)
        eval_map[clean_c] = e
        # Also keep original just in case
        eval_map[c] = e

    for rubric in rubrics:
        criterion = rubric["criterion"]
        points = rubric["points"]

        # 加点可能な最大点数を計算
        if points > 0:
            max_positive_points += points

        # 評価結果を確認
        eval_result = eval_map.get(criterion, {})
        satisfied = eval_result.get("satisfied", False)

        if satisfied:
            earned_points += points

    # スコアを計算（0以下にはならないようにする）
    final_score = max(0, earned_points)
    normalized_score = final_score / max_positive_points if max_positive_points > 0 else 0.0

    return {
        "earned_points": earned_points,
        "max_positive_points": max_positive_points,
        "final_score": final_score,
        "normalized_score": normalized_score
    }


def evaluate_example(
    client: OpenAI,
    model: str,
    grader_model: str,
    example: dict
) -> dict:
    """1つの問題を評価する"""
    prompt_id = example["prompt_id"]
    messages = example["prompt"]
    rubrics = example["rubrics"]

    # 質問テキストを取得
    question = messages[-1]["content"] if messages else ""

    # モデルから回答を取得
    try:
        response = get_model_response(client, model, messages)
    except Exception as e:
        return {
            "prompt_id": prompt_id,
            "error": f"Failed to get model response: {str(e)}",
            "response": None,
            "evaluations": [],
            "score": {"earned_points": 0, "max_positive_points": 0, "final_score": 0, "normalized_score": 0}
        }

    # 回答を評価
    try:
        grading_result = grade_response(client, grader_model, question, response, rubrics)
        evaluations = grading_result.get("evaluations", [])
    except Exception as e:
        return {
            "prompt_id": prompt_id,
            "error": f"Failed to grade response: {str(e)}",
            "response": response,
            "evaluations": [],
            "score": {"earned_points": 0, "max_positive_points": 0, "final_score": 0, "normalized_score": 0}
        }

    # スコアを計算
    score = calculate_score(rubrics, evaluations)

    return {
        "prompt_id": prompt_id,
        "question": question,
        "response": response,
        "rubrics": rubrics,
        "evaluations": evaluations,
        "score": score,
        "example_tags": example.get("example_tags", [])
    }


def run_evaluation(
    client: OpenAI,
    model: str,
    grader_model: str,
    examples: list,
    n_threads: int
) -> list:
    """全問題を並列で評価する"""
    results = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(evaluate_example, client, model, grader_model, ex): ex
            for ex in examples
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            print(f"Progress: {i+1}/{len(examples)} - {result['prompt_id']}")

    # prompt_idでソート
    results.sort(key=lambda x: x["prompt_id"])
    return results


def generate_summary(results: list, model: str, grader_model: str) -> dict:
    """評価結果のサマリーを生成する"""
    raw_total_earned = sum(r["score"]["earned_points"] for r in results)
    total_earned = max(0, raw_total_earned)
    total_max = sum(r["score"]["max_positive_points"] for r in results)

    # エラーがない結果のみでスコアを計算
    valid_results = [r for r in results if "error" not in r]

    return {
        "model": model,
        "grader_model": grader_model,
        "timestamp": datetime.now().isoformat(),
        "total_examples": len(results),
        "successful_examples": len(valid_results),
        "total_earned_points": total_earned,
        "total_max_points": total_max,
        "overall_score": total_earned / total_max if total_max > 0 else 0.0,
        "normalized_score": sum(r["score"]["normalized_score"] for r in valid_results) / len(valid_results) if valid_results else 0.0
    }


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JBC Evaluation Report - {{ summary.model }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .score-big {
            font-size: 48px;
            font-weight: bold;
            color: #667eea;
        }
        .result-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .prompt-id {
            font-weight: bold;
            color: #333;
        }
        .score-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        .score-positive {
            background-color: #d4edda;
            color: #155724;
        }
        .score-negative {
            background-color: #f8d7da;
            color: #721c24;
        }
        .score-neutral {
            background-color: #e2e3e5;
            color: #383d41;
        }
        .question, .response {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .question-label, .response-label {
            font-weight: bold;
            color: #666;
            margin-bottom: 5px;
        }
        .evaluation-item {
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid;
            background-color: #f8f9fa;
        }
        .eval-satisfied {
            border-color: #28a745;
        }
        .eval-not-satisfied {
            border-color: #dc3545;
        }
        .points {
            font-weight: bold;
        }
        .points-positive {
            color: #28a745;
        }
        .points-negative {
            color: #dc3545;
        }
        .tags {
            margin-top: 10px;
        }
        .tag {
            display: inline-block;
            background-color: #e9ecef;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 12px;
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Japanese Breast Cancer Benchmark</h1>
        <p>評価対象モデル: {{ summary.model }}</p>
        <p>評価モデル: {{ summary.grader_model }}</p>
        <p>評価日時: {{ summary.timestamp }}</p>
    </div>

    <div class="summary-card">
        <h2>評価サマリー</h2>
        <div class="score-big">{{ "%.1f"|format(summary.overall_score * 100) }}%</div>
        <p>獲得点数: {{ "%.1f"|format(summary.total_earned_points) }} / {{ "%.1f"|format(summary.total_max_points) }} 点</p>
        <p>評価問題数: {{ summary.successful_examples }} / {{ summary.total_examples }}</p>
    </div>

    <h2>詳細結果</h2>
    {% for result in results %}
    <div class="result-card">
        <div class="result-header">
            <span class="prompt-id">{{ result.prompt_id }}</span>
            {% set score_pct = result.score.normalized_score * 100 %}
            {% if score_pct >= 80 %}
            <span class="score-badge score-positive">{{ "%.0f"|format(score_pct) }}%</span>
            {% elif score_pct >= 50 %}
            <span class="score-badge score-neutral">{{ "%.0f"|format(score_pct) }}%</span>
            {% else %}
            <span class="score-badge score-negative">{{ "%.0f"|format(score_pct) }}%</span>
            {% endif %}
        </div>

        <div class="question">
            <div class="question-label">質問</div>
            {{ result.question }}
        </div>

        <div class="response">
            <div class="response-label">モデルの回答</div>
            {{ result.response | default('エラー: 回答を取得できませんでした', true) }}
        </div>

        <h4>評価結果</h4>
        {% for eval in result.evaluations %}
        <div class="evaluation-item {{ 'eval-satisfied' if eval.satisfied else 'eval-not-satisfied' }}">
            <div>
                <strong>{{ eval.criterion }}</strong>
                {% for rubric in result.rubrics %}
                    {% if rubric.criterion == eval.criterion %}
                    <span class="points {{ 'points-positive' if rubric.points > 0 else 'points-negative' }}">
                        ({{ "%+.1f"|format(rubric.points) }}点)
                    </span>
                    {% endif %}
                {% endfor %}
            </div>
            <div>結果: {{ "満たしている" if eval.satisfied else "満たしていない" }}</div>
            <div>理由: {{ eval.reasoning }}</div>
        </div>
        {% endfor %}

        <div class="tags">
            {% for tag in result.example_tags %}
            <span class="tag">{{ tag }}</span>
            {% endfor %}
        </div>
    </div>
    {% endfor %}
</body>
</html>
"""


def generate_html_report(summary: dict, results: list) -> str:
    """HTMLレポートを生成する"""
    template = Template(HTML_TEMPLATE)
    return template.render(summary=summary, results=results)


def save_results(results: list, summary: dict, model: str, output_dir: Path):
    """結果を保存する"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model.replace("/", "_").replace(":", "_")

    # 出力ディレクトリを作成
    model_dir = output_dir / model_safe
    model_dir.mkdir(parents=True, exist_ok=True)

    # JSON結果を保存
    json_path = model_dir / f"jbc_results_{model_safe}_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {json_path}")

    # HTMLレポートを保存
    html_path = model_dir / f"jbc_report_{model_safe}_{timestamp}.html"
    html_content = generate_html_report(summary, results)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML report saved to: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Japanese Breast Cancer Benchmark Evaluation"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="評価対象モデル"
    )
    parser.add_argument(
        "--data",
        default="jbc_data.jsonl",
        help="JSONLデータファイルのパス (default: jbc_data.jsonl)"
    )
    parser.add_argument(
        "--grader",
        default="gpt-4.1",
        help="評価に使用するモデル (default: gpt-4.1)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="カスタムAPIベースURL"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="APIキー (default: OPENAI_API_KEY環境変数)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="評価する問題数 (default: 全件)"
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=10,
        help="並列評価スレッド数 (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="結果の出力ディレクトリ (default: results)"
    )

    args = parser.parse_args()

    # APIキーを取得
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: APIキーが設定されていません。--api-keyオプションまたはOPENAI_API_KEY環境変数を設定してください。")
        sys.exit(1)

    # OpenAIクライアントを作成
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    # データを読み込み
    print(f"Loading data from: {args.data}")
    examples = load_data(args.data, args.num_examples)
    print(f"Loaded {len(examples)} examples")

    # 評価を実行
    print(f"Evaluating model: {args.model}")
    print(f"Using grader: {args.grader}")
    print(f"Threads: {args.n_threads}")
    print("-" * 50)

    results = run_evaluation(
        client=client,
        model=args.model,
        grader_model=args.grader,
        examples=examples,
        n_threads=args.n_threads
    )

    # サマリーを生成
    summary = generate_summary(results, args.model, args.grader)

    # 結果を表示
    print("-" * 50)
    print(f"Overall Score: {summary['overall_score']*100:.1f}%")
    print(f"Total Points: {summary['total_earned_points']:.1f} / {summary['total_max_points']:.1f}")

    # 結果を保存
    output_dir = Path(args.output_dir)
    save_results(results, summary, args.model, output_dir)


if __name__ == "__main__":
    main()
