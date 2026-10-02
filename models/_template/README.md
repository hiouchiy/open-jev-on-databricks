# <model-name>

> このフォルダは新しいモデルを追加するときのひな形です。`models/<model-name>/` にコピーして中身を書き換えてください。

[`<hf-org>/<hf-repo>`](https://huggingface.co/<hf-org>/<hf-repo>) — 1 行の説明

| 項目 | 内容 |
|---|---|
| ベース | |
| ライセンス | |
| 言語 / 長さ | |
| 質問 | choice / score / noul の対応範囲 |
| 推論コード | 入手方法（スナップショット同梱 / pip / git）とリビジョン |

## 構成

```
01_ai_runtime/
  notebook/batch_inference.py      ノートブック版
  cli/batch_inference.py + .yaml   CLI 版（air run、YAML の root_path は "."）
02_model_serving/
  notebook/01_register.py          pyfunc 化 + UC 登録（GPU で env_pack が必要ならここで）
  notebook/02_serve_<gpu>.py       エンドポイント作成 + クエリ
  cli/<model>_pyfunc.py            pyfunc 本体
  cli/query_endpoint.py            クエリ用（入出力の形をそろえれば既存のものを流用できます）
```

## ① AI Runtime で動かす

- 評価データ（できればモデル作者が公開しているテストセット）と、使う GPU
- 実行手順（ノートブック / CLI）
- **検証結果**: accuracy / ECE / スループット / ピークメモリ、モデルカードとの比較

## ② Model Serving でホストする

- workload type の選び方（[docs/rtx-serving.md](../../docs/rtx-serving.md)）
- pyfunc の入出力
  - 既存モデルとそろえることを推奨: 入力 `state`、`questions`（JSON 文字列）、出力 `answers`、`latency_ms`、`error`
- 手順（登録 → エンドポイント作成 → クエリ）
- **検証結果**: READY までの時間、レイテンシ（p50 / p95）、回答例

## 制約
