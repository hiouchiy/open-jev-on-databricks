# open-jev-deberta-v3-large

[`com-kotobalabs/open-jev-deberta-v3-large`](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)
— DeBERTa-v3-large（435M パラメータ）をベースにした Jev 型の typed-decision モデルです。

| 項目 | 内容 |
|---|---|
| ベース | `microsoft/deberta-v3-large` |
| ライセンス | Apache-2.0（Kotoba Labs / Mithril） |
| 言語 / 長さ | 英語のみ / 512 tokens（state は 256 tokens まで） |
| 質問 | `choice` は選択肢 2〜255 個、`score` は 2〜10 レベル、`noul` は Yes/No |
| 学習データ | banking77 / SST-5 / BoolQ（公開されている正解ラベルのみ） |
| 推論コード | スナップショットに同梱されている `typed_decisions/`（`pip install` 不要）。Hub リビジョン `188ee67a5c` に固定 |

## 構成

```
01_ai_runtime/                 ① AI Runtime で動かす
  notebook/batch_inference.py      ノートブック版（Serverless GPU A10, AI v5）
  cli/batch_inference.py + .yaml   CLI 版（air run）
02_model_serving/              ② Model Serving でホストする
  notebook/01_register.py          pyfunc 化 + env_pack 付きで UC に登録 ← サーブ用はこちら
  notebook/02_serve_rtx.py         GPU_LARGE_RTX エンドポイントの作成 / 更新 + クエリ
  cli/open_jev_pyfunc.py           pyfunc 本体（ノートブック内のものと同じコード）
  cli/register.py + register.yaml  CLI 版の登録（⚠ 既知の問題あり。下記参照）
  cli/query_endpoint.py            ローカルからのクエリ（標準ライブラリのみ）
  cli/benchmark_endpoints.py       GPU 別の精度・レイテンシ・スループット・コスト比較（標準ライブラリのみ）
  benchmark_results.json           上記の計測結果（T4 / A10G / RTX）
  typesafe_api/                    TypeSafe AI API 互換で提供する（公式 SDK ごと差し替え）
```

---

## ① AI Runtime で動かす — GPU バッチ推論 + 評価

モデル作者が公開しているテストセット（[kotoba-lang/typed-decisions](https://github.com/kotoba-lang/typed-decisions) `e65215d`、
`data-fam/{test,ood-test}.jsonl`、正解ラベル付き）を bf16 でバッチ推論します。
accuracy / Brier / ECE（全体・質問タイプ別・データセット別）とスループットを MLflow に記録し、全予測を UC Volume に Parquet で書き出します。

- `test` … 学習時と同じ種類の質問（in-domain）。テキスト 3,000 件、質問 7,000 問
- `ood-test` … 同じテキストに対する**未見の指示文・選択肢セット**（out-of-domain）。テキスト 3,000 件、質問 8,000 問

**ノートブック:** `01_ai_runtime/notebook/batch_inference.py` を Import する。
コンピュートで **Serverless GPU** を選び、環境パネルで Accelerator **A10**、Base environment **AI v5** にして Apply し、Run all。

**CLI:**
```bash
cd 01_ai_runtime/cli
COPYFILE_DISABLE=1 air run --file batch_inference.yaml --watch --profile <profile>
#   スモーク（各 split 100 件）:  --override env_variables.MAX_STATES=100
#   カタログ変更:                --override env_variables.UC_CATALOG=<catalog>
```

**検証結果（1x A10G、bf16、全件）:**

| split | 質問数 | accuracy | ECE | 参考: モデルカード |
|---|---|---|---|---|
| test（in-domain） | 7,000 | **0.861** | 0.015 | 0.854 / 0.022 |
| ood-test | 8,000 | **0.691** | 0.032 | 0.690 / 0.035 |

- in-domain の内訳
  - データセット別: banking77 0.942 / BoolQ 0.878 / SST-5 0.775
  - 質問タイプ別: `choice` 0.891 / `noul` 0.920 / `score` 0.594
- 所要約 2.5 分（ダウンロード込み）、ピーク GPU メモリ 7.6 GiB、約 100〜250 questions/s
- bf16 は fp32 の約 2 倍速く、精度は同じ
- ECE が小さい（キャリブレーションが良い）ので、`confidence` をしきい値にして「自動で処理する / 人に回す」を振り分けられます。

---

## ② Model Serving でホストする — RTX PRO 6000

RTX PRO 6000 は Blackwell 世代なので、torch 2.10（cu128）を env_pack で同梱して登録します。
仕組みは [docs/rtx-serving.md](../../docs/rtx-serving.md) を参照してください。

### pyfunc の入出力

- **入力**（1 行 = 1 state）
  - `state`: 文字列
  - `questions`: 型付き質問のリストを JSON 文字列にしたもの。行ごとに質問の数や種類が違ってかまいません。
- **出力**
  - `answers`: 質問ごとの回答（JSON）
  - `latency_ms`
  - `error`: 行単位のバリデーションエラー。例外にはせず返すので、1 行が不正でもリクエスト全体は失敗しません。
- リクエスト内の全行を **1 回のバッチ forward**（GPU では bf16 autocast）で処理します。
- 重みと推論コード（1.7 GB）はアーティファクトとして同梱しているので、サーブ時に Hugging Face へは通信しません。

### 手順

1. **登録（ノートブック）:** `02_model_serving/notebook/01_register.py` を Serverless GPU（A10）で実行する（約 10 分）。
   ヘッドレスで実行する場合:
   ```bash
   databricks workspace import /Workspace/Users/<you>/jev/01_register \
     --file 02_model_serving/notebook/01_register.py --language PYTHON --format SOURCE --overwrite --profile <profile>
   databricks jobs submit --profile <profile> --json '{
     "run_name": "open-jev-register",
     "tasks": [{"task_key": "register",
       "notebook_task": {"notebook_path": "/Workspace/Users/<you>/jev/01_register",
                         "base_parameters": {"uc_catalog": "<catalog>"}},
       "environment_key": "gpu", "compute": {"hardware_accelerator": "GPU_1xA10"}}],
     "environments": [{"environment_key": "gpu", "spec": {"client": "4"}}]}'
   ```
   > ⚠️ CLI 版の `02_model_serving/cli/register.yaml` でも登録はできます。ただし、そのバージョンは**デプロイが "Creating resources" から進みません**
   > （[既知の問題](../../docs/rtx-serving.md#既知の問題-ai-runtime-cli-ジョブから登録したバージョンはデプロイが進まない)）。サーブ用の登録はノートブックで行ってください。

2. **エンドポイント作成:** `02_model_serving/notebook/02_serve_rtx.py`（CPU のサーバレスで可）を実行する。最新バージョンを使い、
   `GPU_LARGE_RTX` / Small / scale-to-zero のエンドポイント `open-jev-rtx` を作成（既存なら更新）します。READY になるまで待ってからクエリを送ります。
   Databricks CLI で作成する場合:
   ```bash
   databricks serving-endpoints create --profile <profile> --json '{
     "name": "open-jev-rtx",
     "config": {"served_entities": [{
       "entity_name": "<catalog>.air_samples.open_jev_deberta_v3_large", "entity_version": "<version>",
       "workload_type": "GPU_LARGE_RTX", "workload_size": "Small", "scale_to_zero_enabled": true}]}}'
   ```

3. **クエリ:**
   ```bash
   python 02_model_serving/cli/query_endpoint.py --profile <profile> --endpoint open-jev-rtx --bench 50
   ```
   curl の場合:
   ```bash
   curl -s -X POST "https://<workspace-host>/serving-endpoints/open-jev-rtx/invocations" \
     -H "Authorization: Bearer $(databricks auth token --profile <profile> | jq -r .access_token)" \
     -H "Content-Type: application/json" \
     -d '{"dataframe_records":[{"state":"Someone used my card abroad. I did not make that payment!",
          "questions":"[{\"type\":\"choice\",\"instructions\":\"Which team should handle this?\",\"options\":[\"billing\",\"fraud & security\",\"card delivery\"]},{\"type\":\"noul\",\"instructions\":\"The message needs an urgent response.\"}]"}]}'
   ```

### 検証結果（ap-northeast-1 のワークスペース、`GPU_LARGE_RTX` Small）

- エンドポイント作成から READY まで**約 3 分**（Provisioning 2 分 → Deploying 1 分）
- 1 テキスト × 4 問のリクエストを 50 回
  - サーバー側: **p50 21.7 ms** / p95 90.8 ms
  - 日本からの往復: p50 173 ms / p95 292 ms
- 回答例
  - 「Someone used my card in another country...」→ `fraud & security`（conf 0.73）、urgent p(yes)=0.72
  - 「I was charged twice ... money back」の担当チーム → `fraud & security`（conf 0.45。本来は billing）
  - 学習分布外の質問は精度が落ちます。**confidence が低いものは人に回す**運用を前提にしてください。

### GPU 別の比較（T4 / A10G / RTX PRO 6000）

同じモデル（v3）を 3 種類の GPU にデプロイし、`02_model_serving/cli/benchmark_endpoints.py` で同じ条件で測りました。
条件は ap-northeast-1、workload size Small（同時 4 リクエスト）、日本からのクライアントです。生データは `02_model_serving/benchmark_results.json` にあります。

| | T4（`GPU_SMALL`） | A10G（`GPU_MEDIUM`） | RTX PRO 6000（`GPU_LARGE_RTX`） |
|---|---|---|---|
| 精度（in-domain 692 問） | 0.834（fp16） | 0.834（bf16） | 0.834（bf16） |
| レイテンシ（1 テキスト×4 問、サーバー側 p50 / p95） | 60.1 / 77.5 ms | 44.2 / 53.2 ms | **21.1 / 23.5 ms** |
| 往復 p50（日本→東京） | 216 ms | 203 ms | 174 ms |
| スループット（32 テキスト×同時 4） | 56 q/s | 118 q/s | **379 q/s** |
| 対 T4 のスループット | 1.0× | 2.1× | 6.8× |
| DBU / 時 | 10.48 | 20 | 80 |
| $/時（$0.09/DBU、東京リスト価格） | **$0.94** | $1.80 | $7.20 |
| フル稼働時の $ / 100 万問 | $4.65 | **$4.24** | $5.28 |
| デプロイ所要時間（作成→READY） | 7 分 | 9 分 | 3 分 |

- **精度は同じ。** T4 は bf16 を持たないので pyfunc が自動で fp16 に切り替えますが、精度は落ちません。
- **スループットは各 GPU の上限で測れています。** バッチを 64 テキストにしても q/s は変わりませんでした（T4 55 / A10G 116 / RTX 377）。
- **DBU の出所**
  - RTX: `system.billing.usage` からの実測（80 DBU/時）
  - T4 / A10G: [Model Serving の料金ページ](https://www.databricks.com/product/pricing/model-serving)の値
  - いずれもディスカウント前のリスト価格です。$/DBU はリージョンとプランで変わるので、`system.billing.list_prices` の `*_SERVERLESS_REAL_TIME_INFERENCE_*` を確認してください。

**ROI の見方（この 435M モデルの場合）**

- **質問あたりのコストが最も安いのは A10G**（$4.24 / 100 万問）。T4 は約 1.1 倍、RTX は約 1.25 倍です。
  - 負荷が十分にあってレプリカをフル稼働させられるなら、A10G が最も費用対効果が高くなります。
- **時間あたりの料金が最も安いのは T4**（$0.94/時）。
  - トラフィックが少ない・まばらで GPU が遊んでいる時間が長い場合は、T4 が有利です。
  - 例: 10 q/s の常時負荷なら、どの GPU でも 1 レプリカで足ります。月額は T4 約 $690 / A10G 約 $1,300 / RTX 約 $5,300 です。
- **RTX が効くのは次のようなとき**
  - **レイテンシの SLA が厳しいとき**: サーバー側 21 ms は A10G の約半分、T4 の約 1/3。
  - **1 レプリカで大量の負荷をさばきたいとき**: A10G 3.2 台分。
  - **今後、より大きい Jev 実装を載せるとき**: 96 GB。
- 必要なレプリカ数の目安は `ceil(ピーク q/s ÷ 1 レプリカの q/s)` です。
  - 例: ピーク 300 q/s なら、T4 は 6 台（$5.6/時）、A10G は 3 台（$5.4/時）、RTX は 1 台（$7.2/時）。
- scale-to-zero を使うと、アイドル時は 0 です。ただし、起動したらしばらくは課金が続く点に注意してください。

再計測:
```bash
python 02_model_serving/cli/benchmark_endpoints.py --profile <profile> \
  --endpoint T4=<t4-endpoint> --endpoint A10G=<a10-endpoint> --endpoint RTX=<rtx-endpoint> \
  --dbu T4=10.48 --dbu A10G=20 --dbu RTX=80 --usd-per-dbu <your $/DBU>
```

### TypeSafe AI API 互換で提供する

TypeSafe AI の System One API（`POST /v1/systemone`）と同じ仕様で公開することもできます。
公式の `typesafe-sdk` の `base_url` と `api_key` を変えるだけで、既存のコードがそのまま Databricks 上の open-jev を呼びます。
構成は Model Serving と Databricks App のゲートウェイの 2 段です。
手順、`criteria` の説明文の扱いの評価結果、検証結果は [typesafe_api/README.md](02_model_serving/typesafe_api/README.md) を参照してください。

---

## 制約（モデルカードより）

- 質問は部分的にしか読めていません。in-domain 0.85 に対して OOD 0.69 で、とくに未見の順序スケール（`score`）は多数派クラスをやや上回る程度です。
- 英語のみで、学習ドメインは 3 つ（銀行サポート、映画レビュー、Wikipedia の Yes/No）です。それ以外のドメインでは、使う前に評価してください。
- OOD では confidence が平均 0.03 ほど過信気味です。自分のデータで再キャリブレーションすることを推奨します。
