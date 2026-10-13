# TypeSafe AI API 互換で提供する（SDK ごと差し替え）

open-jev を **TypeSafe AI の System One API と同じ仕様** で Databricks 上に公開します。
公式の Python SDK（[`typesafe-sdk`](https://docs.typesafe.ai/sdk/python/usage)）の **`base_url` と `api_key` を変えるだけで**、
既存の TypeSafe 向けコードがそのまま Databricks 上の open-jev を呼ぶようになります。

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

client = TypeSafeClient(
    base_url="https://<app-name>-<workspace-id>.aws.databricksapps.com",  # ← ゲートウェイ（Databricks App）
    api_key="<Databricks の OAuth トークン>",                              # ← databricks auth token
)
r = client.system_one(state="I was charged twice. Please refund the duplicate charge.",
                      questions={"refund": Noul(instructions="Is the customer requesting a refund?")})
print(r.nouls["refund"].noul)
```

## 構成

```
TypeSafe クライアント（typesafe-sdk / curl）
   │  POST /v1/systemone, GET /v1/models   Authorization: Bearer <Databricks OAuth トークン>
   ▼
Databricks App「typesafe_gateway」（FastAPI）── ルーティング・エラーの変換・認証は Apps が担当
   │  POST /serving-endpoints/<endpoint>/invocations（App のサービスプリンシパルで呼ぶ）
   ▼
Model Serving（GPU_LARGE_RTX など）── typesafe_pyfunc.py + typesafe_adapter.py
   │  TypeSafe の質問 → open-jev の選択肢テキストに変換し、1 回のバッチ forward で採点
   ▼
open-jev-deberta-v3-large
```

Model Serving だけでは URL（`/serving-endpoints/<name>/invocations`）とリクエスト・レスポンスの包み
（`dataframe_records` / `predictions`）が固定されているので、TypeSafe と同じパスを生やすためにゲートウェイを置いています。
変換のルールはすべてモデル側（`typesafe_adapter.py`）にあり、ゲートウェイは中身を見ずに中継するだけです。

| ファイル | 役割 |
|---|---|
| `typesafe_adapter.py` | TypeSafe のリクエスト ⇄ open-jev の変換と、422 のバリデーション（torch / MLflow に依存しない） |
| `typesafe_pyfunc.py` | サーブ用の pyfunc。1 行 = 1 リクエスト（`request` 列に JSON 文字列）、出力は `status` / `body` |
| `register_typesafe.py` | ノートブック。torch cu128 を入れて env_pack 付きで UC に登録（RTX でもサーブできる） |
| `typesafe_gateway/` | Databricks App（`app.py`、`app.yaml`、`requirements.txt`） |
| `eval_criteria_modes.py` + `.yaml` | `criteria` の説明文の渡し方による精度の違いを測る（AI Runtime CLI） |
| `sdk_smoke_test.py` | 公式 SDK から呼んで動作とレイテンシを確認する |

## 対応している仕様

公式の API リファレンス（[docs.typesafe.ai/api](https://docs.typesafe.ai/api)、[OpenAPI](https://api.typesafe.ai/openapi.json)）と、
`typesafe-sdk` 0.7.3 のレスポンス型（strict モードで検証される）に合わせています。

| 項目 | 対応 |
|---|---|
| `POST /v1/systemone` | ✅ `model` / `state`（文字列・オブジェクト・配列） / `questions`（名前 → 質問） |
| `GET /v1/models` | ✅ `MODEL_ALIASES`（既定は `jev-latest` と `open-jev-deberta-v3-large`） |
| `noul` | ✅ `noul` = p(yes)。`criteria.true` / `false` にも対応（下記） |
| `choice` | ✅ `choice`（名前）、`probabilities`（名前 → 確率）、`confidence`。説明が `null` の選択肢も可 |
| `score` | ✅ `score`（期待値）、`legend`、`probabilities`（`"0"`, `"1"`, ... → 確率）、`confidence` |
| `usage` | `input_tokens` は実際のトークン数。生成しないので `output_tokens` は 0 |
| エラー | 422 `{"detail": [{"loc", "msg", "type"}]}`、429（エンドポイント側のレート制限）、503（エンドポイントが使えないとき） |

open-jev 固有の制約: `choice` の選択肢は 255 個まで、`score` のレベルは 10 個まで、入力全体で 512 tokens まで
（`state` は 256 tokens で切り詰め）、英語での学習。これを超えるリクエストは 422 を返します。

## `criteria` の説明文の扱い（実測にもとづく既定値）

TypeSafe では選択肢に「名前 + 説明文」を付けますが、open-jev は「fees & charges」のような**短いラベルで学習されています**。
そこで、モデル作者の in-domain テストセット（banking77 / SST-5）を TypeSafe 形式に書き直し、説明文を手で書いて
渡し方を比べました（`eval_criteria_modes.py`、A10G、2026-10）。

| 質問タイプ | 渡し方 | accuracy |
|---|---|---|
| choice（2,000 問） | 学習時のラベルそのまま（参考） | 0.866 |
| | **名前だけ**（`choice_text_mode="name"`、既定） | **0.855** |
| | 名前: 説明文（`name_description`） | 0.807 |
| | 説明文だけ（`description`） | 0.745 |
| noul（2,000 問） | **criteria を使わない**（既定。instructions がないときだけ使う） | **0.935** |
| | true / false の説明を 2 択にする（`noul_criteria_mode="options"`） | 0.914 |
| score（1,000 問） | レベルのラベル（参考） | 0.590 |
| | レベルの説明文 | 0.525 |

- **説明文を足すと精度が下がりました。** そのため既定値は、choice は名前だけ、noul の criteria は instructions がないときだけ使う、にしています。
  名前は `_` を空白に置き換えて渡すので、`fees_and_charges` のように**意味の分かる名前を付ける**のがコツです。
- score の criteria は仕様上説明文そのものなので、選びようがありません。短いラベル（"very negative" など）にすると精度が上がります。
- どちらも `register_typesafe.py` のウィジェット（`choice_text_mode` / `noul_criteria_mode`）で変えられます。
- **仕様が同じでも、答えまで公式の Jev と同じになるわけではありません。** 中身は open-jev なので、手元のデータで評価してから使ってください。

## 手順

1. **フォルダをワークスペースに置く**
   - リポジトリを Git フォルダとしてクローンするか、次のコマンドで取り込む
   ```bash
   databricks workspace import-dir typesafe_api /Workspace/Users/<you>/jev/typesafe_api --overwrite --profile <profile>
   ```
2. **登録:** `register_typesafe` ノートブックを Serverless GPU（A10）で実行する（約 10 分）
   - ヘッドレスで実行する場合:
   ```bash
   databricks jobs submit --profile <profile> --json '{
     "run_name": "open-jev-register-typesafe",
     "tasks": [{"task_key": "register",
       "notebook_task": {"notebook_path": "/Workspace/Users/<you>/jev/typesafe_api/register_typesafe",
                         "base_parameters": {"uc_catalog": "<catalog>"}},
       "environment_key": "gpu", "compute": {"hardware_accelerator": "GPU_1xA10"}}],
     "environments": [{"environment_key": "gpu", "spec": {"client": "4"}}]}'
   ```
3. **エンドポイント作成**（RTX の例）
   ```bash
   databricks serving-endpoints create --profile <profile> --json '{
     "name": "open-jev-typesafe",
     "config": {"served_entities": [{
       "entity_name": "<catalog>.air_samples.open_jev_deberta_v3_large_typesafe", "entity_version": "1",
       "workload_type": "GPU_LARGE_RTX", "workload_size": "Small", "scale_to_zero_enabled": true}]}}'
   ```
4. **ゲートウェイ（Databricks App）を作ってデプロイ**
   ```bash
   databricks apps create --profile <profile> --json '{
     "name": "open-jev-typesafe-gw",
     "resources": [{"name": "serving-endpoint",
                    "serving_endpoint": {"name": "open-jev-typesafe", "permission": "CAN_QUERY"}}]}'
   databricks workspace import-dir typesafe_gateway /Workspace/Users/<you>/jev/typesafe_gateway --overwrite --profile <profile>
   databricks apps deploy open-jev-typesafe-gw --source-code-path /Workspace/Users/<you>/jev/typesafe_gateway --profile <profile>
   ```
   - リソース名は `serving-endpoint` にする（`app.yaml` の `valueFrom` と同じ）
5. **公式 SDK で確認**
   ```bash
   pip install typesafe-sdk
   python sdk_smoke_test.py --profile <profile> --app-url https://<app-name>-<workspace-id>.aws.databricksapps.com
   ```

## 検証結果（ap-northeast-1、`GPU_LARGE_RTX` Small、`typesafe-sdk` 0.7.3）

- 公式 SDK の各機能が、コードを変えずに動くことを確認
  - `models.list()`
  - `system_one()`（noul / choice / score、文字列とオブジェクトの `state`）
  - 422 のときに `TypeSafeUnprocessableEntityError` になること
- TypeSafe のドキュメントに載っている curl の例も、そのまま TypeSafe 形式のレスポンスを返す
- **SDK からの往復（ゲートウェイ経由、3 問）: p50 80 ms / 最大 167 ms**（日本 → 東京リージョン、30 回）
- 回答例: 「I was charged twice. Please refund the duplicate charge.」
  - refund p(yes) = 0.917
  - department = `billing`（0.805）
  - frustration score = 0.91（「Frustrated but civil」寄り）

## 注意

- **認証は Databricks の OAuth トークンです。** `api_key` に `databricks auth token` で取ったアクセストークンを渡します。
  - トークンは約 1 時間で切れるので、長く動かす場合は呼び出し側で更新してください（サービスプリンシパルの OAuth M2M なども使えます）。
  - トークンなしのリクエストは、Apps がログイン画面へのリダイレクト（302）を返します。TypeSafe の 401 とは異なります。
- **scale-to-zero の場合、最初のリクエストはエンドポイントの起動待ちで数分かかります。** SDK の `timeout` を長めにするか、本番では scale-to-zero を切ってください。
- エンドポイントを直接呼ぶこともできます。その場合は `{"dataframe_records": [{"request": "<TypeSafe のリクエスト JSON>"}]}` を送ると、
  `predictions[0].status` と `predictions[0].body` が返ります。
