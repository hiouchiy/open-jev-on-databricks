# Jev on Databricks

**Jev 型（typed-decision）モデル**を Databricks で動かすためのサンプル集です。
どのモデルも同じ **2 本立て** で構成しています。

1. **AI Runtime で動かす** — サーバレス GPU 上でモデルを動かし、バッチ推論と評価を行う
2. **Model Serving でホストする** — Unity Catalog に登録し、REST エンドポイントとして公開する

モデルは `models/<モデル名>/` ごとに独立しています。新しい実装のモデルを試すときは、フォルダを 1 つ追加するだけです（[追加方法](#新しいモデルを追加する)）。

## Jev 型モデルとは

1 つのテキスト（**state**）と、型付きの **質問** を何個でも渡します。すると、**1 回の forward** で質問ごとに
**キャリブレーション済みの確率分布** が返ります。

| 質問タイプ | 渡すもの | 返るもの |
|---|---|---|
| `choice` | 指示文 + 選択肢 | 選ばれた選択肢、確率分布、confidence |
| `score` | 指示文 + 順序付きレベル | レベルの期待値、確率分布、confidence |
| `noul` | Yes/No の文 | p(yes) |

テキストを生成しないので、出力フォーマットが崩れることはありません。LLM に任せがちなルーティング・トリアージ・属性判定を、
小さなエンコーダで高速・安価に置き換えるのが狙いです。ラベル空間が固定されていないので、
**質問を変えるだけで、再学習なしに新しい判定ができます**。

この「state + 型付き質問 → 確率分布」という形は TypeSafe AI の Jev（"System One Model"）の公開仕様にならったものです。
ここで扱うモデルは独立した再現実装で、TypeSafe とは関係ありません。

## モデル一覧

| モデル | ベース | パラメータ | AI Runtime | Model Serving | 状態 |
|---|---|---|---|---|---|
| [open-jev-deberta-v3-large](models/open-jev-deberta-v3-large/) | DeBERTa-v3-large | 435M | A10 でバッチ推論 + 評価 | RTX PRO 6000（`GPU_LARGE_RTX`）、[TypeSafe AI API 互換](models/open-jev-deberta-v3-large/02_model_serving/typesafe_api/) | ✅ 検証済み |

## リポジトリ構成

```
open-jev-on-databricks/
├── README.md                     ← このファイル（全体像・モデル一覧）
├── LICENSE                   ← MIT
├── setup.sh                      ← UC スキーマ / Volume の事前作成（全モデル共通、任意）
├── docs/
│   ├── rtx-serving.md            ← RTX PRO 6000 でサーブする仕組み（Blackwell / env_pack）
│   └── troubleshooting.md        ← 共通のつまずきポイント
└── models/
    ├── _template/                ← 新しいモデルを追加するときのひな形
    └── <モデル名>/
        ├── README.md             ← そのモデルの説明・実行手順・検証結果
        ├── 01_ai_runtime/        ← ① AI Runtime で動かす
        │   ├── notebook/         ←    Databricks ノートブック（Import → Run all）
        │   └── cli/              ←    AI Runtime CLI（air run）用の .py + .yaml
        └── 02_model_serving/     ← ② Model Serving でホストする
            ├── notebook/         ←    01_register → 02_serve_*
            └── cli/              ←    pyfunc 本体、登録スクリプト、クエリ用スクリプト
```

どのステップにも **ノートブック版** と **CLI 版** があり、どちらも**コードを変更せずに**実行できます
（設定はノートブックのウィジェット、または YAML の `env_variables` で変更します）。

## 前提

- Databricks ワークスペース（Unity Catalog、サーバレス GPU / AI Runtime が有効なもの）
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/)：`databricks auth login --host <workspace-url> --profile <profile>`
- [AI Runtime CLI](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/)（`air`）：CLI 版を使う場合のみ
- macOS で `air run` する場合
  - `brew install gnu-tar` のうえで `export PATH="$(brew --prefix gnu-tar)/libexec/gnubin:$PATH"`
  - 実行時に `COPYFILE_DISABLE=1` を付ける
- Unity Catalog の書き込み先（既定は `main.air_samples`、スキーマ・Volume は自動作成）
  - 変更する場合、ノートブックはウィジェットで、CLI は `--override env_variables.UC_CATALOG=<catalog>` で指定します。

## クイックスタート（open-jev-deberta-v3-large）

```bash
cd models/open-jev-deberta-v3-large

# ① AI Runtime: GPU バッチ推論 + 評価（A10、約 3 分）
(cd 01_ai_runtime/cli && COPYFILE_DISABLE=1 air run --file batch_inference.yaml --watch --profile <profile>)

# ② Model Serving: ノートブックで登録 → RTX エンドポイント作成 → クエリ
#    02_model_serving/notebook/01_register.py と 02_serve_rtx.py をワークスペースに Import して実行
python 02_model_serving/cli/query_endpoint.py --profile <profile> --endpoint open-jev-rtx
```

詳細は [モデルの README](models/open-jev-deberta-v3-large/README.md) を参照してください。

## 新しいモデルを追加する

1. `models/_template/` を `models/<新しいモデル名>/` にコピーする
2. ① `01_ai_runtime/` にモデルの読み込みとバッチ推論を書く
   - 評価データには、可能ならモデル作者が公開しているテストセットを使います。そうすればモデルカードの数値と比べられます。
3. ② `02_model_serving/` に pyfunc を書く
   - **入出力の形を既存モデルとそろえると、クライアントや比較スクリプトを使い回せます。**
   - 入力: `state`、`questions`（JSON 文字列）
   - 出力: `answers`（JSON）、`latency_ms`、`error`
4. モデルの README に検証結果を書き、上の「モデル一覧」に 1 行追加する

どの GPU でサーブするかはモデルの大きさと要件で選びます（[docs/rtx-serving.md](docs/rtx-serving.md) に workload type の一覧があります）。

## ライセンス / 出典

- このリポジトリのコードとドキュメント: [MIT](LICENSE)
- 各モデルのライセンスは、それぞれのモデル README とモデルカードに従います。
- open-jev-deberta-v3-large: Apache-2.0（Kotoba Labs / Mithril）
- 評価データと推論コード: [kotoba-lang/typed-decisions](https://github.com/kotoba-lang/typed-decisions)（Apache-2.0）
