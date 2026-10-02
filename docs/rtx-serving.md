# RTX PRO 6000（`GPU_LARGE_RTX`）でサーブする

## GPU の workload type

| Model Serving `workload_type` | GPU | メモリ | DBU/時 | 備考 |
|---|---|---|---|---|
| `GPU_SMALL` | 1x T4 | 16 GB | 10.48 | Turing。bf16 と FlashAttention は使えない（fp16 で動かす） |
| `GPU_MEDIUM` | 1x A10G | 24 GB | 20 | 汎用 |
| **`GPU_LARGE_RTX`** | **1x RTX PRO 6000 Blackwell** | **96 GB** | 80（実測） | `us-west-2`, `us-east-1`, `us-east-2`, `ap-northeast-1`（東京）, `ap-northeast-2` |
| `GPU_XLARGE` | 1x H100 | 80 GB | 100 | `us-west-2` のみ、要 enrollment、scale-to-zero 不可（Beta） |

出典: [Serve custom LLMs](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms)、[Model Serving pricing](https://www.databricks.com/product/pricing/model-serving)。
RTX の DBU は料金ページに載っていないため、`system.billing.usage` で測った値です（2026-10、ap-northeast-1）。
GPU ごとの性能とコストの比較例は [open-jev-deberta-v3-large の README](../models/open-jev-deberta-v3-large/README.md#gpu-別の比較t4--a10g--rtx-pro-6000) を参照してください。

AI Runtime（AIR CLI）の `GPU_1xA10` は名前どおり A10 です。一方、Model Serving の `GPU_SMALL` は **T4** なので、名前の付け方が違う点に注意してください。

## Blackwell には CUDA 12.8 以降の torch が必要

RTX PRO 6000 は Blackwell 世代（compute capability 12.0 / `sm_120`）です。PyTorch が Blackwell 用のカーネルを含むのは
**CUDA 12.8 以降のビルド（cu128）** だけです。

| 環境 | torch | `sm_120` |
|---|---|---|
| AI Runtime env "4"（標準） | 2.7.1+cu126 | ❌（`sm_90` まで） |
| 本リポジトリのサーブ環境 | 2.10.0+cu128 | ✅ |

確認方法: `torch.cuda.get_arch_list()` の結果に `sm_120` が含まれていればよい（登録ノートブックは含まれていることを assert しています）。

## env_pack（GPU Express Deployment）を使う

標準のサーブではコンテナを再ビルドするため、意図しない torch に差し替えられることがあります。そこで
`mlflow.register_model(..., env_pack="databricks_model_serving")` を使い、**登録時の Python 環境をまるごとパックして、
サーブ時にそのまま復元**させます。

- 登録するノートブックで、torch cu128 と、それに合った `torchvision` を `%pip` で入れる
  - サーバレス GPU のイメージにはベースの torch 向けの torchvision が入っています。torch だけ上げると ABI が合わず、`transformers` の import で壊れます。
- `mlflow==3.12.0` と `databricks-sdk>=0.102.0` を入れる。これより古い SDK だと、数 GB あるパック済み環境のアップロードがタイムアウトします。
- `log_model` は `registered_model_name` を付けずに実行し、そのあと `register_model(..., env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False))` で登録する
- 重みは MLflow アーティファクトとして同梱する。こうしておくとサーブ時に Hugging Face へ通信せずに済み、リビジョンも固定できます。

## 既知の問題: AI Runtime CLI ジョブから登録したバージョンはデプロイが進まない

AIR CLI ジョブ（`air run`）の中で同じ env_pack 登録を行っても、登録そのものは成功します。ところが、
**そのバージョンでエンドポイントを作ると "Creating resources for served entity." から 10 時間以上進みません**。
A10G と RTX、2 つのワークスペースで再現しました。

同じコードを**ノートブック**（UI から実行、または `databricks jobs submit` でノートブックタスクとして実行）で登録したバージョンは、
数秒で Provisioning に進み、約 3 分で READY になりました。

→ **サーブ用の登録はノートブックで行ってください。** 原因はまだ特定できていません（2026-10 時点）。
CLI 版の `register.yaml` は再現用として残しています。

## RTX の空き（キャパシティ）

RTX は**リージョンや時間帯によって空きがない**ことがあります。その場合、約 2 時間後に
`DEPLOYMENT_ABORTED: ... the resource did not become available in time` で失敗します（us-west-2 で実際に発生）。
東京（ap-northeast-1）のワークスペースでは、作成から約 3 分で READY になりました。
