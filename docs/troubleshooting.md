# トラブルシューティング（全モデル共通）

| 症状 | 原因 / 対処 |
|---|---|
| `Fast download using 'hf_transfer' is enabled ... not available` | AI Runtime は `HF_HUB_ENABLE_HF_TRANSFER=1` を設定しますが、`hf_transfer` は入っていません。`huggingface_hub` を import する**前に** `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"` を設定してください（`setdefault` では上書きされないので不可） |
| `tar: Option --anchored is not supported`（macOS） | `air` は GNU tar のオプションを使います。`brew install gnu-tar` して GNU tar を PATH の先頭に置いてください |
| `environment.version requires inline dependencies` | AIR の YAML では、`version` を書く場合に `dependencies` が 1 件以上必要です |
| AIR ジョブが長く RUNNING のまま | GPU の空き待ちのことが多いです。`databricks jobs get-run-output <task_run_id>` の `ai_runtime_task_output.status_message` に `Waiting for GPU compute capacity` と出ていないか確認してください |
| `Error: Could not find or load main class org.apache.spark.launcher.Main` | GPU ノード上で MLflow が Spark を探しに行くときの無害な出力です。処理には影響しません |
| `Py4JSecurityException ... not whitelisted` | サーバレス環境での MLflow のタグ取得で出る無害な警告です。スクリプトでは該当ロガーを抑制しています |
| `/Volumes/...` への書き込みで `Operation not supported` | Volume は mkdir では作れません。SDK（`w.volumes.create(..., volume_type=VolumeType.MANAGED)`）か `setup.sh` で作成してください。また、ローカルに書いてからコピーしてください（FUSE は seek を伴う書き込みに弱いため） |
| `serving-endpoints query` が 60 秒でタイムアウトする | CLI 側のタイムアウトです。スケールゼロからのコールドスタートを待てないので、各モデルの `query_endpoint.py`（900 秒）か curl を使ってください |
| エンドポイントが "Creating resources" から進まない | AIR CLI ジョブから env_pack 登録したバージョンです → [rtx-serving.md](rtx-serving.md#既知の問題-ai-runtime-cli-ジョブから登録したバージョンはデプロイが進まない) |
| `DEPLOYMENT_ABORTED ... did not become available in time` | GPU の空きがありません。別リージョンのワークスペースで試すか、時間を置いてください |
| サーブ時に `no kernel image is available` / `sm_120` 関連のエラー | Blackwell に対応していない torch が入っています → [rtx-serving.md](rtx-serving.md) |
