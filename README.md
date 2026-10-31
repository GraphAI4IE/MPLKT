# MPLKT

Multi-Skill Interaction Modeling with Personalized Temporal Practice Load for Knowledge Tracing.

## Run

From this directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 al2005_mplkt_full.csv
```

`run.ps1` takes the CSV filename as its first argument. To use another dataset, replace the filename in the command and place the file under `data`. The script automatically switches to the project directory.

Epochs, batch size, learning rate, history length and contrastive-loss weight use the defaults in `train.py`; the other settings are explicit in `run.ps1`.

The current settings are 100 epochs, learning rate 0.0005, batch size 64, dropout 0.1, history length 20, contrastive-loss weight 0.01, and seed 42. Each invocation runs one dataset and one seed. Training selects the checkpoint with the highest validation AUC for final test evaluation.

Outputs are saved to `logs/MPLKT-<timestamp>/`:

- `MPLKT.pt`: best checkpoint.
- `log.txt`: training, validation, and final test metrics.
- `practice_load_sensitivity.json`: configuration and metrics, including `test.auc` and `test.acc`.
