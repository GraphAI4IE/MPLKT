[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$DataFile
)

$ErrorActionPreference = 'Stop'
$trainingExitCode = 1

Push-Location -LiteralPath $PSScriptRoot
try {
    $dataPath = Join-Path -Path 'data' -ChildPath $DataFile
    if (-not (Test-Path -LiteralPath $dataPath -PathType Leaf)) {
        throw "Data file not found: $dataPath"
    }

    # Use train.py defaults for epochs, batch size, learning rate, history K and CL weight.
    & python -u train.py `
        --data-dir data `
        --data-file $DataFile `
        --save-dir logs `
        --max-seq-len 200 `
        --min-seq-len 5 `
        --train-ratio 0.6 `
        --val-ratio 0.2 `
        --shuffle True `
        --seed 42 `
        --lr-decay 200 `
        --gamma 0.5 `
        --early-stop-patience 0 `
        --hid-dim 32 `
        --emb-dim 32 `
        --q-graph-dim 64 `
        --qs-gnn-layers 2 `
        --dropout 0.1 `
        --fatigue-dim 16 `
        --fatigue-hidden-dim 64 `
        --fatigue-time-threshold-quantile 0.5 `
        --fatigue-decay-lambda 0.5 `
        --practice-load-time-source current `
        --history-gru-hidden-dim 32 `
        --contrastive-neg-k 10 `
        --contrastive-tau 0.2 `
        --use-fatigue True `
        --use-qs-structural-learning True `
        --skill-update-mode combo
    $trainingExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $trainingExitCode
