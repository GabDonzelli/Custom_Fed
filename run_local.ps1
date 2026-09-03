# Run the simulation locally on Windows.
#
# Wraps `flwr run` with the four environment fixes this machine needs, and --
# most importantly -- derives `num-supernodes` from `num-partitions` in
# pyproject.toml so the two can never drift apart.
#
# Why each piece is here (all four were hit for real on 2026-09-02):
#
#   PATH        Flower spawns `flower-superlink` from PATH. Without .venv first,
#               it picks the system Python 3.13 / flwr 1.32.1 instead of the
#               venv's 3.11 / 1.31.0, and then `uv sync --python <3.13>` fails
#               because `ray` has no Windows wheel for 3.13.
#   UV_LINK_MODE  Flower >=1.31 builds an isolated runtime env with `uv sync`.
#               The project sits in OneDrive, which rejects hardlinks with
#               "os error 396"; copy mode sidesteps it.
#   PYTHONIOENCODING / PYTHONUTF8
#               flwr prints emoji; the default Windows cp1252 console cannot
#               encode them and the run dies with a charmap codec error.
#
# The num-supernodes match matters more than it looks: when it disagrees with
# num-partitions, every client raises in `_read_partition_config`, the strategy
# keeps the previous global model, and the run completes all its rounds writing
# an identical accuracy every time. That looks exactly like "the model did not
# improve" but means "nothing was ever trained". The `num_clients_trained`
# column in the results CSV is the tell: it reads 0 instead of a client count.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Read num-partitions straight from pyproject.toml so it cannot drift.
$pyproject = Get-Content -Raw -Path "pyproject.toml"
$match = [regex]::Match($pyproject, '(?m)^\s*num-partitions\s*=\s*(\d+)')
if (-not $match.Success) {
    throw "Could not find 'num-partitions' in pyproject.toml."
}
$numPartitions = [int]$match.Groups[1].Value

# One client at a time: DistilGPT-2 is 82M parameters, and running several
# simultaneously on CPU exhausts both RAM and threads. Give each all 8 cores.
$cpus = 8

Write-Host "num-partitions = $numPartitions  ->  num-supernodes = $numPartitions" -ForegroundColor Cyan
Write-Host "client-resources-num-cpus = $cpus (clients run one at a time)" -ForegroundColor Cyan

$env:PATH = "$PSScriptRoot\.venv\Scripts;$env:PATH"
$env:UV_LINK_MODE = "copy"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$federationConfig = "num-supernodes=$numPartitions " +
                    "client-resources-num-cpus=$cpus " +
                    "client-resources-num-gpus=0 " +
                    "init-args-num-cpus=$cpus " +
                    "init-args-num-gpus=0"

& "$PSScriptRoot\.venv\Scripts\python.exe" -m flwr.cli.app run . --stream `
    --federation-config="$federationConfig"
