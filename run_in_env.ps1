$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "E:\conda_env\mujoco_rl\python.exe"
$env:TMP = Join-Path $ProjectRoot ".tmp"
$env:TEMP = $env:TMP
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot ".pip_cache"
$env:TORCH_HOME = Join-Path $ProjectRoot ".torch"

New-Item -ItemType Directory -Force -Path $env:TMP, $env:PIP_CACHE_DIR, $env:TORCH_HOME | Out-Null

if ($args.Count -lt 1) {
    throw "Usage: .\run_in_env.ps1 <script> [args...]"
}

$ScriptPath = $args[0]
$ScriptArgs = @()
if ($args.Count -gt 1) {
    $ScriptArgs = $args[1..($args.Count - 1)]
}

& $PythonExe $ScriptPath @ScriptArgs
