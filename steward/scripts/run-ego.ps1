# Run the robot's Ego Harness on one task: scripts\run-ego.ps1 "把杯子放到桌上"
# Windows counterpart of run-ego.sh. Loads steward\.env (DEEPSEEK_API_KEY) and starts dsh in the
# robot's own empty workspace, so the agent's working directory holds no simulator data.
# Requires `steward up` (or `steward ego`).
. (Join-Path $PSScriptRoot 'steward-env.ps1')
Assert-DshLauncher
Import-StewardEnv
$homeDir = Join-Path $DshVar 'robot-1'
$workspace = Join-Path $homeDir 'workspace'
New-Item -ItemType Directory -Force -Path $workspace | Out-Null
$env:DSH_HOME = $homeDir
Set-Location -LiteralPath $workspace
& $DshLauncher --profile ego @args
exit $LASTEXITCODE
