# World Harness launcher, the Windows counterpart of run-world.sh. Requires `steward up` or
# `steward world`; loads steward\.env.
#   scripts\run-world.ps1                 browser UI on port 3091 (profile "world")
#   scripts\run-world.ps1 "<task>"        one-shot task (profile "world-task", headless)
. (Join-Path $PSScriptRoot 'steward-env.ps1')
Assert-DshLauncher
Import-StewardEnv
$env:DSH_HOME = Join-Path $DshVar 'world'
Set-Location -LiteralPath $StewardRoot
if ($args.Count -gt 0 -and -not "$($args[0])".StartsWith('-')) {
    & $DshLauncher --profile world-task @args
} elseif ($args.Count -gt 0) {
    & $DshLauncher --profile world @args
} else {
    & $DshLauncher --profile world --port 3091
}
exit $LASTEXITCODE
