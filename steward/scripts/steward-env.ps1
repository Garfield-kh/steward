# Shared by the Windows PowerShell scripts (dot-source it): Steward's paths, its DSH launcher and
# steward/.env, matching what the .sh scripts compute on macOS and Linux.
$ErrorActionPreference = 'Stop'
$StewardRoot = Split-Path $PSScriptRoot -Parent
# $IsWindows does not exist in Windows PowerShell 5.1; pwsh on macOS and Linux runs these scripts too.
$OnWindows = $env:OS -eq 'Windows_NT'
# npm generates dsh.CMD on Windows; the extensionless dsh next to it is the POSIX sh launcher.
$DshLauncher = Join-Path $StewardRoot $(if ($OnWindows) { 'dsh\node_modules\.bin\dsh.CMD' } else { 'dsh/node_modules/.bin/dsh' })
$DshVar = if ($env:STEWARD_DSH_VAR) { $env:STEWARD_DSH_VAR } else { Join-Path $StewardRoot 'var/dsh' }
$StewardPython = Join-Path $StewardRoot $(if ($OnWindows) { '.venv\Scripts\python.exe' } else { '.venv/bin/python' })
$StewardExe = Join-Path $StewardRoot $(if ($OnWindows) { '.venv\Scripts\steward.exe' } else { '.venv/bin/steward' })

# dsh and Steward speak UTF-8 whatever the console code page (936 on Chinese Windows).
$Utf8 = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = $Utf8
$OutputEncoding = $Utf8
$env:PYTHONUTF8 = '1'

function Import-StewardEnv {
    <# Set the KEY=VALUE lines of steward/.env as process environment variables; no file sets none. #>
    $file = Join-Path $StewardRoot '.env'
    if (-not (Test-Path -LiteralPath $file)) { return }
    foreach ($line in Get-Content -LiteralPath $file -Encoding UTF8) {
        $line = $line.Trim()
        if (-not $line -or $line.StartsWith('#') -or -not $line.Contains('=')) { continue }
        $key, $value = $line.Split('=', 2)
        $key = $key.Trim()
        if ($key.StartsWith('export ')) { $key = $key.Substring(7).Trim() }
        Set-Item -LiteralPath "Env:$key" -Value $value.Trim().Trim([char[]]@("'", '"'))
    }
}

function Assert-Exit([string]$What) {
    <# Stop the script when the native command just run failed. #>
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

function Assert-DshLauncher {
    <# Fail with the setup step when the Steward DSH workspace is not installed. #>
    if (-not (Test-Path -LiteralPath $DshLauncher)) {
        throw "dsh launcher not found: $DshLauncher. Run scripts\setup-dsh-profiles.ps1 first."
    }
}
