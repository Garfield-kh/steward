# Launch Steward and open the control panel in the browser; `steward up` gives the scenario's task
# to the robot's Ego Harness, whose conversation the console shows. The Windows counterpart of
# launch-desktop.sh.
# Usage: launch-desktop.ps1 [scenario]
#   With no scenario argument, shows a chooser (or uses STEWARD_SCENARIO).
param([string]$Scenario = '')
. (Join-Path $PSScriptRoot 'steward-env.ps1')

$logDir = if ($env:STEWARD_LOG_DIR) { $env:STEWARD_LOG_DIR } else { Join-Path $env:LOCALAPPDATA 'Steward\Logs' }
$log = Join-Path $logDir 'desktop-launcher.log'
$wcaUrl = 'http://127.0.0.1:7400'
$egoUrl = 'http://127.0.0.1:7420'
$consoleUrl = "$wcaUrl/console"
$scenarios = [ordered]@{
    'puzzle-dog' = '拼图·狗'; 'puzzle-cat' = '拼图·猫'; 'puzzle-greatwall' = '拼图·长城'
    'phase0-chores' = '家务整理'; 'phase0-shuttle' = '循环搬运'; 'phase0-kitchen' = '杯子放到桌上'
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log([string]$Text) {
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Text" -Encoding UTF8
}

function Stop-Launch([string]$Text) {
    Write-Log "FAIL: $Text"
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show($Text, 'Steward', 'OK', 'Warning') | Out-Null
    exit 1
}

function Select-Scenario {
    if ($Scenario) { return $Scenario }
    if ($env:STEWARD_SCENARIO) { return $env:STEWARD_SCENARIO }
    $rows = foreach ($id in $scenarios.Keys) { [pscustomobject]@{ 场景 = $scenarios[$id]; id = $id } }
    if (Get-Command Out-GridView -ErrorAction SilentlyContinue) {
        $picked = $rows | Out-GridView -Title '选择要启动的场景' -OutputMode Single
        if (-not $picked) { exit 0 }
        return $picked.id
    }
    for ($i = 0; $i -lt $rows.Count; $i++) { Write-Host "$($i + 1). $($rows[$i].场景) ($($rows[$i].id))" }
    $answer = Read-Host '选择要启动的场景（序号，回车取消）'
    if (-not $answer) { exit 0 }
    return $rows[[int]$answer - 1].id
}

function Wait-Http([string]$Url, [string]$Label, [int]$Seconds = 90) {
    for ($i = 0; $i -lt $Seconds; $i++) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 1 | Out-Null
            return
        } catch {
            # Connection refused or timed out (WebException in Windows PowerShell, HttpRequestException
            # in pwsh): the service is not up yet, retried until the deadline below.
            Start-Sleep -Seconds 1
        }
    }
    Stop-Launch "等待 $Label 超时（$Seconds 秒）：$Url"
}

function Stop-Existing {
    # Ego Harness first, then steward up and its Ego Runtime children.
    $patterns = @('--profile\s+ego\b', 'steward(\.exe)?"?\s+up\b', 'steward\.cli\b.*\s(cobot-)?ego\b')
    Get-CimInstance Win32_Process | Where-Object {
        $line = $_.CommandLine
        $_.ProcessId -ne $PID -and $line -and ($patterns | Where-Object { $line -match $_ })
    } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    # Free the fixed ports if a stale process still holds them.
    foreach ($port in 7400, 7410, 7420) {
        Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    }
    Start-Sleep -Seconds 1
}

Write-Log "----- cwd=$StewardRoot"
if (-not (Test-Path -LiteralPath $StewardExe)) { Stop-Launch "找不到 steward 可执行文件：$StewardExe。请先在 steward\ 运行 uv sync。" }
if (-not (Test-Path -LiteralPath (Join-Path $StewardRoot '.env'))) { Stop-Launch "找不到 $StewardRoot\.env（需要 DEEPSEEK_API_KEY）。" }
if (-not (Test-Path -LiteralPath $DshLauncher)) { Stop-Launch '找不到 dsh。请先运行 steward\scripts\setup-dsh-profiles.ps1。' }

$picked = Select-Scenario
Write-Log "scenario=$picked"
Stop-Existing

Import-StewardEnv
$steward = Start-Process -FilePath $StewardExe -WorkingDirectory $StewardRoot -WindowStyle Hidden -PassThru `
    -ArgumentList @('up', '--scenario', $picked, '--speed', 'max', '--freeze-idle') `
    -RedirectStandardOutput (Join-Path $logDir 'steward.out.log') -RedirectStandardError (Join-Path $logDir 'steward.err.log')
Write-Log "steward_pid=$($steward.Id)"
Wait-Http "$wcaUrl/console" '控制台'
Wait-Http "$egoUrl/ego/state" 'Ego API'

Start-Process $consoleUrl
Write-Log "opened $consoleUrl"
