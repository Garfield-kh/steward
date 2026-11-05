# Windows counterpart of install-desktop-app.sh: put a Steward shortcut on the desktop that runs
# launch-desktop.ps1 (Steward + Ego Harness, then the control panel in the browser).
. (Join-Path $PSScriptRoot 'steward-env.ps1')
$icon = Join-Path $StewardRoot 'var\desktop\Steward.ico'
$png = Join-Path $PSScriptRoot 'desktop\AppIcon-1024.png'
if (-not (Test-Path -LiteralPath $StewardPython)) { throw "missing $StewardPython; run uv sync in steward\ first" }
New-Item -ItemType Directory -Force -Path (Split-Path $icon -Parent) | Out-Null
& $StewardPython -c 'import sys; from PIL import Image; Image.open(sys.argv[1]).save(sys.argv[2], sizes=[(256, 256), (64, 64), (48, 48), (32, 32), (16, 16)])' $png $icon
Assert-Exit 'writing the shortcut icon'

$target = if ($env:STEWARD_DESKTOP_APP) { $env:STEWARD_DESKTOP_APP } else { Join-Path ([Environment]::GetFolderPath('Desktop')) 'Steward.lnk' }
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($target)
$shortcut.TargetPath = (Get-Process -Id $PID).Path  # powershell.exe or pwsh.exe, whichever runs this
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$(Join-Path $PSScriptRoot 'launch-desktop.ps1')`""
$shortcut.WorkingDirectory = $StewardRoot
$shortcut.IconLocation = $icon
$shortcut.Description = 'Steward: world simulation, Ego Harness and control panel'
$shortcut.Save()
Write-Output "installed $target"
