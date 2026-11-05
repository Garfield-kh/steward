# Windows counterpart of check.sh, the Steward compatibility and regression checks (steward.md §7 P8):
#   1. Python: scene model, assets, WSim, Ego Runtime, Observer and the end-to-end mission.
#   2. DSH plugins: TypeScript build and unit tests.
#   3. Composed-configuration snapshots of the ego and world profiles.
# Set UPDATE_SNAPSHOTS=1 to accept the current composed configuration.
. (Join-Path $PSScriptRoot 'steward-env.ps1')
$snapshots = Join-Path $StewardRoot 'compat\snapshots'
$previousDshHome = $env:DSH_HOME

Write-Output '== python'
Push-Location -LiteralPath $StewardRoot
try { & $StewardPython -m pytest -q; Assert-Exit 'pytest' } finally { Pop-Location }

Write-Output '== dsh plugins'
Push-Location -LiteralPath (Join-Path $StewardRoot 'dsh')
try {
    pnpm run build; Assert-Exit 'Steward DSH plugin build'
    node --test 'test/*.test.mjs'; Assert-Exit 'Steward DSH plugin tests'
} finally { Pop-Location }

Write-Output '== composed configuration'
$pairs = @(@('robot-1', 'ego'), @('world', 'world'), @('world', 'world-task'))
if ($pairs | Where-Object { -not (Test-Path -LiteralPath (Join-Path $DshVar "$($_[0])\profiles\$($_[1])")) }) {
    & (Join-Path $PSScriptRoot 'setup-dsh-profiles.ps1')
}
New-Item -ItemType Directory -Force -Path $snapshots | Out-Null
$status = 0
try {
    foreach ($pair in $pairs) {
        $profileName = $pair[1]
        $env:DSH_HOME = Join-Path $DshVar $pair[0]
        $lines = & $DshLauncher --profile $profileName --dump-config
        Assert-Exit "dumping profile $profileName"
        # Snapshots are LF text, as check.sh writes them.
        $current = ($lines -join "`n") + "`n"
        $file = Join-Path $snapshots "$profileName.yml"
        if ($env:UPDATE_SNAPSHOTS -eq '1' -or -not (Test-Path -LiteralPath $file)) {
            [IO.File]::WriteAllText($file, $current, $Utf8)
            Write-Output "snapshot $profileName.yml written"
        } elseif (([IO.File]::ReadAllText($file, $Utf8) -replace "`r`n", "`n") -ne $current) {
            $temp = [IO.Path]::GetTempFileName()
            [IO.File]::WriteAllText($temp, $current, $Utf8)
            git --no-pager diff --no-index -- $file $temp
            Remove-Item -LiteralPath $temp
            Write-Error "composed configuration of profile $profileName changed; review, then rerun with UPDATE_SNAPSHOTS=1" -ErrorAction Continue
            $status = 1
        } else {
            Write-Output "profile $profileName unchanged"
        }
    }
} finally {
    $env:DSH_HOME = $previousDshHome
}
exit $status
