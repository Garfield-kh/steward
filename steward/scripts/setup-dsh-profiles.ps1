# Windows counterpart of setup-dsh-profiles.sh: build the Steward DSH plugins and create two
# isolated DeepSeek Harness homes:
#   var\dsh\robot-1  profile "ego"   (Ego Harness, from the shipped headless template by default)
#   var\dsh\world    profile "world" (World Harness, from the shipped web template)
# Everything goes through upstream launcher commands; no upstream file is modified.
. (Join-Path $PSScriptRoot 'steward-env.ps1')
$robotTemplate = if ($env:ROBOT_TEMPLATE) { $env:ROBOT_TEMPLATE } else { 'headless' }
$previousDshHome = $env:DSH_HOME

pnpm --dir (Join-Path $StewardRoot 'dsh') install
Assert-Exit 'pnpm install'
pnpm --dir (Join-Path $StewardRoot 'dsh') run build
Assert-Exit 'Steward DSH plugin build'
Assert-DshLauncher

function Add-StewardPlugin([string]$HomeDir, [string]$ProfileName, [string]$Bundle) {
    $env:DSH_HOME = $HomeDir
    & $DshLauncher plugin --profile $ProfileName add (Join-Path $StewardRoot "dsh\packages\$Bundle")
    Assert-Exit "adding $Bundle to $ProfileName"
}

function Initialize-Profile([string]$HomeDir, [string]$ProfileName, [string]$Template, [string]$Bundle) {
    New-Item -ItemType Directory -Force -Path $HomeDir | Out-Null
    $env:DSH_HOME = $HomeDir
    if (-not (Test-Path -LiteralPath (Join-Path $HomeDir "profiles\$ProfileName\package.json"))) {
        & $DshLauncher --profile $ProfileName --from-default-profile $Template --dump-default-config | Out-Null
        Assert-Exit "creating profile $ProfileName"
    }
    Add-StewardPlugin $HomeDir $ProfileName $Bundle
    Write-Output "profile $ProfileName ready in $HomeDir"
}

try {
    $robot = Join-Path $DshVar 'robot-1'
    $world = Join-Path $DshVar 'world'
    Initialize-Profile $robot 'ego' $robotTemplate 'dsh-bundle-ego'
    Add-StewardPlugin $robot 'ego' 'dsh-bundle-rgbd-video-gen'
    Add-StewardPlugin $robot 'ego' 'dsh-bundle-novelanything'
    Initialize-Profile $world 'world' 'web' 'dsh-bundle-world'
    Initialize-Profile $world 'world-task' 'headless' 'dsh-bundle-world'
} finally {
    $env:DSH_HOME = $previousDshHome
}
