$ErrorActionPreference = 'Stop'
$stewardRoot = Split-Path $PSScriptRoot -Parent
$previousDshHome = $env:DSH_HOME
Push-Location $stewardRoot
try {
    pnpm --dir dsh install
    if ($LASTEXITCODE -ne 0) { throw 'Steward DSH dependency installation failed.' }
    pnpm --dir dsh run build
    if ($LASTEXITCODE -ne 0) { throw 'Steward DSH plugin build failed.' }
    $dshLauncher = Join-Path $stewardRoot 'dsh/node_modules/.bin/dsh.cmd'
    $novelBundle = Join-Path $stewardRoot 'dsh/packages/dsh-bundle-novelanything'
    $novelPackage = '@steward/dsh-bundle-novelanything'
    $env:DSH_HOME = Join-Path $stewardRoot 'var/dsh/world'
    foreach ($worldProfile in @('world', 'world-task')) {
        $worldManifest = Join-Path $env:DSH_HOME "profiles/$worldProfile/package.json"
        if (Test-Path -LiteralPath $worldManifest) {
            $worldPackages = Get-Content -LiteralPath $worldManifest -Raw | ConvertFrom-Json
            if ($worldPackages.dependencies.$novelPackage) {
                & $dshLauncher plugin --profile $worldProfile remove $novelPackage
                if ($LASTEXITCODE -ne 0) { throw "Unable to remove NovelAnything from $worldProfile." }
            }
        }
    }
    $env:DSH_HOME = Join-Path $stewardRoot 'var/dsh/robot-1'
    New-Item -ItemType Directory -Force -Path (Join-Path $env:DSH_HOME 'workspace') | Out-Null
    $egoManifest = Join-Path $env:DSH_HOME 'profiles/ego/package.json'
    if (-not (Test-Path -LiteralPath $egoManifest)) {
        & $dshLauncher --profile ego --from-default-profile headless --dump-default-config | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Unable to initialize the ego profile.' }
    }
    $egoPackages = Get-Content -LiteralPath $egoManifest -Raw | ConvertFrom-Json
    if (-not $egoPackages.dependencies.'@steward/dsh-bundle-ego') {
        & $dshLauncher plugin --profile ego add (Join-Path $stewardRoot 'dsh/packages/dsh-bundle-ego')
        if ($LASTEXITCODE -ne 0) { throw 'Unable to install Ego tools.' }
    }
    & $dshLauncher plugin --profile ego add $novelBundle
    if ($LASTEXITCODE -ne 0) { throw 'Unable to add NovelAnything to ego.' }
    Write-Output 'NovelAnything tools and skill are installed only in robot-1 ego. Inputs resolve inside the robot workspace.'
} finally {
    $env:DSH_HOME = $previousDshHome
    Pop-Location
}
