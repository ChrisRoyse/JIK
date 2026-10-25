param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Result {
    param([object]$Result, [int]$ExitCode)
    if ($Json) { $Result | ConvertTo-Json -Depth 8 }
    else {
        Write-Output "mnlm_git_guards_ok=$($Result.ok)"
        Write-Output "mnlm_git_hooks_path=$($Result.hooks_path)"
        Write-Output "mnlm_git_origin=$($Result.origin)"
    }
    exit $ExitCode
}

try {
    $repo = [IO.Path]::GetFullPath($RepoRoot).TrimEnd("\")
    $actualRootOutput = @(& git -C $repo rev-parse --show-toplevel 2>&1)
    if ($LASTEXITCODE -ne 0 -or $actualRootOutput.Count -ne 1) {
        throw "MNLM_REPO_ROOT_INVALID: $repo is not a git repository"
    }
    $actualRoot = "$($actualRootOutput[0])"
    $actualRootFull = [IO.Path]::GetFullPath("$actualRoot").TrimEnd("\")
    if ($actualRootFull -ine $repo) { throw "MNLM_REPO_ROOT_INVALID: expected=$repo actual=$actualRootFull" }

    $remoteNames = @(& git -C $repo remote 2>&1 | ForEach-Object { "$_" })
    if ($LASTEXITCODE -ne 0 -or $remoteNames.Count -ne 1 -or $remoteNames[0] -cne "origin") {
        throw "MNLM_REMOTE_SET_INVALID: the only permitted remote is origin -> ChrisRoyse/MNLM"
    }
    $urls = @(& git -C $repo remote get-url --all origin 2>&1) + @(& git -C $repo remote get-url --push --all origin 2>&1)
    $bad = @($urls | Where-Object { "$_" -notmatch '(?i)^(?:https://github\.com/ChrisRoyse/(?:MNLM|JIK)(?:\.git)?|git@github\.com:ChrisRoyse/(?:MNLM|JIK)(?:\.git)?|ssh://git@github\.com/ChrisRoyse/(?:MNLM|JIK)(?:\.git)?)$' })
    if ($urls.Count -eq 0 -or $bad.Count -ne 0) { throw "MNLM_WRONG_ORIGIN: origin must be ChrisRoyse/MNLM (got: $($urls -join ', '))" }

    foreach ($h in @("pre-commit", "pre-push")) {
        $p = Join-Path $repo ".githooks\$h"
        if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "MNLM_GIT_HOOK_MISSING: $p" }
    }

    & git -C $repo config core.hooksPath .githooks
    if ($LASTEXITCODE -ne 0) { throw "MNLM_GIT_CONFIG_FAILED: core.hooksPath" }
    $hooksPath = (& git -C $repo config --get core.hooksPath)
    if ($hooksPath -ne ".githooks") { throw "MNLM_GIT_HOOKS_PATH_READBACK_FAILED: $hooksPath" }

    Write-Result @{ ok = $true; hooks_path = $hooksPath; origin = $urls[0] } 0
}
catch {
    Write-Result @{ ok = $false; code = ("$_" -split ':')[0]; message = "$_" } 1
}
