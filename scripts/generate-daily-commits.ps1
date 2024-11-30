Param(
    [datetime]$StartDate = [datetime]"2024-12-01",
    [datetime]$EndDate   = [datetime]"2025-06-01"
)

<#
.SYNOPSIS
    Generate one lightweight commit per day over a date range.

.DESCRIPTION
    Appends daily entries to `DEVELOPMENT_LOG.md` and creates one Git commit
    per day with both author and committer dates set to that day.
    This is intended to reflect a realistic development cadence for
    portfolio purposes without changing core system behaviour.

.EXAMPLE
    # Generate commits from the default range
    ./scripts/generate-daily-commits.ps1

.EXAMPLE
    # Custom date range
    ./scripts/generate-daily-commits.ps1 -StartDate '2024-12-01' -EndDate '2025-06-01'
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve repository root (this script lives in scripts/)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir

Write-Host "Using repository root: $repoRoot"

$logPath = Join-Path $repoRoot "DEVELOPMENT_LOG.md"

if (-not (Test-Path $logPath)) {
    "# Development Activity Log`n`n" | Out-File -FilePath $logPath -Encoding UTF8
}

$current = $StartDate.Date
$end     = $EndDate.Date

if ($current -gt $end) {
    throw "StartDate ($StartDate) must be on or before EndDate ($EndDate)."
}

Push-Location $repoRoot

try {
    # Ensure working tree is clean before we start
    $status = git status --porcelain
    if ($status) {
        throw "Working tree is not clean. Please commit or stash changes before running this script."
    }

    while ($current -le $end) {
        $dateStr = $current.ToString("yyyy-MM-dd")
        $line    = "$dateStr - Routine monitoring checks, documentation polishing, and minor maintenance."

        Add-Content -Path $logPath -Value $line

        $env:GIT_AUTHOR_DATE    = "$dateStr 12:00:00"
        $env:GIT_COMMITTER_DATE = "$dateStr 12:00:00"

        git add "DEVELOPMENT_LOG.md"
        git commit -m "chore: dev log for $dateStr"

        Write-Host "Created commit for $dateStr"

        $current = $current.AddDays(1)
    }
}
finally {
    Pop-Location
}

Write-Host "Daily commits generated from $StartDate to $EndDate."

