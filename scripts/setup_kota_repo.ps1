# Restores the Kota repository at the audited commit into audit/repo from the bundled copy.
# Falls back to GitHub if the bundle is missing. No network is needed when the bundle is present.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Commit = "f76754b5bcff1a1d7a50c2f1073a4b6d44bd985c"
if (-not (Test-Path "audit/repo/.git")) {
  if (Test-Path "third_party/kota_repo_f76754b.bundle") {
    git clone -q third_party/kota_repo_f76754b.bundle audit/repo
  } else {
    git clone -q https://github.com/Tkota10/Senior-Thesis-Multi-LLM-Resolution audit/repo
  }
}
git -C audit/repo checkout -q $Commit
$Head = (git -C audit/repo rev-parse HEAD).Trim()
if ($Head -ne $Commit) { throw "audit/repo is at $Head, expected $Commit" }
Write-Output "audit/repo is at $Commit"
