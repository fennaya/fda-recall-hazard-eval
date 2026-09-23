# Launch a command as a genuinely independent Windows process.
#
# Why this exists: `nohup cmd &` inside Git Bash disowns the command from
# bash's own job control, but the resulting process is still a descendant of
# the MSYS/mintty process tree Git Bash is running under. When that tree gets
# torn down (a Claude Code session ending or restarting has done this
# repeatedly during this project), the "detached" eval process died with it,
# even though nothing in the eval itself failed. PowerShell's Start-Process
# creates a process with no such parent relationship to the calling shell, so
# it survives the launching session ending -- it only stops if the machine
# itself goes down, or something explicitly stops it.
#
# Usage:
#   powershell -File scripts/launch_detached.ps1 `
#       -Exe "C:\path\to\uv.exe" -Args "run python -u -m fda_hazard.evaluate --workers 1" `
#       -WorkDir "C:\Users\emssa\Downloads\projet 1" `
#       -LogPath "C:\Users\emssa\Downloads\projet 1\logs\eval_full.log" `
#       -PidPath "C:\Users\emssa\Downloads\projet 1\logs\eval_full.pid"
#
# Prints the new process's PID (also written to -PidPath) so it can be
# checked on or stopped later: `Stop-Process -Id <pid>`.

param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$Args,
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [string]$PidPath = $null
)

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null

$proc = Start-Process -FilePath $Exe -ArgumentList $Args -WorkingDirectory $WorkDir `
    -WindowStyle Hidden -RedirectStandardOutput $LogPath `
    -RedirectStandardError "$LogPath.err" -PassThru

if ($PidPath) {
    $proc.Id | Out-File -Encoding ascii $PidPath
}

Write-Output "launched pid $($proc.Id), logging to $LogPath"
Write-Output "parent shell's own pid for comparison: $PID"
