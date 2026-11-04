# Gate 5: no ffmpeg left behind. Starts the spike, ends only that PID, then checks only the child PIDs it recorded.
# Usage: pwsh -File kill_test.ps1 -Exe player-spike.exe -Media file.mp4 -OutRoot out
param([Parameter(Mandatory)][string]$Exe, [Parameter(Mandatory)][string]$Media, [Parameter(Mandatory)][string]$OutRoot)

function Test-Case([string]$Name, [string]$Extra, [bool]$Hard, [int]$WaitSeconds) {
    $out = Join-Path $OutRoot "kill_$Name"
    Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
    $p = Start-Process -FilePath $Exe -PassThru -ArgumentList "`"$Media`" --mute --start 600 $Extra --out `"$out`""
    Start-Sleep -Seconds $WaitSeconds
    $children = Get-Content (Join-Path $out "pids.txt") | ForEach-Object { [int]($_ -split ' ')[1] }
    $aliveBefore = @($children | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count
    if ($Hard) {
        Stop-Process -Id $p.Id -Force
    } else {
        [void]$p.CloseMainWindow()
    }
    $gone = [Diagnostics.Stopwatch]::StartNew()
    $left = @($children)
    while ($gone.Elapsed.TotalSeconds -lt 15) {
        $left = @($children | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
        if ($left.Count -eq 0) { break }
        Start-Sleep -Milliseconds 50
    }
    $ms = [int]$gone.Elapsed.TotalMilliseconds
    "$Name : children recorded $($children.Count), alive before $aliveBefore, alive after: $($left.Count) ($ms ms), app exited: $($p.HasExited)"
    # clean up only what this script started
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    foreach ($id in $left) {
        if ($children -contains $id) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
    }
}

Test-Case "hard_playing" "" $true 10
Test-Case "hard_paused" "--pause-after 7" $true 12
Test-Case "hard_during_startup" "" $true 1
Test-Case "close_window" "" $false 10
