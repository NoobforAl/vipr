# Captures one process's top-level window to a PNG through DWM (PrintWindow), so other windows are not included.
# Usage: pwsh -File capture_window.ps1 -TargetPid 1234 -Out shot.png
param([Parameter(Mandatory)][int]$TargetPid, [Parameter(Mandatory)][string]$Out)

Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class W {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
}
"@

$h = (Get-Process -Id $TargetPid).MainWindowHandle
if ($h -eq [IntPtr]::Zero) { throw "process $TargetPid has no main window" }
$r = New-Object W+RECT
[void][W]::GetWindowRect($h, [ref]$r)
$bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$dc = $g.GetHdc()
$ok = [W]::PrintWindow($h, $dc, 2) # PW_RENDERFULLCONTENT
$g.ReleaseHdc($dc)
$g.Dispose()
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$bmp.Dispose()
"PrintWindow=$ok size=$($r.R - $r.L)x$($r.B - $r.T) out=$Out"
