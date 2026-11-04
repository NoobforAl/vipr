# Starts the player, then samples its window through PrintWindow as fast as possible, from the first moment the
# window exists. Prints one line per sample: time since launch, and the mean colour of the client centre and of a corner.
# Usage: pwsh -File capture_startup.ps1 -Exe player-spike.exe -Out dir -Seconds 8 -PlayerArgs '"clip.mp4" --theme light'
param(
    [Parameter(Mandatory)][string]$Exe,
    [Parameter(Mandatory)][string]$Out,
    [double]$Seconds = 8,
    [string]$PlayerArgs = ""
)

Add-Type -AssemblyName System.Drawing
Add-Type -ReferencedAssemblies System.Drawing, System.Drawing.Primitives, System.Drawing.Common @"
using System;
using System.Drawing;
using System.Runtime.InteropServices;
public static class C {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int L, T, R, B; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint flags);
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc p, IntPtr l);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  public static IntPtr Find(uint target) {
    IntPtr found = IntPtr.Zero;
    EnumWindows((h, l) => {
      uint pid; GetWindowThreadProcessId(h, out pid);
      RECT r;
      if (pid == target && IsWindowVisible(h) && GetWindowRect(h, out r) && r.R - r.L >= 64 && r.B - r.T >= 64) { found = h; return false; }
      return true;
    }, IntPtr.Zero);
    return found;
  }
  public static string Mean(Bitmap b, int x0, int y0, int w, int h) {
    long r = 0, g = 0, bl = 0, n = 0;
    for (int y = y0; y < y0 + h; y += 4) for (int x = x0; x < x0 + w; x += 4) {
      var c = b.GetPixel(x, y); r += c.R; g += c.G; bl += c.B; n++;
    }
    return (r / n) + "," + (g / n) + "," + (bl / n);
  }
}
"@

New-Item -ItemType Directory -Force $Out | Out-Null
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $Exe -ArgumentList $PlayerArgs -PassThru
$i = 0
$seen = $false
while ($sw.Elapsed.TotalSeconds -lt $Seconds -and -not $p.HasExited) {
    $h = [C]::Find([uint32]$p.Id)
    if ($h -eq [IntPtr]::Zero) { continue }
    if (-not $seen) { $seen = $true; "{0,6:N0} ms window visible" -f $sw.Elapsed.TotalMilliseconds }
    $r = New-Object C+RECT
    [void][C]::GetWindowRect($h, [ref]$r)
    $w = $r.R - $r.L; $ht = $r.B - $r.T
    if ($w -lt 64 -or $ht -lt 64) { continue }
    $bmp = New-Object System.Drawing.Bitmap $w, $ht
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    # through DWM, so windows above this one are not captured
    $dc = $g.GetHdc()
    [void][C]::PrintWindow($h, $dc, 2)
    $g.ReleaseHdc($dc)
    $g.Dispose()
    $t = $sw.Elapsed.TotalMilliseconds
    $centre = [C]::Mean($bmp, [int]($w / 2 - 60), [int]($ht / 2 - 60), 120, 120)
    $corner = [C]::Mean($bmp, 24, 60, 60, 60)
    "{0,6:N0} ms centre {1} corner {2}" -f $t, $centre, $corner
    if ($i -lt 6 -or $i % 10 -eq 0) { $bmp.Save((Join-Path $Out ("f{0:D3}.png" -f $i)), [System.Drawing.Imaging.ImageFormat]::Png) }
    $bmp.Dispose()
    $i++
}
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
"pid $($p.Id) samples $i"
