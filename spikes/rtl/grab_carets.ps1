# Runs the spike's --carets script and saves what the real window shows at each step.
# Takes two images per step, 550 ms apart, because the caret blinks with a 1 s period.
# Usage: pwsh spikes/rtl/grab_carets.ps1 <path to rtl-spike.exe> <shots dir>
param([string]$Exe, [string]$Dir)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System; using System.Collections.Generic; using System.Runtime.InteropServices;
public class W {
  public delegate bool Cb(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr dc, uint f);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out R r);
  [DllImport("user32.dll")] public static extern bool EnumWindows(Cb cb, IntPtr l);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  public struct R { public int L, T, Rt, B; }
  public static IntPtr Main(uint target) {
    IntPtr best = IntPtr.Zero; int area = 0;
    EnumWindows((h, l) => {
      uint pid; GetWindowThreadProcessId(h, out pid);
      R r; GetWindowRect(h, out r);
      int a = (r.Rt - r.L) * (r.B - r.T);
      if (pid == target && IsWindowVisible(h) && a > area) { best = h; area = a; }
      return true;
    }, IntPtr.Zero);
    return best;
  }
}
"@
function Grab([uint32]$procId, [string]$out) {
    $h = [W]::Main($procId)
    $r = New-Object W+R
    [void][W]::GetWindowRect($h, [ref]$r)
    $bmp = New-Object System.Drawing.Bitmap ($r.Rt - $r.L), ($r.B - $r.T)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $dc = $g.GetHdc()
    [void][W]::PrintWindow($h, $dc, 2)
    $g.ReleaseHdc($dc)
    $bmp.Save($out)
    $g.Dispose(); $bmp.Dispose()
}
New-Item -ItemType Directory -Force $Dir | Out-Null
$request = Join-Path $Dir 'request.txt'
Remove-Item $request -ErrorAction Ignore
$p = Start-Process -FilePath $Exe -ArgumentList '--carets', "`"$Dir`"" -PassThru
$deadline = (Get-Date).AddMinutes(4)
while (-not $p.HasExited -and (Get-Date) -lt $deadline) {
    if (Test-Path $request) {
        $name = (Get-Content $request -Raw).Trim()
        Start-Sleep -Milliseconds 150
        Grab $p.Id (Join-Path $Dir "${name}_a.png")
        Start-Sleep -Milliseconds 550
        Grab $p.Id (Join-Path $Dir "${name}_b.png")
        Remove-Item $request
    }
    Start-Sleep -Milliseconds 50
}
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Confirm:$false; "timed out" } else { "done" }
