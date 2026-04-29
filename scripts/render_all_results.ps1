# render_all_results.ps1 — render one mp4 per saved run under results/.
#
# Output: docs/figures/results_overview/<run_name>.mp4
# Skips runs that lack reference.npy (e.g. pretrain_symmetry) and overnight
# sweeps. Each run is rendered independently so one crash doesn't stop the
# rest; failures are logged to stderr and to a summary at the end.
#
# Usage (from project root):
#   python -V                                # ensure .venv is active
#   .\scripts\render_all_results.ps1
#
# Tweak --eps / --steps below for shorter/longer clips.

$ErrorActionPreference = "Continue"

$xml      = "assets\mjcf\walker2d_custom.xml"
$outDir   = "docs\figures\results_overview"
$eps      = 1
$steps    = 1500

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$runs = @(
  @{ dir = "results\restart_b1_dm";                                           ckpt = "2000000" },
  @{ dir = "results\restart_b1_dm_bc";                                        ckpt = "2000000" },
  @{ dir = "results\restart_b2_k30";                                          ckpt = "final"   },
  @{ dir = "results\restart_b2_xvel";                                         ckpt = "final"   },
  @{ dir = "results\walker2d_phase_cycle_s1scaled_sum_20260422-175117";       ckpt = "final"   },
  @{ dir = "results\walker2d_phase_cycle_s1scaled_sum_20260423-213031";       ckpt = "final"   },
  @{ dir = "results\walker2d_phase_cycle_sum_20260409-211537";                ckpt = "final"   },
  @{ dir = "results\walker2d_phase_full_sum_20260410-105306";                 ckpt = "final"   },
  @{ dir = "results\walker2d_phase_full_sum_20260410-124935";                 ckpt = "final"   }
)

$ok   = @()
$fail = @()

foreach ($r in $runs) {
  $name = Split-Path $r.dir -Leaf
  $mp4  = Join-Path $outDir "$name.mp4"
  Write-Host "=== [$name] -> $mp4 ===" -ForegroundColor Cyan

  python src\walker2d\render_phase.py `
      --xml   $xml `
      --eps   $eps `
      --steps $steps `
      --mp4   $mp4 `
      "$($r.dir):$($r.ckpt)"

  if ($LASTEXITCODE -eq 0) { $ok   += $name }
  else                     { $fail += $name }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Yellow
Write-Host ("Rendered: {0}" -f ($ok.Count))
$ok   | ForEach-Object { Write-Host "  ok    $_" -ForegroundColor Green }
Write-Host ("Failed:   {0}" -f ($fail.Count))
$fail | ForEach-Object { Write-Host "  FAIL  $_" -ForegroundColor Red }
Write-Host ""
Write-Host "Output dir: $outDir"
