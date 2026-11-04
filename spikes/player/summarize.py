"""Print the gate numbers from a stats.json: python summarize.py out/RUN/stats.json"""

import json
import sys


def r(x):
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()}


d = json.load(open(sys.argv[1]))
m = d["meta"]
print(m["video"], "|", m["audio"], "| startup ms", round(m["start_to_first_upload_ms"]), "| latency ms", m["audio_latency_ms_last"])
print(d["surface_format"], d["totals"], d["device_lost_log"])
for p in d["phases"]:
    s, c = p["series"], p["counters"]
    print("==", p["name"], "wall", round(p["wall_secs"], 1), "clock", round(p["clock_secs"], 1), "drop%", round(p["drop_pct_read_vs_uploaded"], 3), "view/s", round(p["view_per_sec"], 2), "up/s", round(p["uploads_per_sec"], 2))
    print("  ", {k: v for k, v in c.items() if v})
    for k in ["clock_error_abs_ms", "clock_error_signed_ms", "upload_gap_ms", "upload_cost_ms", "app_rss_mb"]:
        print("  ", k, r(s[k]))
    for k, v in s["cpu_cores"].items():
        print("  ", k, r(v))
    for st in (p["extra"] or {}).get("steps", []):
        print("    ", st)
    if (p["extra"] or {}).get("closed_early"):
        print("   CLOSED EARLY")
print("seek to first upload", r(d["seek_summary"]["to_first_upload_ms"]))
print("seek to first read", r(d["seek_summary"]["to_first_read_ms"]))
print([(round(s["to_first_upload_ms"]), round(s.get("system_cores_busy") or -1)) for s in d["seeks"]], "(ms, busy cores)")
print("ffmpeg spawned", len(d["ffmpeg_pids"]))
