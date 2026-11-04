# Player spike (S1)

Standalone crate outside the workspace. It tests the design in the plan's "Player" section: ffmpeg child processes,
an `iced::widget::shader` surface on Iced's own device, a cpal audio clock, and seek by process restart.

Build (run from the repo root, in Git Bash):

    CARGO_TARGET_DIR=target-s1 cargo build --release --bins --manifest-path spikes/player/Cargo.toml

Generate the sync clip (a white flash and a 1 kHz beep start together every 2 s, the first at t = 1 s, so the first second is black), with `FF` set to your ffmpeg:

    "$FF" -y -f lavfi -i "color=c=black:s=1280x720:r=30:d=120,drawbox=enable='gte(t\,1)*lt(mod(t-1\,2)\,0.1)':color=white:t=fill" \
      -f lavfi -i "aevalsrc='if(gte(t\,1)*lt(mod(t-1\,2)\,0.1)\,0.5*sin(2*PI*1000*t)\,0)':s=48000:c=stereo:d=120" \
      -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 128k -shortest spikes/player/media/flash_beep.mp4

Sync by eye, with sound:

    target-s1/release/player-spike.exe spikes/player/media/flash_beep.mp4 --out spikes/player/out/sync

Options: `--w --h --fps` output size and rate, `--track K` for `-map 0:a:K`, `--mute` keeps the audio clock but outputs
silence, `--no-audio` uses the wall clock, `--start SECS`.

Auto mode, which writes `stats.json` to `--out` and exits: `--play-secs N`, `--seeks N`, `--pause-test`, `--window-test`.
Test flags: `--first-frame-delay SECS` holds back the first frame, `--theme light|dark`, `--no-window-bg` and
`--no-container-bg` turn off a black layer to test the ones left. The window test takes over the screen with fullscreen for a few seconds. `--pause-after SECS` pauses and stays paused.

`bin/track_check` compares the track ASR decodes with each `-map 0:a:K`: `track_check FFMPEG FILE [SECONDS]`.

`capture_window.ps1` saves one process's window to a PNG. Outputs stay under `out/` and `media/`, which are not committed
because captures of real recordings can contain private content.
