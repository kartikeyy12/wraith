# Data collection protocol v2 (Phase 1: zone localization)

Changes from v1, and why: sessions are now tagged moving or still (a motionless person is nearly
invisible to variance-based features and is the case where fingerprints drift most over time),
every collection day starts with an empty-room baseline (lets us remove day-to-day drift), and
evaluation holds out whole days, not just sessions. See docs/evaluation_plan.md.

## Labels and activities
| label | activity | meaning |
|---|---|---|
| empty | baseline | empty room, 180 s, FIRST thing each day. Only used to calibrate that day, never train/test. |
| empty | empty | empty room, 600 s, a normal train/test session |
| zoneA, zoneB, ... | moving | one person moving naturally inside the zone (pace, sit down, stand up, type) for 600 s |
| zoneA, zoneB, ... | still | one person sitting or standing still inside the zone, normal breathing, for 600 s |

## Fixed setup (do not change between sessions or days)
- Mark zones on the floor with tape. Fix TX and RX positions and take a photo; describe it in `--setup`.
- Use one firmware variant for the whole dataset and pass it as `--firmware <tag>` every time (see docs/firmware_config.md).
- Nobody else in the room or walking past the zones. Keep doors and window blinds as normal; note any change in `--notes`.
- Same RX board on the same USB port every time. Laptop plugged in, sleep disabled.

## One collection day (about 55 min of recording)
Run from `host/logger` (Windows example). `--subject` is an anonymous id (P1, P2), not a name.

    python csi_logger.py --port COM5 --label empty  --activity baseline --duration 180 --room roomK --subject P1 --firmware <tag>
    python csi_logger.py --port COM5 --label empty  --activity empty    --duration 600 --room roomK --subject P1 --firmware <tag>
    python csi_logger.py --port COM5 --label zoneA  --activity moving   --duration 600 --room roomK --subject P1 --firmware <tag>
    python csi_logger.py --port COM5 --label zoneA  --activity still    --duration 600 --room roomK --subject P1 --firmware <tag>
    python csi_logger.py --port COM5 --label zoneB  --activity moving   --duration 600 --room roomK --subject P1 --firmware <tag>
    python csi_logger.py --port COM5 --label zoneB  --activity still    --duration 600 --room roomK --subject P1 --firmware <tag>

Randomise the order of zones between days. Finish the day with one more `empty` session (measures drift).

## How many
- At least 3 different days; aim for 5 or more spread over 2+ weeks. Day = calendar date, one baseline per day.
- If time allows, 2 sessions per (zone, activity) per day.
- A second room with identical boards and firmware gives the cross-room test (`--room` differs).
- A second person (`--subject P2`) is a bonus, not required for v1.

## Quality gate after every session (30 seconds)
Open the `.meta.json` next to the CSV:
- `status` is `complete`; `packets_logged` is close to `duration x rate_hz_device`
- `packet_loss_pct_est` below 5; `rate_hz_device` near the pilot value
Then run `python ../visualize/csi_visualize.py --csv <file>`. A walking session must look visibly different from an empty one.
Bad session: delete it and redo it. Do not keep data you would have to explain away.

## Rules that protect the results
- Never tune features or models on a day you later report as test data.
- Never split randomly by window. Use csi_eval.py splits (session, day, temporal, room).
- Do not commit `data/raw/`. Zip a day's folder and share it (Drive); expect about 70 MB per 10-minute session.
- Keep a short `data/session_log.csv` with day, room, subject, notes, anything unusual.
