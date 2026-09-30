# Data Collection & Labeling Protocol - Phase 1 (Zone Localization)

## Zones (v1 minimum set)
- empty  - no person present, baseline noise floor
- zoneA  - same room as RX node, line-of-sight to TX
- zoneB  - adjacent room / behind one wall from RX

(Add zoneC, zoneD later once this 3-class baseline works cleanly.)

## Per-session procedure
1. Start the TX node, confirm RX node is printing CSI over serial.
2. Run csi_logger.py --label <zone> --duration <N> from host/logger/.
3. Walk into position for that zone BEFORE starting the script, stand there for the full session (do not start recording while still walking in).
4. For empty sessions: leave the room entirely, no motion anywhere near either node.
5. Minimum session length: 10 minutes (600s). At 100 packets/sec that is ~60,000 packets, ~60 ten-second NBVI windows per session.

## Repeat across conditions, not just once per zone
Collect at least 3 separate sessions per zone, spread across:
- different times of day (morning vs evening WiFi congestion differs)
- different days (furniture/clutter shifts, your own position within the zone varies)
- ideally, not back-to-back (avoid a single environmental snapshot dominating the dataset)

This directly addresses the biggest weakness in hobbyist CSI projects: a model that looks great because it memorized one room's static fingerprint, not because it generalizes.

## File naming
csi_logger.py already timestamps output as <label>_<unixtime>.csv - keep this, do not rename. Track session metadata in a simple log at data/session_log.csv with columns:
date, time, label, session_file, notes

Example row:
2026-10-05, morning, zoneA, zoneA_1234567890.csv, baseline furniture

## Train/test split rule (important, do not skip this)
When it is time to train a classifier: split by session, not by window.
- e.g. sessions 1-2 of each zone go to the training set, session 3 is the held-out test set.
- Never randomly shuffle all windows from all sessions together before splitting. This leaks session-specific noise into both train and test and inflates reported accuracy in a way that will not hold up under scrutiny.

## Sanity check after every session
Run csi_visualize.py on the fresh CSV immediately after logging, before moving to the next zone. Confirms the RX node was actually capturing and gives a visual gut-check that motion/stillness is showing up as expected.
