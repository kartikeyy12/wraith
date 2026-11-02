# Firmware configuration

Source: Espressif esp-csi `get-started` csi_send / csi_recv, esp32s3 target, ESP-IDF 5.3.
Defaults (from the source): channel 11, ESP-NOW broadcast at 100 packets/s, MCS0, HT40 on both nodes,
software gain compensation (CONFIG_FORCE_GAIN 0), CSI printed over UART0 at 921600 baud.
The RX board's CSI stream leaves through the USB-C port wired to the UART bridge (labelled UART/COM).

## Variants (build with firmware/make_variant.sh in WSL after `source ~/esp-idf/export.sh`)
| variant | TX/RX bandwidth | gain handling | purpose |
|---|---|---|---|
| default | HT40 | software compensation | what is already built |
| ht20 | HT20 | software compensation | cleaner, documented LLTF + HT-LTF layout |
| ht20-gainlock | HT20 | RX locks AGC/FFT gain in hardware | tests whether gain jitter limits stability |

The variant build step has not been run by the author. If `idf.py build` fails, send the error.
Output goes to `firmware/prebuilt/<variant>/{tx-node,rx-node}/`. The existing flat folders
`firmware/prebuilt/tx-node` and `rx-node` are the default variant.

Flash (PowerShell, venv active, UART-labelled port):

    cd firmware\prebuilt\ht20\rx-node
    python -m esptool --chip esp32s3 -p COM5 -b 460800 write_flash "@flash_args"

## Pilot procedure (before collecting any dataset)
For each variant: flash both boards, put them in the planned positions, leave the room, then
    python csi_logger.py --port COM5 --label empty --activity baseline --duration 120 --firmware <variant>
    python csi_pilot_report.py --csv ../../data/raw/<file>.csv
Then repeat with someone walking in the zone. Compare across variants:
- packet rate and loss (want near 100/s, loss under 5%)
- packet-type mix: a single dominant type is good; several types means other traffic or mixed formats
- LLTF positions active as documented, per-packet amplitude CV (lower = steadier)
- empty-room window turb_std from csi_visualize.py, and the gap between walking and empty

## Decision record (fill in, then freeze)
- chosen variant:
- why (numbers from the pilot):
- date frozen:
- git commit of firmware sources:
Changing any of this after data collection starts invalidates earlier sessions. Record a new variant tag instead.
