#!/usr/bin/env bash
# Build a firmware variant from firmware/tx-node and firmware/rx-node WITHOUT touching those folders.
#
#   source ~/esp-idf/export.sh
#   bash firmware/make_variant.sh ht20            # HT20 TX + RX (cleaner, documented LLTF/HT-LTF layout)
#   bash firmware/make_variant.sh ht20-gainlock   # HT20 + RX locks AGC/FFT gain in hardware after the first packets
#   bash firmware/make_variant.sh default         # unchanged (HT40, software gain compensation)
#
# Output: firmware/prebuilt/<variant>/{tx-node,rx-node}/  (flash with esptool + "@flash_args")
# Set DRY_RUN=1 to only apply and verify the source edits (no idf.py build).
# NOTE: the build step has not been run by the author of this script; if idf.py fails, send the error.
set -euo pipefail

VARIANT="${1:?usage: make_variant.sh default|ht20|ht20-gainlock}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN="${DRY_RUN:-0}"

for NODE in tx-node rx-node; do
  SRC="$ROOT/firmware/$NODE"
  WORK="$ROOT/firmware/_variants/$VARIANT/$NODE"
  rm -rf "$WORK"; mkdir -p "$WORK"
  cp -r "$SRC/CMakeLists.txt" "$SRC/main" "$SRC/sdkconfig.defaults" "$WORK/"
  C="$WORK/main/app_main.c"

  case "$VARIANT" in
    default) ;;
    ht20|ht20-gainlock)
      sed -i -E 's/^(#define CONFIG_WIFI_BANDWIDTH[[:space:]]+)WIFI_BW_HT40/\1WIFI_BW_HT20/;
                 s/^(#define CONFIG_WIFI_2G_BANDWIDTHS[[:space:]]+)WIFI_BW_HT40/\1WIFI_BW_HT20/;
                 s/^(#define CONFIG_ESP_NOW_PHYMODE[[:space:]]+)WIFI_PHY_MODE_HT40/\1WIFI_PHY_MODE_HT20/' "$C"
      grep -Eq '^#define CONFIG_WIFI_BANDWIDTH[[:space:]]+WIFI_BW_HT20' "$C" || { echo "edit failed: WIFI_BANDWIDTH in $NODE"; exit 1; }
      grep -Eq '^#define CONFIG_ESP_NOW_PHYMODE[[:space:]]+WIFI_PHY_MODE_HT20' "$C" || { echo "edit failed: ESP_NOW_PHYMODE in $NODE"; exit 1; }
      if [[ "$VARIANT" == "ht20-gainlock" && "$NODE" == "rx-node" ]]; then
        sed -i -E 's/^(#define CONFIG_FORCE_GAIN[[:space:]]+)0/\11/' "$C"
        grep -Eq '^#define CONFIG_FORCE_GAIN[[:space:]]+1' "$C" || { echo "edit failed: FORCE_GAIN"; exit 1; }
      fi ;;
    *) echo "unknown variant $VARIANT"; exit 1 ;;
  esac
  echo "[$NODE] source edits for '$VARIANT' applied and verified"

  [[ "$DRY_RUN" == "1" ]] && continue
  (cd "$WORK" && idf.py set-target esp32s3 && idf.py build)
  DEST="$ROOT/firmware/prebuilt/$VARIANT/$NODE"
  mkdir -p "$DEST/bootloader" "$DEST/partition_table"
  cp "$WORK/build/flash_args" "$DEST/"
  cp "$WORK/build/bootloader/bootloader.bin" "$DEST/bootloader/"
  cp "$WORK/build/partition_table/partition-table.bin" "$DEST/partition_table/"
  cp "$WORK/build/$NODE.bin" "$DEST/" 2>/dev/null || cp "$WORK"/build/*.bin "$DEST/" 
  echo "[$NODE] -> $DEST"
done
echo "done: $VARIANT"
