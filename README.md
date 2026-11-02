# Wraith

WiFi CSI sensing on two ESP32-S3 boards: zone-level presence and localization, with a leakage-safe
evaluation pipeline.

## Layout
    firmware/   esp-csi TX/RX sources, prebuilt binaries, make_variant.sh
    host/logger csi_logger.py (labeled capture, needs only pyserial), csi_parser.py
    host/dsp    csi_io, csi_features, csi_eval, csi_pilot_report
    host/visualize  csi_visualize.py
    host/tools  synth_csi.py (synthetic data, pipeline tests only)
    docs/       labeling_protocol.md, evaluation_plan.md, firmware_config.md
    tests/      python tests/test_pipeline.py

## Collect data (Windows or Linux)
    pip install -r requirements-collect.txt
    python host/logger/csi_logger.py --port COM5 --label zoneA --activity moving --duration 600 --room roomK --firmware default
See docs/labeling_protocol.md for the full routine.

## Analyse
    pip install -r requirements-analysis.txt
    python tests/test_pipeline.py
    python host/dsp/csi_eval.py --data data/raw --suite --out results/report.md

Raw I/Q is stored losslessly; only the LLTF block (64 subcarriers, 52 usable) is used for features.
Status: pipeline tested on synthetic data only. No real-world accuracy has been measured yet.
