param([string]$Input="input.mp4")
python -m editron_cut_engine.cli --input $Input --output output_cut.mp4 --language auto --device cpu --compute-type int8 --preview-json decision.json
