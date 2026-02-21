set -e

# I used this script to run all experiment configs quickly.
# Each line launches one experiment with a different YAML setup.
python -m pip install -r requirements.txt
python -m src.runExperiment --config configs/baseFewShot.yaml
python -m src.runExperiment --config configs/improvedPrompt.yaml
python -m src.runExperiment --config configs/ablationHighTemp.yaml