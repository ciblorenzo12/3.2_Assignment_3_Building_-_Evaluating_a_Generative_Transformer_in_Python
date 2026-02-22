set -e

# I used this script to run all experiment configs quickly.
# Each line launches one experiment with a different YAML setup.
if [ -x "./.venv/Scripts/python.exe" ]; then
	PYTHON="./.venv/Scripts/python.exe"
elif [ -x "./.venv/bin/python" ]; then
	PYTHON="./.venv/bin/python"
else
	PYTHON="python"
fi

echo "Skipping pip install (dependencies already installed)"
# "$PYTHON" -m pip install -r requirements.txt
python -m src.runExperiment --config configs/baseFewShot.yaml
python -m src.runExperiment --config configs/improvedPrompt.yaml
python -m src.runExperiment --config configs/ablationHighTemp.yaml
python -m src.runExperiment --config configs/ablationLowTopP.yaml
python -m src.runSafetyTests --modelId Qwen/Qwen2.5-1.5B-Instruct --out results/analysis/safetyTests.txt