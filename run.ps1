$ErrorActionPreference = "Stop"

$PythonExe = ".\\.venv\\Scripts\\python.exe"
if (-not (Test-Path $PythonExe)) {
	$PythonExe = "python"
}

Write-Host "Skipping pip install (dependencies already installed)"
# & $PythonExe -m pip install -r requirements.txt
python -m src.runExperiment --config configs/baseFewShot.yaml
python -m src.runExperiment --config configs/improvedPrompt.yaml
python -m src.runExperiment --config configs/ablationHighTemp.yaml
python -m src.runExperiment --config configs/ablationLowTopP.yaml
python -m src.runSafetyTests --modelId Qwen/Qwen2.5-1.5B-Instruct --out results/analysis/safetyTests.txt
