$ErrorActionPreference = "Stop"

$PythonExe = ".\\.venv\\Scripts\\python.exe"
if (-not (Test-Path $PythonExe)) {
	$PythonExe = "python"
}

Write-Host "Skipping pip install (dependencies already installed)"
# & $PythonExe -m pip install -r requirements.txt
& $PythonExe -m src.runExperiment --config configs/baseFewShot.yaml
& $PythonExe -m src.runExperiment --config configs/improvedPrompt.yaml
& $PythonExe -m src.runExperiment --config configs/ablationHighTemp.yaml
