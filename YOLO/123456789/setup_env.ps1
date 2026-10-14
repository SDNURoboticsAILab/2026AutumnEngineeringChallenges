$ErrorActionPreference = 'Continue'

# Idempotent YOLO environment setup / repair script.
# Put it in the project root (next to train.py) and just run it:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1
# It will SKIP the install when the environment already works.

$proj = $PSScriptRoot
$venv = Join-Path $proj '.venv'
$py   = Join-Path $venv 'Scripts\python.exe'

Write-Host "project = $proj"

function Test-Env {
    if (-not (Test-Path $py)) { return $false }
    $out = & $py -c "import torch, ultralytics; print('ENVOK')" 2>&1
    return (($out -join ' ') -match 'ENVOK')
}

if (Test-Env) {
    Write-Host '[SKIP] environment already installed, nothing to do.'
    & $py -c "import torch, torchvision, ultralytics, cv2; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('torchvision', torchvision.__version__); print('ultralytics', ultralytics.__version__); print('opencv', cv2.__version__)"
    Write-Host '[PASS]'
    exit 0
}

Write-Host '[1/5] creating venv (python 3.14) ...'
if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
py -3.14 -m venv $venv
if (-not (Test-Path $py)) { Write-Host '[FAIL] venv python not created'; exit 2 }

Write-Host '[2/5] upgrading pip ...'
& $py -m pip install --upgrade pip

Write-Host '[3/5] installing torch + torchvision (CUDA 12.6, ~2.6GB, be patient) ...'
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Write-Host '[4/5] installing ultralytics ...'
& $py -m pip install ultralytics

Write-Host '[5/5] verifying ...'
& $py -c "import torch, torchvision, ultralytics, cv2; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('torchvision', torchvision.__version__); print('ultralytics', ultralytics.__version__); print('opencv', cv2.__version__)"
& $py -m pip freeze | Out-File -Encoding utf8 (Join-Path $proj 'requirements.txt')

if (Test-Env) { Write-Host '[PASS] environment ready' } else { Write-Host '[FAIL] please paste the output above' }
