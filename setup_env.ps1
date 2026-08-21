$ErrorActionPreference = "Stop"

$EnvironmentName = "bio_diffusion"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found on PATH. Install Miniconda or Anaconda first."
}

$environmentExists = conda env list | Select-String -Pattern "^$EnvironmentName\s"
if (-not $environmentExists) {
    conda create --yes --name $EnvironmentName python=3.10
}

Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Green
conda run --name $EnvironmentName python -m pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing project dependencies..." -ForegroundColor Green
$requirements = Get-Content (Join-Path $ProjectRoot "requirements.txt") | Where-Object { $_ -and $_ -notmatch '^\s*torch\s' }
conda run --name $EnvironmentName python -m pip install $requirements

Write-Host "Running environment verification..." -ForegroundColor Green
conda run --name $EnvironmentName python (Join-Path $ProjectRoot "test_setup.py")

Write-Host "Environment '$EnvironmentName' is ready. Use 'conda activate $EnvironmentName' for interactive work." -ForegroundColor Green