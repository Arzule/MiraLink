# Construye dist\MiraLink.exe y lo verifica con --self-test (sin cámara).
# Uso (desde la carpeta del proyecto, en PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
#   powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -SkipTests   # sin la suite de tests
param([switch]$SkipTests)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\venv\Scripts\python.exe"

Write-Host "1/5 Herramientas de construcción (PyInstaller)"
& $py -m pip install -q -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw "No se pudo instalar requirements-build.txt" }

if (-not (Test-Path "models\face_landmarker.task")) {
    Write-Host "    Falta el modelo: descargándolo"
    & $py scripts\download_model.py
    if ($LASTEXITCODE -ne 0) { throw "No se pudo descargar models\face_landmarker.task" }
}

if (-not $SkipTests) {
    Write-Host "2/5 Tests"
    & $py -m unittest discover -s tests
    if ($LASTEXITCODE -ne 0) { throw "Hay tests fallando: no se construye el ejecutable" }
} else {
    Write-Host "2/5 Tests omitidos (-SkipTests)"
}

Write-Host "3/5 PyInstaller (tarda unos minutos)"
& $py -m PyInstaller MiraLink.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller falló" }

Write-Host "4/5 Self-test del ejecutable en una carpeta limpia"
$check = Join-Path $env:TEMP "miralink_exe_check"
if (Test-Path $check) { Remove-Item -Recurse -Force $check }
New-Item -ItemType Directory $check | Out-Null
Copy-Item "dist\MiraLink.exe" $check
$p = Start-Process -FilePath (Join-Path $check "MiraLink.exe") -ArgumentList "--self-test" -Wait -PassThru
Get-Content -Encoding UTF8 (Join-Path $check "miralink_selftest.txt")
if ($p.ExitCode -ne 0) { throw "El self-test del ejecutable falló (código $($p.ExitCode))" }
if (-not (Test-Path (Join-Path $check "config.json"))) { throw "El ejecutable no creó config.json junto a él" }

Write-Host "5/5 Listo"
$exe = Get-Item "dist\MiraLink.exe"
Write-Host ("Ejecutable: {0} ({1:N0} MB)" -f $exe.FullName, ($exe.Length / 1MB))
Write-Host "Para probarlo en otra PC copia SOLO MiraLink.exe a una carpeta con permisos de escritura."
