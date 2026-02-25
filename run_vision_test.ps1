# PowerShell script to run Vision Module tests
# Usage: .\run_vision_test.ps1
# Usage with webcam: .\run_vision_test.ps1 -Webcam

param(
    [switch]$Webcam
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = "C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe"

if ($Webcam) {
    & $pythonExe $scriptDir/run_vision_test.py --webcam
} else {
    & $pythonExe $scriptDir/run_vision_test.py
}
