@echo off
REM ============================================================================
REM Batch wrapper for project.ps1 - ML Trading Pipeline
REM ============================================================================
REM Usage: run [command] [options]
REM Example: run help
REM          run train-xgb
REM          run dashboard
REM ============================================================================

setlocal enabledelayedexpansion

REM Check if PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: PowerShell is not available on this system.
    exit /b 1
)

REM Pass all arguments to PowerShell script
if "%~1"=="" (
    REM No arguments, show help
    powershell -ExecutionPolicy Bypass -File ".\project.ps1" help
) else (
    REM Pass all arguments
    powershell -ExecutionPolicy Bypass -File ".\project.ps1" %*
)

endlocal