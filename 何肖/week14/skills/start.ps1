$key = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "User")
if (-not $key) {
    $key = [Environment]::GetEnvironmentVariable("DASHSCOPE_API_KEY", "Machine")
}
if (-not $key) {
    Write-Error "DASHSCOPE_API_KEY is not set. Please configure it in system environment variables first."
    exit 1
}
$env:DASHSCOPE_API_KEY = $key
Set-Location $PSScriptRoot
python -m uvicorn src.main:app --host 127.0.0.1 --port 9000
