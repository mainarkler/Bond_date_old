#requires -Version 5.1
<#
VM PDF -> classic Outlook, без Python.
Конфигурация: %USERPROFILE%\vm_report_sender.json
Нужен msedgedriver.exe, совместимый с установленным Microsoft Edge.
#>

$ErrorActionPreference = "Stop"
$ConfigPath = Join-Path $env:USERPROFILE "vm_report_sender.json"
$DownloadDir = Join-Path $env:TEMP "vm_report_sender"
$DriverPort = 9515
$DriverHost = "http://127.0.0.1:$DriverPort"

function Fail([string]$Message) { throw $Message }

function Load-Config {
    if (-not (Test-Path $ConfigPath)) { Fail "Не найден файл конфигурации: $ConfigPath" }
    $cfg = Get-Content -Raw -Encoding UTF8 $ConfigPath | ConvertFrom-Json
    $recipients = @($cfg.recipients | ForEach-Object { [string]$_ } | Where-Object { $_.Trim() })
    if ($recipients.Count -eq 0) { Fail "В конфигурации нет recipients." }
    $timeout = if ($null -ne $cfg.timeout_seconds) { [int]$cfg.timeout_seconds } else { 180 }
    return @{
        url = if ($cfg.url) { [string]$cfg.url } else { "https://bonddate.streamlit.app/?action=vm_pdf" }
        recipients = $recipients
        subject = if ($cfg.subject) { [string]$cfg.subject } else { "VM отчет $(Get-Date -Format 'dd.MM.yyyy')" }
        send = [bool]$cfg.send
        timeout_seconds = $timeout
    }
}

function Invoke-WebDriver {
    param([string]$Method,[string]$Path,[object]$Body=$null)
    $uri = "$DriverHost$Path"
    if ($null -eq $Body) { return Invoke-RestMethod -Method $Method -Uri $uri -UseBasicParsing }
    $json = $Body | ConvertTo-Json -Depth 10 -Compress
    return Invoke-RestMethod -Method $Method -Uri $uri -ContentType "application/json; charset=utf-8" -Body $json -UseBasicParsing
}

function Start-EdgeDriver {
    $driver = $env:MSEDGEDRIVER
    if (-not $driver) {
        $local = Join-Path $PSScriptRoot "msedgedriver.exe"
        if (Test-Path $local) { $driver = $local }
    }
    if (-not $driver) {
        $cmd = Get-Command msedgedriver.exe -ErrorAction SilentlyContinue
        if ($cmd) { $driver = $cmd.Source }
    }
    if (-not $driver -or -not (Test-Path $driver)) {
        Fail "Не найден msedgedriver.exe. Положите его рядом со скриптом или задайте MSEDGEDRIVER."
    }
    $process = Start-Process -FilePath $driver -ArgumentList "--port=$DriverPort" -PassThru -WindowStyle Hidden
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 250
        try { Invoke-WebDriver -Method Get -Path "/status" | Out-Null; $ready = $true; break } catch {}
    }
    if (-not $ready) {
        if (!$process.HasExited) { Stop-Process -Id $process.Id -Force }
        Fail "msedgedriver.exe не запустился на порту $DriverPort."
    }
    return $process
}

function New-WebDriverSession {
    param([string]$DownloadDirectory)
    $prefs = @{
        "download.default_directory" = $DownloadDirectory
        "download.prompt_for_download" = $false
        "download.directory_upgrade" = $true
        "plugins.always_open_pdf_externally" = $true
    }
    $body = @{
        capabilities = @{
            alwaysMatch = @{
                browserName = "MicrosoftEdge"
                "ms:edgeOptions" = @{
                    args = @("--headless=new","--disable-gpu","--no-first-run","--no-default-browser-check","--window-size=1600,1200")
                    prefs = $prefs
                }
            }
        }
    }
    $response = Invoke-WebDriver -Method Post -Path "/session" -Body $body
    if ($response.value.sessionId) { return [string]$response.value.sessionId }
    if ($response.sessionId) { return [string]$response.sessionId }
    Fail "Не удалось создать Edge WebDriver session."
}

function Find-ElementByXPath {
    param([string]$SessionId,[string]$XPath)
    return Invoke-WebDriver -Method Post -Path "/session/$SessionId/element" -Body @{ using="xpath"; value=$XPath }
}

function Wait-ForDownload {
    param([string]$Directory,[int]$TimeoutSeconds)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $partial = @(Get-ChildItem -Path $Directory -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '\.(crdownload|tmp|part)$' })
        $pdfs = @(Get-ChildItem -Path $Directory -Filter "*.pdf" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
        if ($pdfs.Count -gt 0 -and $partial.Count -eq 0) {
            $pdf = $pdfs[0]
            if ($pdf.Length -gt 10000) { Start-Sleep -Milliseconds 500; return $pdf.FullName }
        }
        Start-Sleep -Milliseconds 500
    }
    Fail "PDF не был загружен за $TimeoutSeconds секунд."
}

function Download-VM-Pdf {
    param([string]$Url,[int]$TimeoutSeconds)
    if (Test-Path $DownloadDir) {
        Get-ChildItem -Path $DownloadDir -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    } else {
        New-Item -ItemType Directory -Path $DownloadDir -Force | Out-Null
    }
    $driverProcess = $null
    $sessionId = $null
    try {
        $driverProcess = Start-EdgeDriver
        $sessionId = New-WebDriverSession -DownloadDirectory $DownloadDir
        Invoke-WebDriver -Method Post -Path "/session/$sessionId/url" -Body @{ url=$Url } | Out-Null
        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        $element = $null
        while ((Get-Date) -lt $deadline) {
            try {
                $candidate = Find-ElementByXPath -SessionId $sessionId -XPath "//button[contains(normalize-space(.), 'Скачать VM PDF')]"
                if ($candidate.value.'element-6066-11e4-a52e-4f735466cecf') {
                    $element = $candidate.value.'element-6066-11e4-a52e-4f735466cecf'
                    break
                }
            } catch {}
            Start-Sleep -Seconds 1
        }
        if (-not $element) { Fail "На странице не найдена кнопка 'Скачать VM PDF'." }
        Invoke-WebDriver -Method Post -Path "/session/$sessionId/element/$element/click" -Body @{} | Out-Null
        return Wait-ForDownload -Directory $DownloadDir -TimeoutSeconds $TimeoutSeconds
    }
    finally {
        if ($sessionId) { try { Invoke-WebDriver -Method Delete -Path "/session/$sessionId" | Out-Null } catch {} }
        if ($driverProcess -and !$driverProcess.HasExited) { Stop-Process -Id $driverProcess.Id -Force -ErrorAction SilentlyContinue }
    }
}

function Send-OutlookMail {
    param([string]$PdfPath,[string[]]$Recipients,[string]$Subject,[bool]$Send)
    $outlook = New-Object -ComObject Outlook.Application
    $mail = $outlook.CreateItem(0)
    $mail.To = ($Recipients -join "; ")
    $mail.Subject = $Subject
    $mail.Body = ""
    $mail.Attachments.Add($PdfPath) | Out-Null
    if ($Send) {
        $mail.Send()
        Write-Host "Письмо отправлено: $($Recipients -join ', ')" -ForegroundColor Green
    } else {
        $mail.Display()
        Write-Host "Письмо создано в Outlook, но НЕ отправлено (send=false)." -ForegroundColor Yellow
    }
}

$config = Load-Config
Write-Host "Получаю VM PDF..." -ForegroundColor Cyan
$pdfPath = Download-VM-Pdf -Url $config.url -TimeoutSeconds $config.timeout_seconds
Write-Host "PDF получен: $pdfPath" -ForegroundColor Green
Send-OutlookMail -PdfPath $pdfPath -Recipients $config.recipients -Subject $config.subject -Send $config.send
