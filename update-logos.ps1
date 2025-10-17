# PowerShell script to replace all logos with astra.png
# Created: October 18, 2025

$sourceLogo = "E:\Astra_Ai\static\static\astra.png"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupFolder = "E:\Astra_Ai\logo-backup-$timestamp"

# Check if source file exists
if (-not (Test-Path $sourceLogo)) {
    Write-Host "ERROR: Source file not found: $sourceLogo" -ForegroundColor Red
    exit 1
}

# Create backup folder
New-Item -ItemType Directory -Path $backupFolder -Force | Out-Null
Write-Host "Created backup folder: $backupFolder" -ForegroundColor Green

# List of files to replace
$filesToReplace = @(
    "E:\Astra_Ai\static\static\logo.png",
    "E:\Astra_Ai\static\static\favicon.png",
    "E:\Astra_Ai\static\static\favicon-dark.png",
    "E:\Astra_Ai\static\static\favicon-96x96.png",
    "E:\Astra_Ai\static\static\splash.png",
    "E:\Astra_Ai\static\static\splash-dark.png",
    "E:\Astra_Ai\static\static\apple-touch-icon.png",
    "E:\Astra_Ai\static\static\web-app-manifest-192x192.png",
    "E:\Astra_Ai\static\static\web-app-manifest-512x512.png",
    "E:\Astra_Ai\static\favicon.png"
)

Write-Host "`nReplacing logo files..." -ForegroundColor Cyan

foreach ($file in $filesToReplace) {
    if (Test-Path $file) {
        $fileName = Split-Path $file -Leaf
        Copy-Item $file "$backupFolder\$fileName" -Force
        Write-Host "  Backed up: $fileName" -ForegroundColor Yellow
        
        Copy-Item $sourceLogo $file -Force
        Write-Host "  Replaced: $fileName" -ForegroundColor Green
    } else {
        Write-Host "  Skipped (not found): $file" -ForegroundColor Gray
    }
}

# Generate favicon.ico
Write-Host "`nGenerating favicon.ico..." -ForegroundColor Cyan
$icoPath = "E:\Astra_Ai\static\static\favicon.ico"

if (Test-Path $icoPath) {
    Copy-Item $icoPath "$backupFolder\favicon.ico" -Force
    Write-Host "  Backed up: favicon.ico" -ForegroundColor Yellow
}

try {
    Add-Type -AssemblyName System.Drawing
    $img = [System.Drawing.Image]::FromFile($sourceLogo)
    $bmp = New-Object System.Drawing.Bitmap($img, 32, 32)
    
    $iconStream = New-Object System.IO.MemoryStream
    $bmp.Save($iconStream, [System.Drawing.Imaging.ImageFormat]::Png)
    
    $ico = [System.IO.File]::Create($icoPath)
    
    $ico.Write([byte[]](0, 0, 1, 0, 1, 0), 0, 6)
    
    $imageSize = $iconStream.Length
    $ico.WriteByte(32)
    $ico.WriteByte(32)
    $ico.WriteByte(0)
    $ico.WriteByte(0)
    $ico.Write([byte[]](1, 0), 0, 2)
    $ico.Write([byte[]](32, 0), 0, 2)
    $ico.Write([BitConverter]::GetBytes([int]$imageSize), 0, 4)
    $ico.Write([BitConverter]::GetBytes(22), 0, 4)
    
    $iconStream.Position = 0
    $iconStream.CopyTo($ico)
    
    $ico.Close()
    $iconStream.Close()
    $bmp.Dispose()
    $img.Dispose()
    
    Write-Host "  Generated: favicon.ico (32x32)" -ForegroundColor Green
} catch {
    Write-Host "  Warning: Could not generate favicon.ico: $_" -ForegroundColor Yellow
}

# Generate favicon.svg
Write-Host "`nGenerating favicon.svg..." -ForegroundColor Cyan
$svgPath = "E:\Astra_Ai\static\static\favicon.svg"

if (Test-Path $svgPath) {
    Copy-Item $svgPath "$backupFolder\favicon.svg" -Force
    Write-Host "  Backed up: favicon.svg" -ForegroundColor Yellow
}

$base64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($sourceLogo))

$svgContent = @"
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 100 100">
  <image width="100" height="100" xlink:href="data:image/png;base64,$base64"/>
</svg>
"@

Set-Content -Path $svgPath -Value $svgContent -Encoding UTF8
Write-Host "  Generated: favicon.svg" -ForegroundColor Green

Write-Host "`n=== Logo replacement complete ===" -ForegroundColor Green
Write-Host "Total files replaced: $($filesToReplace.Count + 2)" -ForegroundColor Cyan
Write-Host "Backup location: $backupFolder" -ForegroundColor Cyan
