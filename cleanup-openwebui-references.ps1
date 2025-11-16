# 🎯 Cleanup Script - Remove All Open WebUI References

## This script will replace all "Open WebUI" references with "Astra AI"

Write-Host "🚀 Starting Astra AI Cleanup Script..." -ForegroundColor Green
Write-Host ""

$replacements = @{
    "open-webui" = "astra-ai"
    "open_webui" = "astra_ai"
    "Open WebUI" = "Astra AI"
    "Open-WebUI" = "Astra AI"
    "OpenWebUI" = "AstraAI"
    "OPEN_WEBUI" = "ASTRA_AI"
}

$filesToUpdate = @(
    # Package configs
    "package.json",
    "package-lock.json",
    "pyproject.toml",
    
    # Docker configs
    "docker-compose.yaml",
    "docker-compose.prod.yaml",
    "docker-compose.api.yaml",
    "docker-compose.gpu.yaml",
    "Dockerfile",
    "run.sh",
    "Makefile",
    
    # Documentation
    "README.md",
    "TROUBLESHOOTING.md",
    "docs\SECURITY.md",
    "kubernetes\helm\README.md",
    
    # Frontend components
    "src\routes\error\+page.svelte",
    "src\lib\components\layout\UpdateInfoToast.svelte",
    "src\lib\components\layout\Sidebar\UserMenu.svelte",
    "src\lib\components\chat\ToolServersModal.svelte",
    "src\lib\components\chat\Settings\Tools.svelte",
    "src\lib\components\admin\Settings\General.svelte",
    "src\lib\components\admin\Settings\Connections.svelte",
    "src\lib\components\admin\Functions\FunctionEditor.svelte"
)

$translationFiles = Get-ChildItem -Path "src\lib\i18n\locales" -Filter "translation.json" -Recurse

$totalFiles = $filesToUpdate.Count + $translationFiles.Count
$processedFiles = 0
$updatedFiles = 0
$errors = @()

Write-Host "📊 Found $totalFiles files to process" -ForegroundColor Cyan
Write-Host ""

# Function to replace text in file
function Update-FileContent {
    param(
        [string]$FilePath,
        [hashtable]$Replacements
    )
    
    try {
        if (Test-Path $FilePath) {
            $content = Get-Content $FilePath -Raw -Encoding UTF8
            $originalContent = $content
            $changed = $false
            
            foreach ($key in $Replacements.Keys) {
                $value = $Replacements[$key]
                if ($content -match [regex]::Escape($key)) {
                    $content = $content -replace [regex]::Escape($key), $value
                    $changed = $true
                }
            }
            
            if ($changed) {
                Set-Content -Path $FilePath -Value $content -Encoding UTF8 -NoNewline
                return $true
            }
        } else {
            Write-Host "  ⚠️  File not found: $FilePath" -ForegroundColor Yellow
        }
        return $false
    } catch {
        Write-Host "  ❌ Error processing $FilePath : $_" -ForegroundColor Red
        return $false
    }
}

# Process main files
Write-Host "🔧 Processing configuration files..." -ForegroundColor Yellow
foreach ($file in $filesToUpdate) {
    $processedFiles++
    $percentage = [math]::Round(($processedFiles / $totalFiles) * 100, 1)
    
    Write-Host "[$percentage%] Processing: $file" -ForegroundColor Gray
    
    $fullPath = Join-Path $PWD $file
    if (Update-FileContent -FilePath $fullPath -Replacements $replacements) {
        $updatedFiles++
        Write-Host "  ✅ Updated" -ForegroundColor Green
    } else {
        Write-Host "  ⏭️  No changes needed" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "🌍 Processing translation files..." -ForegroundColor Yellow
foreach ($file in $translationFiles) {
    $processedFiles++
    $percentage = [math]::Round(($processedFiles / $totalFiles) * 100, 1)
    
    Write-Host "[$percentage%] Processing: $($file.FullName.Replace($PWD, '.'))" -ForegroundColor Gray
    
    if (Update-FileContent -FilePath $file.FullName -Replacements $replacements) {
        $updatedFiles++
        Write-Host "  ✅ Updated" -ForegroundColor Green
    } else {
        Write-Host "  ⏭️  No changes needed" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Summary:" -ForegroundColor White
Write-Host "  Total files processed: $processedFiles" -ForegroundColor White
Write-Host "  Files updated: $updatedFiles" -ForegroundColor Green
Write-Host "  Files skipped: $($processedFiles - $updatedFiles)" -ForegroundColor Gray
Write-Host ""

if ($updatedFiles -gt 0) {
    Write-Host "🔍 Next Steps:" -ForegroundColor Yellow
    Write-Host "  1. Review changes: git diff" -ForegroundColor White
    Write-Host "  2. Test the application: npm run dev" -ForegroundColor White
    Write-Host "  3. Commit changes: git add . && git commit -m 'Rebrand to Astra AI'" -ForegroundColor White
    Write-Host "  4. Push to GitHub: git push" -ForegroundColor White
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Test thoroughly before deploying!" -ForegroundColor Red
} else {
    Write-Host "✨ All files are already clean!" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎯 GitHub Links to Update Manually:" -ForegroundColor Yellow
Write-Host "  - Update repository description on GitHub" -ForegroundColor White
Write-Host "  - Update social media links in README" -ForegroundColor White
Write-Host "  - Update any external documentation" -ForegroundColor White
Write-Host ""

# Create backup note
$backupNote = @"
# Cleanup Backup Note
Date: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Files Updated: $updatedFiles
Status: Completed

If you need to revert, use: git reset --hard HEAD
"@

Set-Content -Path "cleanup_backup_note.txt" -Value $backupNote

Write-Host "💾 Backup note created: cleanup_backup_note.txt" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Ready to build your Astra AI brand!" -ForegroundColor Green
