# Generate GitHub Raw URLs for KISYSTEM.txt
# Run this in C:\KISYSTEM\

$baseUrl = "https://raw.githubusercontent.com/JoeBoh71/KISYSTEM/main/"
$outputFile = "FILE_URLS.txt"

Write-Host "=== FETCH CURRENT CODE STATE ===" | Out-File $outputFile
Write-Host "" | Out-File $outputFile -Append

$counter = 3

# Get ALL files recursively, excluding git/cache folders
Get-ChildItem -Recurse -File | Where-Object {
    $_.FullName -notmatch '\\\.git\\' -and
    $_.FullName -notmatch '\\__pycache__\\' -and
    $_.FullName -notmatch '\\venv\\' -and
    $_.FullName -notmatch '\\node_modules\\'
} | ForEach-Object {
    $relativePath = $_.FullName.Replace((Get-Location).Path + "\", "").Replace("\", "/")
    $url = "$baseUrl$relativePath"
    Write-Host "$counter. $url" | Out-File $outputFile -Append
    $counter++
}

Write-Host ""
Write-Host "URLs generated in: $outputFile"
Write-Host "Copy these into your KISYSTEM.txt"
