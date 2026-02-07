# GCP Deployment Script for JD Parser Service
# ============================================
# This script builds and deploys the JD Parser service to Google Cloud Run

param(
    [Parameter(Mandatory=$false)]
    [string]$ProjectId = "zenith-486712",
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "us-central1",
    
    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "jd-parser"
)

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green "=========================================="
Write-ColorOutput Green "  JD Parser - GCP Deployment"
Write-ColorOutput Green "=========================================="

# Check if gcloud is installed
if (!(Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-ColorOutput Red "ERROR: gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
}

# Get project ID if not provided
if ([string]::IsNullOrEmpty($ProjectId)) {
    $ProjectId = gcloud config get-value project 2>$null
    if ([string]::IsNullOrEmpty($ProjectId)) {
        Write-ColorOutput Red "ERROR: No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    }
}

Write-ColorOutput Cyan "Project ID: $ProjectId"
Write-ColorOutput Cyan "Region: $Region"
Write-ColorOutput Cyan "Service: $ServiceName"
Write-ColorOutput Cyan ""

# Enable required APIs
Write-ColorOutput Yellow "Enabling required APIs..."
gcloud services enable containerregistry.googleapis.com --project=$ProjectId
gcloud services enable run.googleapis.com --project=$ProjectId
gcloud services enable cloudbuild.googleapis.com --project=$ProjectId

# Build and push Docker image
$ImageName = "gcr.io/$ProjectId/$ServiceName"
$ImageTag = "$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$FullImageName = "${ImageName}:${ImageTag}"

Write-ColorOutput Yellow "`nBuilding Docker image: $FullImageName"
docker build -t $FullImageName .

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: Docker build failed"
    exit 1
}

Write-ColorOutput Yellow "`nPushing image to GCR..."
docker push $FullImageName

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: Docker push failed"
    exit 1
}

# Deploy to Cloud Run
Write-ColorOutput Yellow "`nDeploying to Cloud Run..."
gcloud run deploy $ServiceName `
    --image=$FullImageName `
    --platform=managed `
    --region=$Region `
    --allow-unauthenticated `
    --port=8001 `
    --memory=1Gi `
    --cpu=1 `
    --max-instances=10 `
    --project=$ProjectId

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput Red "ERROR: Cloud Run deployment failed"
    exit 1
}

# Get service URL
$ServiceUrl = gcloud run services describe $ServiceName --platform=managed --region=$Region --format="value(status.url)" --project=$ProjectId

Write-ColorOutput Green "`n=========================================="
Write-ColorOutput Green "  Deployment Successful!"
Write-ColorOutput Green "=========================================="
Write-ColorOutput Cyan "Service URL: $ServiceUrl"
Write-ColorOutput Cyan "Health Check: $ServiceUrl/health"
Write-ColorOutput Cyan ""
Write-ColorOutput Yellow "To view logs:"
Write-ColorOutput White "  gcloud run logs read --service=$ServiceName --region=$Region --project=$ProjectId"
