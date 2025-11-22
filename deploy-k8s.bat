@echo off
REM MLOps Federated Learning Kubernetes Deployment Script for Windows
setlocal EnableDelayedExpansion

REM Default values
set ENVIRONMENT=development
set NAMESPACE=
set DRY_RUN=false
set SKIP_BUILD=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :validate_args
if "%~1"=="-e" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--environment" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-n" (
    set NAMESPACE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--namespace" (
    set NAMESPACE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-d" (
    set DRY_RUN=true
    shift
    goto :parse_args
)
if "%~1"=="--dry-run" (
    set DRY_RUN=true
    shift
    goto :parse_args
)
if "%~1"=="-s" (
    set SKIP_BUILD=true
    shift
    goto :parse_args
)
if "%~1"=="--skip-build" (
    set SKIP_BUILD=true
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :usage
if "%~1"=="--help" goto :usage
echo ERROR: Unknown option: %~1
goto :usage

:validate_args
if not "%ENVIRONMENT%"=="development" if not "%ENVIRONMENT%"=="production" (
    echo ERROR: Invalid environment: %ENVIRONMENT%. Must be 'development' or 'production'
    exit /b 1
)

REM Set namespace
if "%NAMESPACE%"=="" (
    if "%ENVIRONMENT%"=="production" (
        set NAMESPACE=mlops-fl-prod
    ) else (
        set NAMESPACE=mlops-fl-dev
    )
)

echo [INFO] Starting MLOps Federated Learning deployment...
echo [INFO] Environment: %ENVIRONMENT%
echo [INFO] Namespace: %NAMESPACE%
echo [INFO] Dry run: %DRY_RUN%

REM Check prerequisites
echo [INFO] Checking prerequisites...
where kubectl >nul 2>&1
if errorlevel 1 (
    echo ERROR: kubectl is not installed or not in PATH
    exit /b 1
)

where kustomize >nul 2>&1
if errorlevel 1 (
    echo ERROR: kustomize is not installed or not in PATH
    exit /b 1
)

if "%SKIP_BUILD%"=="false" (
    where docker >nul 2>&1
    if errorlevel 1 (
        echo ERROR: docker is not installed or not in PATH
        exit /b 1
    )
)

kubectl cluster-info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Cannot connect to Kubernetes cluster
    exit /b 1
)

echo [INFO] Prerequisites check passed

REM Build Docker images
if "%SKIP_BUILD%"=="false" (
    echo [INFO] Building Docker images...
    
    echo [INFO] Building FL Server image...
    docker build -f Dockerfile.server -t fl-server:latest .
    if errorlevel 1 exit /b 1
    
    echo [INFO] Building API Service image...
    docker build -f Dockerfile.api -t api-service:latest .
    if errorlevel 1 exit /b 1
    
    echo [INFO] Building Dashboard image...
    docker build -f Dockerfile.dashboard -t dashboard:latest .
    if errorlevel 1 exit /b 1
    
    echo [INFO] Building FL Client image...
    docker build -f Dockerfile.client -t fl-client:latest .
    if errorlevel 1 exit /b 1
    
    echo [INFO] Docker images built successfully
) else (
    echo [WARNING] Skipping Docker image build
)

REM Deploy to Kubernetes
set OVERLAY_PATH=k8s\overlays\%ENVIRONMENT%
if not exist "%OVERLAY_PATH%" (
    echo ERROR: Environment overlay not found: %OVERLAY_PATH%
    exit /b 1
)

echo [INFO] Deploying to Kubernetes environment: %ENVIRONMENT%

if "%DRY_RUN%"=="true" (
    echo [INFO] Performing dry run...
    kustomize build "%OVERLAY_PATH%" | kubectl apply --dry-run=client -f -
) else (
    echo [INFO] Applying Kubernetes manifests...
    kustomize build "%OVERLAY_PATH%" | kubectl apply -f -
    if errorlevel 1 exit /b 1
    
    echo [INFO] Waiting for deployments to be ready...
    kubectl wait --for=condition=available --timeout=300s deployment --all -n %NAMESPACE%
    
    echo [INFO] Deployment status:
    kubectl get pods -n %NAMESPACE% -o wide
    kubectl get services -n %NAMESPACE% -o wide
)

if "%DRY_RUN%"=="false" (
    echo.
    echo [INFO] Deployment completed successfully!
    echo.
    echo [INFO] Access Information:
    echo ====================
    kubectl get services -n %NAMESPACE%
    echo.
    if "%ENVIRONMENT%"=="development" (
        echo [INFO] To access services:
        echo - API Service: http://localhost:30800
        echo - Dashboard: http://localhost:30851
        echo - Grafana: http://localhost:30300 ^(admin/admin^)
    ) else (
        echo [INFO] Check LoadBalancer external IPs above
    )
    echo.
    echo [INFO] Useful commands:
    echo - View pods: kubectl get pods -n %NAMESPACE%
    echo - View logs: kubectl logs -f deployment/^<deployment-name^> -n %NAMESPACE%
    echo - Port forward: kubectl port-forward service/^<service-name^> ^<local-port^>:^<service-port^> -n %NAMESPACE%
    echo - Delete deployment: kubectl delete namespace %NAMESPACE%
)

echo [INFO] Deployment script completed!
goto :end

:usage
echo Usage: %0 [OPTIONS]
echo.
echo Deploy MLOps Federated Learning system to Kubernetes
echo.
echo OPTIONS:
echo     -e, --environment   Environment ^(development^|production^) [default: development]
echo     -n, --namespace     Override namespace
echo     -d, --dry-run       Perform a dry run
echo     -s, --skip-build    Skip Docker image building
echo     -h, --help          Show this help message
echo.
echo EXAMPLES:
echo     %0 --environment development
echo     %0 --environment production --namespace mlops-fl-prod
echo     %0 --dry-run --environment development
exit /b 0

:end
endlocal