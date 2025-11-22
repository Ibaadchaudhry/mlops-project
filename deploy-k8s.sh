#!/bin/bash

# MLOps Federated Learning Kubernetes Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
NAMESPACE=""
DRY_RUN=false
SKIP_BUILD=false

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy MLOps Federated Learning system to Kubernetes

OPTIONS:
    -e, --environment   Environment (development|production) [default: development]
    -n, --namespace     Override namespace
    -d, --dry-run       Perform a dry run
    -s, --skip-build    Skip Docker image building
    -h, --help          Show this help message

EXAMPLES:
    $0 --environment development
    $0 --environment production --namespace mlops-fl-prod
    $0 --dry-run --environment development

EOF
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if kustomize is installed
    if ! command -v kustomize &> /dev/null; then
        error "kustomize is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed (unless skipping build)
    if [ "$SKIP_BUILD" = false ] && ! command -v docker &> /dev/null; then
        error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

build_images() {
    if [ "$SKIP_BUILD" = true ]; then
        warn "Skipping Docker image build"
        return
    fi
    
    log "Building Docker images..."
    
    # Build FL Server
    log "Building FL Server image..."
    docker build -f Dockerfile.server -t fl-server:latest .
    
    # Build API Service
    log "Building API Service image..."
    docker build -f Dockerfile.api -t api-service:latest .
    
    # Build Dashboard
    log "Building Dashboard image..."
    docker build -f Dockerfile.dashboard -t dashboard:latest .
    
    # Build FL Client
    log "Building FL Client image..."
    docker build -f Dockerfile.client -t fl-client:latest .
    
    log "Docker images built successfully"
}

deploy_to_k8s() {
    local overlay_path="k8s/overlays/$ENVIRONMENT"
    
    if [ ! -d "$overlay_path" ]; then
        error "Environment overlay not found: $overlay_path"
        exit 1
    fi
    
    log "Deploying to Kubernetes environment: $ENVIRONMENT"
    
    if [ "$DRY_RUN" = true ]; then
        log "Performing dry run..."
        kustomize build "$overlay_path" | kubectl apply --dry-run=client -f -
    else
        log "Applying Kubernetes manifests..."
        kustomize build "$overlay_path" | kubectl apply -f -
        
        # Wait for deployments to be ready
        log "Waiting for deployments to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment --all -n "$(get_namespace)"
        
        # Show status
        log "Deployment status:"
        kubectl get pods -n "$(get_namespace)" -o wide
        kubectl get services -n "$(get_namespace)" -o wide
    fi
}

get_namespace() {
    if [ -n "$NAMESPACE" ]; then
        echo "$NAMESPACE"
    elif [ "$ENVIRONMENT" = "production" ]; then
        echo "mlops-fl-prod"
    else
        echo "mlops-fl-dev"
    fi
}

show_access_info() {
    local ns="$(get_namespace)"
    
    log "Deployment completed successfully!"
    echo
    log "Access Information:"
    echo "===================="
    
    # Get service information
    kubectl get services -n "$ns" -o custom-columns=NAME:.metadata.name,TYPE:.spec.type,EXTERNAL-IP:.status.loadBalancer.ingress[0].ip,PORT:.spec.ports[0].port
    
    echo
    log "To access services:"
    if [ "$ENVIRONMENT" = "development" ]; then
        echo "- API Service: http://localhost:30800"
        echo "- Dashboard: http://localhost:30851" 
        echo "- Grafana: http://localhost:30300 (admin/admin)"
    else
        echo "- Check LoadBalancer external IPs above"
    fi
    
    echo
    log "Useful commands:"
    echo "- View pods: kubectl get pods -n $ns"
    echo "- View logs: kubectl logs -f deployment/<deployment-name> -n $ns"
    echo "- Port forward: kubectl port-forward service/<service-name> <local-port>:<service-port> -n $ns"
    echo "- Delete deployment: kubectl delete namespace $ns"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -s|--skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "production" ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be 'development' or 'production'"
    exit 1
fi

# Main execution
log "Starting MLOps Federated Learning deployment..."
log "Environment: $ENVIRONMENT"
log "Namespace: $(get_namespace)"
log "Dry run: $DRY_RUN"

check_prerequisites
build_images
deploy_to_k8s

if [ "$DRY_RUN" = false ]; then
    show_access_info
fi

log "Deployment script completed!"