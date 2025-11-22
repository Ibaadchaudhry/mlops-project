# Kubernetes Deployment Guide

## MLOps Federated Learning Kubernetes Manifests

This directory contains Kubernetes deployment manifests for the MLOps Federated Learning system, replacing Docker Compose with a production-ready orchestration solution.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FL Clients    │    │   FL Server     │    │   API Service   │
│   (Jobs/CronJobs│────┤   (Deployment)  │    │   (Deployment)  │
│                 │    │                 │    │   + HPA         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │   Dashboard     │    │   Prometheus    │    │    Grafana      │
         │   (Deployment)  │    │   (Deployment)  │    │   (Deployment)  │
         │                 │    │                 │    │                 │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
                                         │
                 ┌─────────────────────────────────────────┐
                 │         Persistent Storage              │
                 │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│
                 │  │Models│ │Data │ │Drift│ │Prom │ │Graf ││
                 │  │ PVC │ │ PVC │ │ PVC │ │ PVC │ │ PVC ││
                 │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘│
                 └─────────────────────────────────────────┘
```

## Directory Structure

```
k8s/
├── base/                          # Base Kubernetes manifests
│   ├── 00-namespace.yaml          # Namespace, ConfigMap, Secrets
│   ├── 01-storage.yaml            # PersistentVolumeClaims
│   ├── 02-fl-server.yaml          # FL Server Deployment & Service
│   ├── 03-api-service.yaml        # API Service Deployment, Service & HPA
│   ├── 04-dashboard.yaml          # Dashboard Deployment, Service & Ingress
│   ├── 05-prometheus.yaml         # Prometheus Deployment & Service + RBAC
│   ├── 06-grafana.yaml            # Grafana Deployment & Service
│   ├── 07-fl-clients.yaml         # FL Client Jobs & CronJobs
│   └── kustomization.yaml         # Base Kustomization
└── overlays/                      # Environment-specific overlays
    ├── development/               # Development environment
    │   ├── kustomization.yaml     # Dev Kustomization
    │   ├── namespace-patch.yaml   # Dev namespace
    │   ├── storage-patch.yaml     # Smaller storage sizes
    │   └── service-patch.yaml     # NodePort services
    └── production/                # Production environment
        ├── kustomization.yaml     # Prod Kustomization
        ├── namespace-patch.yaml   # Prod namespace
        ├── storage-patch.yaml     # Larger storage + backup
        ├── security-patch.yaml    # NetworkPolicy & Security
        └── resource-patch.yaml    # Higher resource limits
```

## Quick Start

### Prerequisites

1. **Kubernetes cluster** (local or cloud)
2. **kubectl** configured to connect to your cluster
3. **kustomize** (can be installed standalone or use `kubectl -k`)
4. **Docker** (for building images)

### Deployment Options

#### Option 1: Using deployment scripts

**Linux/Mac:**
```bash
# Development deployment
./deploy-k8s.sh --environment development

# Production deployment
./deploy-k8s.sh --environment production

# Dry run
./deploy-k8s.sh --dry-run --environment development
```

**Windows:**
```cmd
REM Development deployment
deploy-k8s.bat --environment development

REM Production deployment
deploy-k8s.bat --environment production

REM Dry run
deploy-k8s.bat --dry-run --environment development
```

#### Option 2: Manual deployment with Kustomize

**Development:**
```bash
# Build and preview manifests
kustomize build k8s/overlays/development

# Apply to cluster
kustomize build k8s/overlays/development | kubectl apply -f -

# Wait for deployments
kubectl wait --for=condition=available --timeout=300s deployment --all -n mlops-fl-dev
```

**Production:**
```bash
# Build and preview manifests
kustomize build k8s/overlays/production

# Apply to cluster
kustomize build k8s/overlays/production | kubectl apply -f -

# Wait for deployments
kubectl wait --for=condition=available --timeout=300s deployment --all -n mlops-fl-prod
```

#### Option 3: Using kubectl with kustomize

```bash
# Development
kubectl apply -k k8s/overlays/development

# Production
kubectl apply -k k8s/overlays/production
```

## Service Access

### Development Environment

Services are exposed via NodePort for easy access:

- **API Service**: `http://localhost:30800`
- **Dashboard**: `http://localhost:30851`
- **Grafana**: `http://localhost:30300` (admin/admin)
- **Prometheus**: Port-forward required (`kubectl port-forward service/prometheus-service 9090:9090 -n mlops-fl-dev`)

### Production Environment

Services use LoadBalancer (requires cloud provider or MetalLB):

```bash
# Check external IPs
kubectl get services -n mlops-fl-prod

# If LoadBalancer not available, use port-forwarding
kubectl port-forward service/api-service 8000:8000 -n mlops-fl-prod
kubectl port-forward service/dashboard-service 8501:8501 -n mlops-fl-prod
kubectl port-forward service/grafana-service 3000:3000 -n mlops-fl-prod
```

## Key Features

### 1. **Multi-Environment Support**
- **Development**: Smaller resources, NodePort services, debug settings
- **Production**: HA setup, LoadBalancers, security policies, larger storage

### 2. **Scalability**
- **API Service**: Horizontal Pod Autoscaler (2-10 replicas based on CPU/memory)
- **FL Server**: Can be scaled for high availability
- **Dashboard**: Multiple replicas for production

### 3. **Storage Management**
- **Shared PVCs**: Models, data, and drift reports shared across pods
- **Monitoring Storage**: Dedicated PVCs for Prometheus and Grafana
- **Environment-specific**: Different sizes for dev/prod

### 4. **Security (Production)**
- **NetworkPolicy**: Restricts pod-to-pod communication
- **ServiceAccount**: Limited permissions for security
- **Secrets**: Encrypted storage for sensitive data
- **Non-root containers**: Security hardening

### 5. **Monitoring & Observability**
- **Prometheus**: Metrics collection with service discovery
- **Grafana**: Dashboards with pre-configured datasources
- **Health checks**: Liveness and readiness probes
- **Resource monitoring**: CPU, memory, and custom metrics

## Federated Learning Workflow

### 1. **Server Startup**
```bash
# FL Server starts and waits for clients
kubectl logs -f deployment/fl-server -n mlops-fl-dev
```

### 2. **Manual Client Execution**
```bash
# Run FL clients manually
kubectl create job --from=job/fl-client-job manual-training-$(date +%s) -n mlops-fl-dev

# Monitor client progress
kubectl get jobs -n mlops-fl-dev
kubectl logs job/manual-training-xxxxx -n mlops-fl-dev
```

### 3. **Scheduled Training (CronJob)**
```bash
# Weekly training runs automatically
kubectl get cronjobs -n mlops-fl-dev

# Manually trigger scheduled job
kubectl create job --from=cronjob/fl-client-cronjob manual-cronjob-run -n mlops-fl-dev
```

### 4. **Model Serving via API**
```bash
# Test API endpoints
curl http://localhost:30800/health
curl http://localhost:30800/predict -X POST -H "Content-Type: application/json" -d '{"features": [...]}'
```

## Storage Configuration

### Persistent Volume Claims

| PVC Name | Size (Dev) | Size (Prod) | Purpose |
|----------|------------|-------------|----------|
| `models-pvc` | 2Gi | 20Gi | ML models storage |
| `data-pvc` | 5Gi | 50Gi | Dataset storage |
| `drift-reports-pvc` | 1Gi | 10Gi | Drift detection reports |
| `prometheus-pvc` | 5Gi | 50Gi | Prometheus metrics |
| `grafana-pvc` | 1Gi | 5Gi | Grafana dashboards |

### Storage Classes

Modify `storageClassName` in storage manifests based on your cluster:

- **Local/Development**: `standard`, `hostpath`
- **AWS**: `gp2`, `gp3`, `efs` (for ReadWriteMany)
- **GCP**: `standard`, `ssd`, `filestore` (for ReadWriteMany)
- **Azure**: `default`, `managed-premium`, `azurefile` (for ReadWriteMany)

## Troubleshooting

### Common Issues

1. **PVC Pending**
   ```bash
   kubectl get pvc -n mlops-fl-dev
   kubectl describe pvc models-pvc -n mlops-fl-dev
   ```
   - Check if storage class exists
   - Verify cluster has dynamic provisioning

2. **Pods Failing**
   ```bash
   kubectl get pods -n mlops-fl-dev
   kubectl describe pod <pod-name> -n mlops-fl-dev
   kubectl logs <pod-name> -n mlops-fl-dev
   ```

3. **Service Not Accessible**
   ```bash
   kubectl get services -n mlops-fl-dev
   kubectl port-forward service/<service-name> <local-port>:<service-port> -n mlops-fl-dev
   ```

4. **Image Pull Errors**
   ```bash
   # Check image exists
   docker images | grep fl-server
   
   # For production, push to registry
   docker tag fl-server:latest your-registry.com/fl-server:v1.0.0
   docker push your-registry.com/fl-server:v1.0.0
   ```

### Useful Commands

```bash
# View all resources
kubectl get all -n mlops-fl-dev

# Check resource usage
kubectl top pods -n mlops-fl-dev
kubectl top nodes

# Check events
kubectl get events -n mlops-fl-dev --sort-by='.lastTimestamp'

# Scale deployment
kubectl scale deployment/api-service --replicas=5 -n mlops-fl-dev

# Update image
kubectl set image deployment/fl-server fl-server=fl-server:v2.0.0 -n mlops-fl-dev

# Rollback deployment
kubectl rollout undo deployment/fl-server -n mlops-fl-dev

# Delete everything
kubectl delete namespace mlops-fl-dev
```

## Customization

### Environment Variables

Modify `fl-config` ConfigMap in `00-namespace.yaml`:

```yaml
data:
  FL_ROUNDS: "10"           # Number of training rounds
  FL_MIN_CLIENTS: "3"       # Minimum clients required
  LOG_LEVEL: "INFO"         # Logging level
  # Add custom variables here
```

### Resource Limits

Adjust resource requests/limits in deployment manifests or patches:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

### Adding New Services

1. Create new deployment manifest in `base/`
2. Add to `base/kustomization.yaml` resources
3. Create environment-specific patches in overlays
4. Update deployment scripts if needed

## Production Considerations

### 1. **Image Registry**
- Push images to private registry
- Update image references in production kustomization
- Configure image pull secrets if needed

### 2. **Ingress Controller**
- Install NGINX, Traefik, or cloud provider ingress
- Configure SSL/TLS certificates
- Update ingress annotations for your controller

### 3. **Backup Strategy**
- Configure backup for PVCs
- Export Grafana dashboards
- Backup ConfigMaps and Secrets

### 4. **Monitoring & Alerting**
- Configure Prometheus alerts
- Set up alerting channels (Slack, PagerDuty)
- Monitor resource usage and set limits

### 5. **Security Hardening**
- Enable Pod Security Standards
- Use network policies
- Regular security scans
- Rotate secrets

This Kubernetes setup provides a robust, scalable foundation for your MLOps federated learning system with proper environment management and production-ready features.