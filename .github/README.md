# 🚀 CI/CD Pipeline Documentation

This directory contains GitHub Actions workflows that implement a comprehensive CI/CD pipeline for the Federated Learning MLOps system.

## 📁 Workflow Files

### 🔧 Core CI/CD Pipeline
- **`ci-cd-pipeline.yml`** - Main CI/CD workflow with build, test, and deployment stages
- **`testing.yml`** - Comprehensive automated testing suite
- **`monitoring.yml`** - System monitoring and alerting automation
- **`model-training.yml`** - Automated model training and retraining
- **`deployment.yml`** - Production deployment with multiple strategies

## 🔄 Workflow Overview

### 1. **CI/CD Pipeline** (`ci-cd-pipeline.yml`)
**Triggers:** Push to main/develop, Pull Requests, Manual dispatch

**Stages:**
- 🧪 **Code Quality & Testing** - Formatting, linting, type checking, unit tests
- 🔗 **Integration Tests** - Data ingestion, model creation, drift detection
- 🐳 **Build & Push Images** - Multi-service Docker image builds
- 🔒 **Security Scanning** - Vulnerability scanning with Trivy
- 🚀 **Deploy to Staging** - Automated staging deployment
- 🎯 **Deploy to Production** - Manual production deployment with approvals

### 2. **Automated Testing** (`testing.yml`)
**Triggers:** Push, Pull Requests, Daily schedule

**Test Types:**
- 🧩 **Unit Tests** - Cross-platform (Ubuntu, Windows, macOS) and Python versions (3.8-3.10)
- 🔗 **Integration Tests** - API integration, FL client-server communication
- 🌍 **End-to-End Tests** - Full system testing with Docker
- ⚡ **Performance Tests** - Model inference speed, training benchmarks
- 🔒 **Security Tests** - Dependency scanning, code analysis, secret detection

### 3. **Model Training** (`model-training.yml`)
**Triggers:** Manual dispatch, Weekly schedule, Repository dispatch (drift detection)

**Features:**
- 📊 **Training Condition Checks** - Automated decision making for retraining
- 🤖 **Federated Learning Execution** - Multi-client training orchestration
- ✅ **Model Validation** - Performance validation across clients
- 📝 **Model Registry Updates** - Automatic versioning and metadata tracking
- 🚀 **Deployment Triggers** - Automatic deployment initiation

### 4. **System Monitoring** (`monitoring.yml`)
**Triggers:** Hourly schedule, Manual dispatch, External alerts

**Monitoring Areas:**
- 🏥 **System Health** - API, database, resource usage monitoring
- 📊 **Data Drift Detection** - Automated drift pattern analysis
- 📈 **Performance Monitoring** - Model performance trend analysis
- 🔒 **Security Monitoring** - Dependency vulnerabilities, secret detection
- 📱 **Alerting & Notifications** - Slack notifications, dashboard generation

### 5. **Deployment Pipeline** (`deployment.yml`)
**Triggers:** Model training completion, Manual dispatch

**Deployment Strategies:**
- 🔄 **Rolling Deployment** - Gradual instance updates
- 🔵🟢 **Blue-Green Deployment** - Zero-downtime environment switching
- 🐦 **Canary Deployment** - Gradual traffic shifting with monitoring

## 🛠️ Setup Instructions

### 1. **Repository Secrets**
Configure these secrets in your GitHub repository:

```bash
# Required for container registry
GITHUB_TOKEN  # Automatically provided

# Optional for notifications
SLACK_WEBHOOK_URL  # Slack webhook for notifications
ALERT_EMAIL        # Email for critical alerts
```

### 2. **Environment Configuration**
The workflows support multiple environments:

- **`staging`** - Automated deployments from main branch
- **`production`** - Manual deployments with approvals

Configure environment protection rules in GitHub:
- Production environment requires manual approval
- Staging environment allows automatic deployments

### 3. **Branch Protection**
Configure branch protection rules for `main` branch:

```yaml
Required status checks:
  - CI/CD Pipeline / Code Quality & Unit Tests
  - CI/CD Pipeline / Integration Tests
  - Automated Testing / Unit Tests
  - Automated Testing / Integration Tests

Require pull request reviews: 1
Dismiss stale reviews: true
Require review from CODEOWNERS: true
```

## 🚦 Workflow Triggers

### Automatic Triggers
- **Code Push/PR** → Testing + CI/CD Pipeline
- **Weekly Schedule** → Model Training + Full Testing
- **Hourly Schedule** → System Monitoring
- **High Drift Detection** → Automatic Model Retraining
- **Performance Degradation** → Alert + Optional Retraining

### Manual Triggers
- **Deploy Specific Environment** - Manual deployment to staging/production
- **Force Model Retraining** - Manual training trigger with custom parameters
- **Run Specific Monitoring** - Target specific monitoring checks
- **Emergency Deployment** - Fast-track deployment for hotfixes

## 📊 Monitoring & Alerting

### Automated Alerts
The system automatically triggers alerts for:

- 🚨 **Critical System Issues** - Service downtime, resource exhaustion
- ⚠️ **Data Drift** - Significant drift detection (>30% features affected)
- 📉 **Performance Degradation** - Model performance drops >5%
- 🔒 **Security Issues** - High-severity vulnerabilities detected

### Notification Channels
- **Slack** - Real-time notifications for all events
- **Email** - Critical alerts only
- **GitHub Issues** - Automatic issue creation for failures
- **Dashboard** - HTML monitoring dashboard generated hourly

## 🔧 Customization

### Adding New Tests
1. Create test files in appropriate directories (`tests/`, `tests/integration/`, etc.)
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures and markers for organization

### Adding New Monitoring
1. Add monitoring logic to `monitoring.yml`
2. Define alert thresholds and conditions
3. Update notification templates

### Custom Deployment Strategies
1. Add new strategy in `deployment.yml`
2. Implement strategy-specific logic
3. Update environment configurations

## 📚 Artifacts & Reports

Each workflow generates artifacts that are stored for analysis:

### CI/CD Pipeline
- **Training Artifacts** (30 days) - Models, drift reports, datasets
- **Validation Reports** (30 days) - Model performance validation
- **Security Reports** (7 days) - Vulnerability scan results

### Testing
- **Coverage Reports** - Code coverage analysis
- **Performance Benchmarks** - Performance test results
- **Security Scan Results** - Dependency and code security analysis

### Monitoring
- **Monitoring Reports** (7 days) - System health snapshots
- **Drift Analysis** - Data drift pattern analysis
- **Performance Trends** - Model performance over time

## 🚀 Getting Started

1. **Enable GitHub Actions** in your repository settings
2. **Configure secrets** as described above
3. **Set up environment protection rules**
4. **Create your first PR** to trigger the pipeline
5. **Monitor workflow execution** in the Actions tab

The CI/CD pipeline will automatically:
- ✅ Test your code changes
- 🐳 Build Docker images
- 🚀 Deploy to staging
- 📊 Monitor system health
- 🤖 Retrain models when needed

## 🔍 Troubleshooting

### Common Issues

1. **Test Failures**
   - Check test logs in workflow output
   - Ensure all dependencies are properly installed
   - Verify test data and mock configurations

2. **Build Failures**
   - Check Dockerfile syntax
   - Verify all required files are present
   - Check for resource limitations

3. **Deployment Issues**
   - Verify environment configurations
   - Check service health endpoints
   - Review deployment strategy settings

4. **Monitoring Alerts**
   - Check system resource usage
   - Verify data quality and drift patterns
   - Review model performance metrics

### Debug Mode
Enable debug logging by setting workflow inputs:
```yaml
env:
  DEBUG: true
  LOG_LEVEL: DEBUG
```

## 🤝 Contributing

When contributing to the CI/CD pipeline:

1. **Test locally** before pushing
2. **Update documentation** for new features
3. **Follow security best practices**
4. **Add appropriate tests** for new functionality
5. **Update monitoring** for new services

---

**📝 Note:** This CI/CD pipeline is designed for production use and includes enterprise-grade features like security scanning, performance monitoring, and automated rollbacks. Customize according to your specific requirements.