# Make It Executable:
# Run the following command to give it execute permissions:
# chmod +x create_structure.sh
# Execute the Script:
# Run the script by typing:
# ./create_structure.sh


#!/bin/bash
# Create top-level files
mkdir -p project
cd project || exit

touch main.py pyproject.toml README.md Makefile .gitignore .pre-commit-config.yaml .editorconfig

# Create agent directory and files
mkdir -p agent
touch agent/__init__.py \
      agent/bot.py \
      agent/meta_parser.py \
      agent/plan_agent.py \
      agent/action_agent.py \
      agent/critic_agent.py

# Create API layer directories and files
mkdir -p api/routes/graphql/resolvers
mkdir -p api/middleware
mkdir -p api/grpc/protos api/grpc/services
mkdir -p api/websockets

touch api/__init__.py \
      api/routes/__init__.py \
      api/routes/agent_routes.py \
      api/routes/health_routes.py \
      api/routes/graphql/__init__.py \
      api/routes/graphql/schema.py \
      api/routes/graphql/resolvers/__init__.py \
      api/middleware/__init__.py \
      api/middleware/auth.py \
      api/middleware/logging.py \
      api/grpc/__init__.py \
      api/grpc/services/__init__.py \
      api/websockets/__init__.py \
      api/websockets/handlers.py

# Create infrastructure directories and files
mkdir -p infrastructure/terraform/environments
mkdir -p infrastructure/ansible
mkdir -p infrastructure/pulumi

touch infrastructure/terraform/main.tf \
      infrastructure/terraform/variables.tf \
      infrastructure/terraform/environments/staging.tfvars \
      infrastructure/terraform/environments/production.tfvars \
      infrastructure/ansible/site.yml \
      infrastructure/pulumi/Pulumi.yaml

# Create monitoring directories and files
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/elasticsearch/templates

touch monitoring/prometheus/prometheus.yml \
      monitoring/grafana/dashboards/default_dashboard.json \
      monitoring/elasticsearch/templates/default_template.json

# Create security directories
mkdir -p security/audit security/certs security/policies security/scanners

# Create ML models directories
mkdir -p ml_models/training ml_models/inference ml_models/pipelines

# Create DB directories
mkdir -p db/migrations db/seeds db/schemas

# Create cache directories and files
mkdir -p cache/redis cache/memcached
touch cache/redis/redis.conf \
      cache/memcached/memcached.conf

# Create queues directories and files
mkdir -p queues/kafka queues/rabbitmq queues/redis_streams
touch queues/kafka/kafka.conf \
      queues/rabbitmq/rabbitmq.conf \
      queues/redis_streams/redis_streams.conf

# Create localization directories and files
mkdir -p localization/translations localization/templates
touch localization/translations/en.json \
      localization/templates/locale_template.json

# Create benchmarks directories and files
mkdir -p benchmarks/load_tests benchmarks/stress_tests
touch benchmarks/load_tests/load_test.py \
      benchmarks/stress_tests/stress_test.py

# Create docs directories and files
mkdir -p docs/architecture/diagrams docs/architecture/decisions
mkdir -p docs/api/openapi docs/api/postman
mkdir -p docs/runbooks
touch docs/architecture/diagrams/system_architecture.png \
      docs/architecture/decisions/design_decisions.md \
      docs/api/openapi/openapi.yaml \
      docs/api/postman/collection.json \
      docs/runbooks/deployment_runbook.md

# Create scripts directories and files
mkdir -p scripts/dev scripts/deploy scripts/maintenance
touch scripts/dev/start_dev.sh \
      scripts/deploy/deploy.sh \
      scripts/maintenance/backup.sh

# Create tools directories and files
mkdir -p tools/linters tools/formatters tools/generators
touch tools/linters/pylint.rc \
      tools/formatters/black.toml \
      tools/generators/scaffold_generator.py

# Create tests directories and files
mkdir -p tests/unit tests/integration tests/e2e
touch tests/__init__.py \
      tests/unit/__init__.py \
      tests/unit/test_bot.py \
      tests/unit/test_integrations.py
mkdir -p tests/integration
touch tests/integration/__init__.py \
      tests/integration/test_api.py
mkdir -p tests/e2e
touch tests/e2e/__init__.py \
      tests/e2e/test_workflows.py

# Create GitHub workflows
mkdir -p .github/workflows
touch .github/workflows/ci.yml \
      .github/workflows/cd-staging.yml \
      .github/workflows/cd-production.yml

# Create Kubernetes directories and files
mkdir -p k8s/base k8s/staging/patches k8s/production/patches
touch k8s/base/deployment.yaml \
      k8s/base/service.yaml \
      k8s/base/configmap.yaml \
      k8s/base/secrets.yaml \
      k8s/base/kustomization.yaml
touch k8s/staging/kustomization.yaml \
      k8s/staging/patches/deployment-patch.yaml
touch k8s/production/kustomization.yaml \
      k8s/production/patches/deployment-patch.yaml

# Create docker directories and files
mkdir -p docker
touch docker/Dockerfile \
      docker/Dockerfile.dev \
      docker/docker-compose.yml

echo "Directory structure created successfully."
