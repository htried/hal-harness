#!/bin/bash
# Test libnoisy with different error modes

MODE=${1:-connection_refused}

echo "Testing with error mode: $MODE"
echo ""

hal-eval \
  --benchmark gaia \
  --agent_dir agents/test_libnoisy_agent/ \
  --agent_function main.run \
  --agent_name "LibNoisy Test ($MODE)" \
  --docker \
  --max_tasks 1 \
  --crash-test \
  --noisy-mode both \
  --failure-rate 1.0 \
  --error-mode "$MODE" \
  --crash-test-debug

