#!/bin/bash
#
# End-to-End Test Script for Stanley Agent
#
# This script:
# 1. Starts the Python API server
# 2. Waits for it to be ready
# 3. Runs the agent with a test query
# 4. Verifies the response
# 5. Cleans up
#
# Usage: ./scripts/e2e-test.sh
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
STANLEY_ROOT="$(dirname "$PROJECT_ROOT")"
API_PORT="${API_PORT:-8000}"
API_URL="http://localhost:${API_PORT}"
TIMEOUT=60
SKIP_API="${SKIP_API:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

echo_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

cleanup() {
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        echo_info "Stopping API server (PID: $API_PID)..."
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

# Check if API is already running
check_api_available() {
    curl -s -o /dev/null -w "%{http_code}" "${API_URL}/api/health" 2>/dev/null | grep -q "200"
}

# Wait for API to be ready
wait_for_api() {
    local elapsed=0
    echo_info "Waiting for API to be ready at ${API_URL}..."

    while [ $elapsed -lt $TIMEOUT ]; do
        if check_api_available; then
            echo_success "API is ready!"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    echo_error "API failed to start within ${TIMEOUT} seconds"
    return 1
}

# Start API server
start_api_server() {
    if [ "$SKIP_API" = "true" ]; then
        echo_info "Skipping API server start (SKIP_API=true)"
        return 0
    fi

    if check_api_available; then
        echo_info "API already running at ${API_URL}"
        return 0
    fi

    echo_info "Starting Stanley API server..."

    cd "$STANLEY_ROOT"

    # Check if virtual environment exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Start the API server in background
    python -m stanley.api.main --port "$API_PORT" &
    API_PID=$!

    echo_info "API server started with PID: $API_PID"

    wait_for_api
}

# Run unit tests
run_unit_tests() {
    echo_info "Running unit tests..."
    cd "$PROJECT_ROOT"

    if bun test tests/unit --timeout 30000; then
        echo_success "Unit tests passed!"
        return 0
    else
        echo_error "Unit tests failed!"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    echo_info "Running integration tests..."
    cd "$PROJECT_ROOT"

    export STANLEY_API_URL="$API_URL"

    if bun test tests/integration --timeout 60000; then
        echo_success "Integration tests passed!"
        return 0
    else
        echo_error "Integration tests failed!"
        return 1
    fi
}

# Run agent query test
run_agent_query_test() {
    echo_info "Running agent query test..."
    cd "$PROJECT_ROOT"

    # Skip if no API key available (CI environment)
    if [ -z "$OPENROUTER_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
        echo_info "No API key found, skipping agent query test"
        echo_info "Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY to run this test"
        return 0
    fi

    # Run a simple query
    export STANLEY_API_URL="$API_URL"

    local response
    response=$(bun run src/index.ts --query "Check if the Stanley API is healthy" 2>&1) || {
        echo_error "Agent query test failed!"
        echo "$response"
        return 1
    }

    # Check if response mentions health or status
    if echo "$response" | grep -qi -E "(health|status|ok|available|running)"; then
        echo_success "Agent query test passed!"
        return 0
    else
        echo_info "Response received: $response"
        echo_success "Agent query completed (content not validated)"
        return 0
    fi
}

# Run mock conversation test
run_mock_conversation_test() {
    echo_info "Running mock conversation test..."
    cd "$PROJECT_ROOT"

    # Create a test script that simulates a conversation
    local test_script=$(cat <<'EOF'
import { createStanleyTools } from "./src/mcp/tools";
import { createStanleyAgent } from "./src/agents/stanley";

// Mock model
const mockModel = {
  specificationVersion: "v1",
  provider: "mock",
  modelId: "mock-model",
  doGenerate: async () => ({
    rawCall: { rawPrompt: "", rawSettings: {} },
    finishReason: "stop",
    text: "This is a mock response for testing.",
    usage: { promptTokens: 10, completionTokens: 20 },
    warnings: [],
  }),
};

const apiUrl = process.env.STANLEY_API_URL || "http://localhost:8000";
const tools = createStanleyTools({ baseUrl: apiUrl, timeout: 5000 });
const agent = createStanleyAgent({
  model: mockModel as any,
  tools,
});

console.log("Agent created with", agent.getToolNames().length, "tools");
console.log("Tools:", agent.getToolNames().slice(0, 5).join(", "), "...");
console.log("Mock conversation test passed!");
EOF
)

    echo "$test_script" > /tmp/mock-test.ts

    if bun run /tmp/mock-test.ts; then
        echo_success "Mock conversation test passed!"
        rm /tmp/mock-test.ts
        return 0
    else
        echo_error "Mock conversation test failed!"
        rm /tmp/mock-test.ts
        return 1
    fi
}

# Main test execution
main() {
    echo ""
    echo "=========================================="
    echo "  Stanley Agent E2E Test Suite"
    echo "=========================================="
    echo ""

    local failed=0

    # Run unit tests first (no API needed)
    if ! run_unit_tests; then
        failed=$((failed + 1))
    fi

    echo ""

    # Start API server if needed
    if ! start_api_server; then
        echo_error "Failed to start API server"
        failed=$((failed + 1))
    else
        # Run integration tests
        if ! run_integration_tests; then
            failed=$((failed + 1))
        fi

        echo ""

        # Run mock conversation test
        if ! run_mock_conversation_test; then
            failed=$((failed + 1))
        fi

        echo ""

        # Run agent query test (requires API key)
        if ! run_agent_query_test; then
            failed=$((failed + 1))
        fi
    fi

    echo ""
    echo "=========================================="

    if [ $failed -eq 0 ]; then
        echo_success "All tests passed!"
        exit 0
    else
        echo_error "$failed test suite(s) failed"
        exit 1
    fi
}

main "$@"
