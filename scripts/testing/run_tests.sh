#!/bin/bash
# Run E2E tests with CUDA 11 compatibility fix

# Set project root (go up two levels from scripts/testing)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"

# Set CUDA compatibility library path
export LD_LIBRARY_PATH="${PROJECT_ROOT}/.cuda_compat:/usr/local/cuda-12.1/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

# Set Python path
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "============================================================"
echo "AVATAR E2E TEST SUITE"
echo "============================================================"
echo ""
echo "🔧 Environment:"
echo "   CUDA Compat: ${PROJECT_ROOT}/.cuda_compat"
echo "   Python Path: ${PROJECT_ROOT}/src"
echo ""

# Run quick service tests
echo "📋 Step 1: Quick Service Tests"
echo "------------------------------------------------------------"
poetry run python tests/quick_service_test.py
quick_result=$?

if [ $quick_result -ne 0 ]; then
    echo ""
    echo "⚠️  Quick tests failed. Fix issues before running E2E tests."
    exit $quick_result
fi

echo ""
echo "📋 Step 2: Full E2E Pipeline Tests"
echo "------------------------------------------------------------"
poetry run python tests/e2e/e2e_pipeline_test.py
e2e_result=$?

echo ""
echo "============================================================"
echo "TEST SUITE COMPLETE"
echo "============================================================"
echo ""
echo "Quick Tests: $([ $quick_result -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "E2E Tests:   $([ $e2e_result -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo ""

exit $e2e_result
