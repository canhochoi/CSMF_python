#!/bin/bash
# Run CSMF GPU tests with proper Python interpreter

cd /home/lqluan/CSMF_python
PYTHON=/home/lqluan/CSMF_python/.venv/bin/python

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         CSMF GPU Testing Suite - Complete Validation          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Using Python: $PYTHON"
echo "Interpreting PyTorch availability..."
$PYTHON -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "⚠ PyTorch not available (GPU fallback to CPU)"
echo ""

# Test 1: Main CSMF Validation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: GPU CSMF Validation (30 outer × 50 inner iterations)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON test_gpu_csmf.py
TEST1_RESULT=$?
echo ""

# Test 2: Convergence Analysis
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Convergence Analysis (1→50 outer iterations)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON test_gpu_convergence.py
TEST2_RESULT=$?
echo ""

# Test 3: Reference Verification
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Reference Scatter Plot Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$PYTHON verify_scatter_plots.py
TEST3_RESULT=$?
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                        TEST SUMMARY                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ $TEST1_RESULT -eq 0 ]; then
    echo "✅ TEST 1 PASSED - GPU CSMF validation successful"
else
    echo "❌ TEST 1 FAILED - GPU CSMF validation failed"
fi

if [ $TEST2_RESULT -eq 0 ]; then
    echo "✅ TEST 2 PASSED - Convergence analysis complete"
else
    echo "❌ TEST 2 FAILED - Convergence analysis failed"
fi

if [ $TEST3_RESULT -eq 0 ]; then
    echo "✅ TEST 3 PASSED - Reference verification successful"
else
    echo "❌ TEST 3 FAILED - Reference verification failed"
fi

echo ""
echo "Generated Outputs:"
echo "  • test_gpu_csmf_scatter_w_c.png"
echo "  • test_gpu_csmf_scatter_w_s0.png"
echo "  • verify_scatter_plots.png (from Test 3)"
echo ""

if [ $TEST1_RESULT -eq 0 ] && [ $TEST2_RESULT -eq 0 ] && [ $TEST3_RESULT -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ ALL TESTS PASSED - GPU CSMF Implementation Validated ✅  ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  ⚠️  Some tests failed - see output above for details         ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    exit 1
fi
