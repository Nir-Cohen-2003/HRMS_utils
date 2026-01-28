"""
Quick test to verify threshold configuration is correct for both implementations.
"""

import sys
from pathlib import Path

# Add experiments directory to path
sys.path.insert(0, str(Path(__file__).parent))

from approximate_similarity import SimilarityConfig

print("Testing SimilarityConfig threshold behavior:")
print("=" * 60)

# Test 1: Default auto-reduction (approx_threshold < 0)
print("\nTest 1: Auto-reduction (approx_threshold=-1.0, default)")
cfg1 = SimilarityConfig(
    upper_mass_bound=1000.0,
    bin_size=0.0001,
    threshold=0.8,
    approx_threshold=-1.0,  # Trigger auto-reduction
)
print(f"  threshold: {cfg1.threshold}")
print(f"  approx_threshold: {cfg1.approx_threshold}")
print(f"  Expected: 0.65 (0.8 - 0.15)")
assert cfg1.approx_threshold == 0.65, "Auto-reduction failed!"
print("  ✓ PASS")

# Test 2: Explicit approx_threshold (no auto-reduction)
print("\nTest 2: Explicit approx_threshold=0.65 (no auto-reduction)")
cfg2 = SimilarityConfig(
    upper_mass_bound=1000.0,
    bin_size=0.0001,
    threshold=0.8,
    approx_threshold=0.65,  # Explicitly set
)
print(f"  threshold: {cfg2.threshold}")
print(f"  approx_threshold: {cfg2.approx_threshold}")
print(f"  Expected: 0.65 (explicitly set)")
assert cfg2.approx_threshold == 0.65, "Explicit setting failed!"
print("  ✓ PASS")

# Test 3: Verify both configs are equivalent
print("\nTest 3: Verify both approaches yield same approx_threshold")
assert cfg1.approx_threshold == cfg2.approx_threshold, "Mismatch!"
print(f"  Both have approx_threshold={cfg1.approx_threshold}")
print("  ✓ PASS")

print("\n" + "=" * 60)
print("All threshold tests passed! ✓")
print("\nConclusion: To ensure both old and new implementations use")
print("the same threshold, set approx_threshold=0.65 explicitly.")
