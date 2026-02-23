"""
R-EAFT Test Suite
=================

Automated tests to verify R-EAFT implementation correctness.
Run with: python -m pytest test_r_eaft.py -v

Or run directly: python test_r_eaft.py
"""

import torch
import numpy as np
import sys

# Import the R-EAFT loss
exec(open('r_eaft.py').read())


def test_instantiation():
    """Test that REAFTLoss can be instantiated with various parameters."""
    # Default parameters
    loss_fn = REAFTLoss()
    assert loss_fn.shock_threshold == 5.0
    assert loss_fn.alpha == 3.0

    # Custom parameters
    loss_fn = REAFTLoss(shock_threshold=3.0, alpha=2.0, eps=1e-8)
    assert loss_fn.shock_threshold == 3.0
    assert loss_fn.alpha == 2.0
    assert loss_fn.eps == 1e-8

    print("✓ test_instantiation passed")


def test_forward_pass():
    """Test basic forward pass."""
    loss_fn = REAFTLoss()

    batch_size, seq_len, vocab_size = 2, 10, 1000
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(logits, labels)

    assert loss.shape == torch.Size([])
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss >= 0

    print("✓ test_forward_pass passed")


def test_backward_pass():
    """Test gradient computation."""
    loss_fn = REAFTLoss()

    logits = torch.randn(2, 10, 1000, requires_grad=True)
    labels = torch.randint(0, 1000, (2, 10))

    loss = loss_fn(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()

    print("✓ test_backward_pass passed")


def test_ignore_tokens():
    """Test that -100 labels are properly ignored."""
    loss_fn = REAFTLoss()

    logits = torch.randn(1, 10, 1000)
    labels = torch.randint(0, 1000, (1, 10))

    # Compute loss with all tokens
    loss_all = loss_fn(logits, labels.clone())

    # Ignore first 5 tokens
    labels_partial = labels.clone()
    labels_partial[:, :5] = -100
    loss_partial = loss_fn(logits, labels_partial)

    # Losses should be different
    assert loss_all != loss_partial

    print("✓ test_ignore_tokens passed")


def test_shocked_vs_protected():
    """Test that shocked tokens get amplified and protected tokens get dampened."""
    loss_fn = REAFTLoss(shock_threshold=5.0, alpha=3.0)

    vocab_size = 1000

    # Create confident-wrong scenario (should be SHOCKED)
    logits_wrong = torch.zeros(1, 10, vocab_size)
    logits_wrong[0, 4, 500] = 20.0  # Confident on token 500
    labels_wrong = torch.full((1, 10), -100)
    labels_wrong[0, 5] = 0  # But target is token 0

    diag_wrong = loss_fn.forward_with_diagnostics(logits_wrong, labels_wrong)

    # Create confident-correct scenario (should be PROTECTED)
    logits_correct = torch.zeros(1, 10, vocab_size)
    logits_correct[0, 4, 0] = 20.0  # Confident on token 0
    labels_correct = torch.full((1, 10), -100)
    labels_correct[0, 5] = 0  # Target IS token 0

    diag_correct = loss_fn.forward_with_diagnostics(logits_correct, labels_correct)

    # Shocked token should have weight >> 1
    assert diag_wrong['n_shocked'] == 1, f"Expected 1 shocked, got {diag_wrong['n_shocked']}"
    assert diag_wrong['mean_weight'] > 10, f"Shocked weight should be >10, got {diag_wrong['mean_weight']}"

    # Protected token should have weight << 1
    assert diag_correct['n_protected'] == 1, f"Expected 1 protected, got {diag_correct['n_protected']}"
    assert diag_correct['mean_weight'] < 0.01, f"Protected weight should be <0.01, got {diag_correct['mean_weight']}"

    print("✓ test_shocked_vs_protected passed")


def test_gradient_amplification():
    """Test that shocked tokens produce larger gradients than protected tokens."""
    loss_fn = REAFTLoss(shock_threshold=5.0, alpha=3.0)

    vocab_size = 1000

    # Shocked scenario
    logits_shocked = torch.zeros(1, 10, vocab_size, requires_grad=True)
    with torch.no_grad():
        logits_shocked.data[0, 4, 500] = 20.0
    labels = torch.full((1, 10), -100)
    labels[0, 5] = 0

    loss_shocked = loss_fn(logits_shocked, labels)
    loss_shocked.backward()
    grad_shocked = logits_shocked.grad.abs().mean().item()

    # Protected scenario
    logits_protected = torch.zeros(1, 10, vocab_size, requires_grad=True)
    with torch.no_grad():
        logits_protected.data[0, 4, 0] = 20.0

    loss_protected = loss_fn(logits_protected, labels)
    loss_protected.backward()
    grad_protected = logits_protected.grad.abs().mean().item()

    # Shocked gradients should be much larger
    assert grad_shocked > grad_protected * 100 or grad_protected < 1e-10, \
        f"Shocked grad ({grad_shocked}) should be >> protected grad ({grad_protected})"

    print("✓ test_gradient_amplification passed")


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    loss_fn = REAFTLoss()

    # Test with very small logits (uniform distribution)
    logits_uniform = torch.zeros(1, 10, 1000)
    labels = torch.randint(0, 1000, (1, 10))
    loss = loss_fn(logits_uniform, labels)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Test with very large logits
    logits_large = torch.zeros(1, 10, 1000)
    logits_large[:, :, 0] = 100.0
    loss = loss_fn(logits_large, labels)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    print("✓ test_numerical_stability passed")


def test_batch_consistency():
    """Test that batched computation is consistent with individual samples."""
    loss_fn = REAFTLoss()

    torch.manual_seed(42)

    # Create two different samples
    logits1 = torch.randn(1, 10, 1000)
    logits2 = torch.randn(1, 10, 1000)
    labels1 = torch.randint(0, 1000, (1, 10))
    labels2 = torch.randint(0, 1000, (1, 10))

    # Compute individually
    loss1 = loss_fn(logits1, labels1)
    loss2 = loss_fn(logits2, labels2)

    # Compute batched
    logits_batch = torch.cat([logits1, logits2], dim=0)
    labels_batch = torch.cat([labels1, labels2], dim=0)
    loss_batch = loss_fn(logits_batch, labels_batch)

    # Batched loss should be average of individual losses (approximately)
    expected = (loss1 + loss2) / 2
    assert abs(loss_batch.item() - expected.item()) < 0.1, \
        f"Batch loss {loss_batch.item()} != avg individual {expected.item()}"

    print("✓ test_batch_consistency passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("R-EAFT Test Suite")
    print("=" * 60)

    tests = [
        test_instantiation,
        test_forward_pass,
        test_backward_pass,
        test_ignore_tokens,
        test_shocked_vs_protected,
        test_gradient_amplification,
        test_numerical_stability,
        test_batch_consistency,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} tests failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
