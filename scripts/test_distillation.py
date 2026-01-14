#!/usr/bin/env -S uv run
"""Test script for knowledge distillation integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from tiny_icf.model import UniversalICF
from tiny_icf.distillation import (
    LanguageModelTeacher,
    DistillationLoss,
    DistilledICFModel,
)


def test_teacher_model():
    """Test that teacher model can be created and used."""
    print("=" * 70)
    print("Test 1: Teacher Model Creation")
    print("=" * 70)
    
    try:
        teacher = LanguageModelTeacher(
            model_name="all-MiniLM-L6-v2",
            model_type="sentence-transformers",
            device=torch.device("cpu"),
        )
        print(f"✅ Teacher model created: {teacher.model_name}")
        print(f"   Embedding dimension: {teacher.embedding_dim}")
        
        # Test embedding extraction
        words = ["the", "quick", "brown", "fox"]
        embeddings = teacher.get_embeddings(words)
        print(f"✅ Embeddings extracted: {embeddings.shape}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Import error: {e}")
        print("   Install with: uv pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distillation_loss():
    """Test distillation loss computation."""
    print("\n" + "=" * 70)
    print("Test 2: Distillation Loss")
    print("=" * 70)
    
    batch_size = 8
    
    # Create dummy predictions
    student_predictions = torch.rand(batch_size, 1) * 0.5 + 0.25  # ICF in [0.25, 0.75]
    teacher_predictions = torch.rand(batch_size, 1) * 0.5 + 0.25
    ground_truth = torch.rand(batch_size, 1) * 0.5 + 0.25
    
    # Create loss function
    distillation_loss = DistillationLoss(
        temperature=3.0,
        alpha=0.5,
        beta=0.1,
        use_feature_distillation=False,
    )
    
    # Compute loss
    total_loss, components = distillation_loss(
        student_predictions=student_predictions,
        teacher_predictions=teacher_predictions,
        ground_truth=ground_truth,
    )
    
    print(f"✅ Distillation loss computed")
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Supervised loss: {components['supervised_loss'].item():.4f}")
    print(f"   Distillation loss: {components['distillation_loss'].item():.4f}")
    print(f"   Feature loss: {components['feature_loss'].item():.4f}")
    
    return True


def test_distilled_model():
    """Test distilled model wrapper."""
    print("\n" + "=" * 70)
    print("Test 3: Distilled Model Wrapper")
    print("=" * 70)
    
    try:
        # Create student model
        student = UniversalICF()
        print(f"✅ Student model created: {sum(p.numel() for p in student.parameters())} params")
        
        # Create teacher model
        teacher = LanguageModelTeacher(
            model_name="all-MiniLM-L6-v2",
            model_type="sentence-transformers",
            device=torch.device("cpu"),
        )
        print(f"✅ Teacher model created")
        
        # Wrap with distillation
        distilled_model = DistilledICFModel(
            student_model=student,
            teacher_model=teacher,
        )
        print(f"✅ Distilled model wrapper created")
        
        # Test forward pass
        batch_size = 4
        max_length = 20
        byte_tensors = torch.randint(0, 256, (batch_size, max_length))
        words = ["the", "quick", "brown", "fox"]
        
        outputs = distilled_model(
            byte_tensors=byte_tensors,
            words=words,
            return_features=False,
        )
        
        print(f"✅ Forward pass successful")
        print(f"   Student predictions: {outputs['student_predictions'].shape}")
        print(f"   Teacher predictions: {outputs['teacher_predictions'].shape}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Import error: {e}")
        print("   Install with: uv pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """Test end-to-end distillation training step."""
    print("\n" + "=" * 70)
    print("Test 4: End-to-End Distillation Step")
    print("=" * 70)
    
    try:
        # Create models
        student = UniversalICF()
        teacher = LanguageModelTeacher(
            model_name="all-MiniLM-L6-v2",
            model_type="sentence-transformers",
            device=torch.device("cpu"),
        )
        distilled_model = DistilledICFModel(
            student_model=student,
            teacher_model=teacher,
        )
        
        # Create loss
        distillation_loss = DistillationLoss(
            temperature=3.0,
            alpha=0.5,
            beta=0.0,  # No feature distillation for simplicity
        )
        
        # Create dummy batch
        batch_size = 4
        max_length = 20
        byte_tensors = torch.randint(0, 256, (batch_size, max_length))
        icf_targets = torch.rand(batch_size, 1) * 0.5 + 0.25
        words = ["the", "quick", "brown", "fox"]
        
        # Forward pass
        outputs = distilled_model(
            byte_tensors=byte_tensors,
            words=words,
            return_features=False,
        )
        
        # Compute loss
        loss, components = distillation_loss(
            student_predictions=outputs['student_predictions'],
            teacher_predictions=outputs['teacher_predictions'],
            ground_truth=icf_targets,
        )
        
        # Backward pass
        loss.backward()
        
        print(f"✅ End-to-end distillation step successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Gradients computed: {any(p.grad is not None for p in student.parameters())}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Testing Knowledge Distillation Integration")
    print("=" * 70)
    
    results = []
    
    # Test 1: Teacher model
    results.append(("Teacher Model", test_teacher_model()))
    
    # Test 2: Distillation loss
    results.append(("Distillation Loss", test_distillation_loss()))
    
    # Test 3: Distilled model wrapper
    results.append(("Distilled Model", test_distilled_model()))
    
    # Test 4: End-to-end
    results.append(("End-to-End", test_end_to_end()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    print(f"\n{'✅ All tests passed!' if all_passed else '⚠️  Some tests failed or skipped'}")
    
    if not all_passed:
        print("\n💡 Note: Some tests require sentence-transformers:")
        print("   Install with: uv pip install sentence-transformers")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

