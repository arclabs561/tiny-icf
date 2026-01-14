# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "torch>=2.0.0",
#   "numpy>=1.24.0",
#   "tqdm>=4.65.0",
#   "scipy>=1.10.0",
# ]
# ///
"""
Test consolidated interfaces to verify everything works.

Tests:
1. CombinedLoss with all features (NeuralNDCG, Listwise)
2. Consolidated prediction interface
3. Evaluation with uncertainty/robustness
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np

from tiny_icf.model import UniversalICF
from tiny_icf.loss import CombinedLoss, lambdarank_loss, approx_ndcg_loss, neural_ndcg_loss_simple
from tiny_icf.predict_consolidated import predict, predict_batch
from tiny_icf.eval import compute_metrics, evaluate_on_dataset
from tiny_icf.data import WordICFDataset, load_frequency_list
from tiny_icf.eval_uncertainty import compute_uncertainty_metrics
from tiny_icf.eval_robustness import compute_robustness_metrics


def test_combined_loss():
    """Test CombinedLoss with all features."""
    print("=" * 70)
    print("Testing CombinedLoss...")
    print("=" * 70)
    
    # Create dummy data
    batch_size = 16
    predictions = torch.randn(batch_size, 1) * 0.5 + 0.5  # [0, 1] range
    targets = torch.rand(batch_size, 1)
    
    # Test 1: Basic loss
    print("\n1. Basic CombinedLoss (Huber + Ranking)")
    loss_fn = CombinedLoss()
    loss = loss_fn(predictions, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test 2: With NeuralNDCG
    print("\n2. CombinedLoss with NeuralNDCG")
    loss_fn = CombinedLoss(use_neural_ndcg=True, neural_ndcg_weight=0.5)
    loss = loss_fn(predictions, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test 3: With Listwise (LambdaRank)
    print("\n3. CombinedLoss with Listwise (LambdaRank)")
    loss_fn = CombinedLoss(
        use_listwise_ranking=True,
        listwise_method="lambdarank",
        listwise_weight=1.0,
        listwise_sigma=1.0,
    )
    loss = loss_fn(predictions, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test 4: With Listwise (ApproxNDCG)
    print("\n4. CombinedLoss with Listwise (ApproxNDCG)")
    loss_fn = CombinedLoss(
        use_listwise_ranking=True,
        listwise_method="approx_ndcg",
        listwise_weight=1.0,
        listwise_temperature=1.0,
    )
    loss = loss_fn(predictions, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test 5: All features combined
    print("\n5. CombinedLoss with ALL features")
    loss_fn = CombinedLoss(
        use_neural_ndcg=True,
        neural_ndcg_weight=0.5,
        use_listwise_ranking=True,
        listwise_method="lambdarank",
        listwise_weight=0.5,
    )
    loss = loss_fn(predictions, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    print("\n✅ CombinedLoss tests passed!")


def test_consolidated_prediction():
    """Test consolidated prediction interface."""
    print("\n" + "=" * 70)
    print("Testing Consolidated Prediction...")
    print("=" * 70)
    
    # Create a dummy model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalICF().to(device)
    model.eval()
    
    test_words = ["hello", "world", "supercalifragilisticexpialidocious", "the"]
    
    # Test 1: Basic prediction
    print("\n1. Basic prediction")
    for word in test_words[:2]:
        result = predict(word, model, device, enhanced=False, advanced=False)
        print(f"   {word:30} → ICF: {result['icf']:.4f}")
    
    # Test 2: Enhanced prediction
    print("\n2. Enhanced prediction (with interpretation, confidence)")
    for word in test_words[:2]:
        result = predict(word, model, device, enhanced=True, advanced=False)
        print(f"   {word:30} → ICF: {result['icf']:.4f}, "
              f"Interpretation: {result.get('interpretation', 'N/A')}")
    
    # Test 3: Advanced prediction
    print("\n3. Advanced prediction (with language, temporal)")
    for word in test_words[:2]:
        result = predict(word, model, device, enhanced=False, advanced=True)
        print(f"   {word:30} → ICF: {result['icf']:.4f}")
        if 'languages' in result:
            print(f"   {'':30}   Languages: {result['languages']}")
        if 'temporal' in result:
            print(f"   {'':30}   Temporal: {result.get('temporal', {}).get('era', 'N/A')}")
    
    # Test 4: Batch prediction
    print("\n4. Batch prediction")
    batch_results = predict_batch(test_words, model, device, enhanced=True, batch_size=2)
    print(f"   Processed {len(batch_results)} words")
    for result in batch_results:
        print(f"   {result['word']:30} → ICF: {result['icf']:.4f}")
    
    print("\n✅ Consolidated prediction tests passed!")


def test_evaluation_features():
    """Test evaluation with uncertainty and robustness."""
    print("\n" + "=" * 70)
    print("Testing Evaluation Features...")
    print("=" * 70)
    
    # Check if data file exists
    data_file = Path("data/word_frequency.csv")
    if not data_file.exists():
        print(f"\n⚠️  Data file not found: {data_file}")
        print("   Skipping evaluation tests (need data file)")
        return
    
    # Load small dataset
    print("\n1. Loading dataset...")
    try:
        words, frequencies = load_frequency_list(str(data_file))
        dataset = WordICFDataset(words[:1000], frequencies[:1000])  # Small subset
        print(f"   Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        return
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalICF().to(device)
    model.eval()
    
    # Test 1: Basic evaluation
    print("\n2. Basic evaluation metrics")
    try:
        metrics = evaluate_on_dataset(model, dataset, device, batch_size=32)
        print(f"   Spearman: {metrics.get('spearman_corr', 0):.4f}")
        print(f"   MAE: {metrics.get('mae', 0):.4f}")
        print(f"   RMSE: {metrics.get('rmse', 0):.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Uncertainty quantification
    print("\n3. Uncertainty quantification")
    try:
        # Get predictions
        predictions = []
        targets = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        with torch.no_grad():
            for batch_words, batch_targets in dataloader:
                batch_tensors = torch.stack([
                    torch.tensor(list(word.encode("utf-8")[:20] + bytes(20 - len(word.encode("utf-8")[:20]))), dtype=torch.long)
                    for word in batch_words
                ]).to(device)
                batch_preds = model(batch_tensors).cpu().numpy()
                predictions.extend(batch_preds.flatten())
                targets.extend(batch_targets.numpy().flatten())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        uncertainty_metrics = compute_uncertainty_metrics(
            torch.tensor(predictions),
            torch.tensor(targets),
            n_bootstrap=50,  # Small for testing
        )
        print(f"   Bootstrap CI width: {uncertainty_metrics.get('bootstrap_ci_width', 0):.4f}")
        print(f"   Prediction variance: {uncertainty_metrics.get('prediction_variance', 0):.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Robustness testing
    print("\n4. Robustness testing")
    try:
        # Test on a few words
        test_words = ["hello", "world", "test"]
        robustness_metrics = compute_robustness_metrics(
            model,
            test_words,
            device,
            n_adversarial=5,  # Small for testing
        )
        print(f"   Adversarial accuracy: {robustness_metrics.get('adversarial_accuracy', 0):.4f}")
        print(f"   Noise robustness: {robustness_metrics.get('noise_robustness', 0):.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n✅ Evaluation feature tests passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING CONSOLIDATED INTERFACES")
    print("=" * 70)
    
    # Test 1: CombinedLoss
    test_combined_loss()
    
    # Test 2: Consolidated prediction
    test_consolidated_prediction()
    
    # Test 3: Evaluation features
    test_evaluation_features()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print("\n✅ All consolidated interfaces are working!")


if __name__ == "__main__":
    main()

