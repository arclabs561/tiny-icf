"""Universal Frequency Estimator for word commonality prediction.

This package provides models, losses, evaluation, and utilities for predicting
Inverse Collection Frequency (ICF) from character-level features.
"""

__version__ = "0.1.0"

# Core models
from tiny_icf.model import UniversalICF
from tiny_icf.model_residual import ResidualICF
from tiny_icf.nano_model import NanoICF

# Loss functions (consolidated)
from tiny_icf.loss import (
    huber_loss,
    ranking_loss,
    neural_ndcg_loss_simple,
    lambdarank_loss,
    approx_ndcg_loss,
    SpearmanLoss,
    CombinedLoss,
)

# Adaptive loss weighting (optional)
try:
    from tiny_icf.loss_adaptive import (
        RealTimeNormalizedLoss,
        UncertaintyWeightedLoss,
        compute_gradient_norms,
        monitor_loss_components,
    )
    HAS_ADAPTIVE_LOSS = True
except ImportError:
    HAS_ADAPTIVE_LOSS = False

# Loss monitoring utilities
try:
    from tiny_icf.loss_monitoring import (
        compute_loss_component_metrics,
        detect_loss_imbalance,
        compute_gradient_balance,
        log_loss_components,
    )
    HAS_LOSS_MONITORING = True
except ImportError:
    HAS_LOSS_MONITORING = False

# Data loading
from tiny_icf.data import (
    WordICFDataset,
    load_frequency_list,
    compute_normalized_icf,
)

# Evaluation (consolidated)
from tiny_icf.eval import (
    compute_metrics,
    evaluate_on_dataset,
    expected_calibration_error,
    stratified_evaluation,
    evaluate_by_rarity_category,
    # RBO metrics (if available)
)

# Prediction
from tiny_icf.predict import (
    word_to_bytes,
    predict_icf,
)

# Training utilities
from tiny_icf.training_utils import (
    train_epoch_unified,
    validate_unified,
    save_checkpoint,
    load_checkpoint,
    create_optimizer,
    create_scheduler,
)

# Initialization
from tiny_icf.initialization import (
    init_model_weights,
)

# Baselines
from tiny_icf.baselines import (
    character_unigram_baseline,
    character_bigram_baseline,
    word_length_baseline,
    evaluate_baselines,
)

__all__ = [
    # Models
    "UniversalICF",
    "ResidualICF",
    "NanoICF",
    # Losses
    "huber_loss",
    "ranking_loss",
    "neural_ndcg_loss_simple",
    "lambdarank_loss",
    "approx_ndcg_loss",
    "SpearmanLoss",
    "CombinedLoss",
    # Adaptive losses (if available)
    *(["RealTimeNormalizedLoss", "UncertaintyWeightedLoss", "compute_gradient_norms", "monitor_loss_components"] if HAS_ADAPTIVE_LOSS else []),
    # Loss monitoring (if available)
    *(["compute_loss_component_metrics", "detect_loss_imbalance", "compute_gradient_balance", "log_loss_components"] if HAS_LOSS_MONITORING else []),
    # Data
    "WordICFDataset",
    "load_frequency_list",
    "compute_normalized_icf",
    # Evaluation
    "compute_metrics",
    "evaluate_on_dataset",
    "expected_calibration_error",
    "stratified_evaluation",
    "evaluate_by_rarity_category",
    # RBO metrics (if available)
    # Prediction
    "word_to_bytes",
    "predict_icf",
    # Training
    "train_epoch_unified",
    "validate_unified",
    "save_checkpoint",
    "load_checkpoint",
    "create_optimizer",
    "create_scheduler",
    # Initialization
    "init_model_weights",
    # Baselines
    "character_unigram_baseline",
    "character_bigram_baseline",
    "word_length_baseline",
    "evaluate_baselines",
]
