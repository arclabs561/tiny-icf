"""Enhanced PyTorch Lightning module for flexible training with all features."""

from typing import Dict, Any

import torch
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR

from tiny_icf.loss import CombinedLoss
from tiny_icf.model import UniversalICF
from tiny_icf.model_residual import ResidualICF
from tiny_icf.training_utils import generate_ranking_pairs
from tiny_icf.eval import compute_metrics

# Optional unified multi-task loss
try:
    from tiny_icf.loss_unified import UnifiedMultiTaskLoss

    HAS_UNIFIED_LOSS = True
except ImportError:
    HAS_UNIFIED_LOSS = False

# Optional research-aligned loss
try:
    from tiny_icf.loss_research_aligned import ResearchAlignedICFLoss

    HAS_RESEARCH_ALIGNED_LOSS = True
except ImportError:
    HAS_RESEARCH_ALIGNED_LOSS = False

# Optional distillation support
try:
    from tiny_icf.distillation import (
        LanguageModelTeacher,
        DistillationLoss,
        DistilledICFModel,
    )

    HAS_DISTILLATION = True
except ImportError:
    HAS_DISTILLATION = False

# Optional imports for enhanced evaluation
try:
    from tiny_icf.eval_ranking_metrics import compute_ranking_metrics

    HAS_RANKING_METRICS = True
except ImportError:
    HAS_RANKING_METRICS = False

try:
    from tiny_icf.eval_confidence import compute_metrics_with_ci

    HAS_CONFIDENCE_INTERVALS = True
except ImportError:
    HAS_CONFIDENCE_INTERVALS = False


class FlexibleIDFLightningModule(LightningModule):
    """Lightning module with all flexible training features."""

    def __init__(
        self,
        config: Dict[str, Any],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        # Store for Aim metadata logging (set by training script)
        self._aim_config = None
        self._aim_logger = None

        # Create model based on config
        # Support multi-task model if enabled
        if config.get("use_multi_task_model", False):
            try:
                from tiny_icf.model_multi_task import MultiTaskICF

                output_tasks = config.get("output_tasks", ["icf"])
                self.model = MultiTaskICF(
                    base_model=None,  # Create new base
                    output_tasks=output_tasks,
                    num_languages=int(config.get("num_languages", 10)),
                    num_eras=int(config.get("num_eras", 5)),
                    num_hygiene=int(config.get("num_hygiene", 8)),
                    temporal_decades=config.get("temporal_decades", (1800, 1900, 2000)),
                    base_model_kwargs=config.get("base_model_kwargs", None),
                )
                self.use_multi_task_model = True
            except ImportError:
                print("⚠️  MultiTaskICF not available, falling back to single-task model")
                if config.get("model_type") == "residual":
                    self.model = ResidualICF()
                else:
                    self.model = UniversalICF()
                self.use_multi_task_model = False
        else:
            # Support transformer models from HuggingFace
            if config.get("model_type") == "transformer":
                try:
                    from tiny_icf.model_transformer import TransformerICF

                    model_name = config.get("transformer_model_name", "distilbert-base-uncased")
                    pooling = config.get("transformer_pooling", "mean")
                    freeze_backbone = config.get("freeze_transformer_backbone", False)
                    self.model = TransformerICF(
                        model_name=model_name,
                        pooling=pooling,
                        hidden_dim=config.get("hidden_dim", 128),
                        dropout=config.get("dropout", 0.3),
                        freeze_backbone=freeze_backbone,
                        use_pretrained=config.get("use_pretrained_transformer", True),
                    )
                    print(f"✅ Using TransformerICF with {model_name}")
                except ImportError as e:
                    print(f"⚠️  TransformerICF not available: {e}, falling back to UniversalICF")
                    self.model = UniversalICF()
            elif config.get("model_type") == "char_transformer":
                try:
                    from tiny_icf.model_transformer import CharacterLevelTransformerICF

                    model_name = config.get("transformer_model_name", "google/byt5-small")
                    freeze_backbone = config.get("freeze_transformer_backbone", False)
                    self.model = CharacterLevelTransformerICF(
                        model_name=model_name,
                        hidden_dim=config.get("hidden_dim", 128),
                        dropout=config.get("dropout", 0.3),
                        freeze_backbone=freeze_backbone,
                    )
                    print(f"✅ Using CharacterLevelTransformerICF with {model_name}")
                except ImportError as e:
                    print(
                        f"⚠️  CharacterLevelTransformerICF not available: {e}, falling back to UniversalICF"
                    )
                    self.model = UniversalICF()
            elif config.get("model_type") == "residual":
                self.model = ResidualICF()
            else:
                # Support attention mechanism in UniversalICF
                self.model = UniversalICF(
                    use_attention=config.get("use_attention", False),
                    attention_heads=config.get("attention_heads", 4),
                )
            self.use_multi_task_model = False

        # Create loss function - support research-aligned, unified multi-task loss, or legacy CombinedLoss
        use_research_aligned = (
            config.get("use_research_aligned_loss", False) and HAS_RESEARCH_ALIGNED_LOSS
        )
        use_unified_loss = (
            config.get("use_unified_loss", False) and HAS_UNIFIED_LOSS and not use_research_aligned
        )

        if use_research_aligned:
            # Use research-aligned loss (incorporates adaptive regularization, focal loss, monotonicity, etc.)
            self.criterion = ResearchAlignedICFLoss(
                huber_delta=config.get("huber_delta", 0.1),
                asymmetry_factor=config.get("asymmetry_factor", 2.0),
                rank_margin=config.get("rank_margin", 0.1),
                rank_weight=config.get("rank_weight", 0.5),
                ranking_method=config.get(
                    "ranking_method", "sigmoid"
                ),  # "sigmoid", "neural_sort", "probabilistic", "smooth_i"
                adaptive_reg=config.get("adaptive_reg", True),
                base_reg_strength=config.get("base_reg_strength", 1.0),
                use_focal=config.get("use_focal", True),
                focal_gamma=config.get("focal_gamma", 2.0),
                use_monotonicity=config.get("use_monotonicity", False),
                monotonicity_weight=config.get("monotonicity_weight", 0.1),
                monotonicity_constraints=config.get("monotonicity_constraints", None),
                use_quantile=config.get("use_quantile", False),
                quantile=config.get("quantile", 0.5),
                quantile_weight=config.get("quantile_weight", 0.3),
                use_spearman=config.get("use_spearman", True),
                spearman_weight=config.get("spearman_weight", 10.0),
            )
            self.use_research_aligned_loss = True
            self.use_unified_loss = False
        elif use_unified_loss:
            # Use unified multi-task loss (all tasks with rank-relax and AMOO)
            self.criterion = UnifiedMultiTaskLoss(
                icf_weight=config.get("icf_weight", 1.0),
                text_reduction_weight=config.get("text_reduction_weight", 0.5),
                temporal_weight=config.get("temporal_weight", 0.3),
                language_weight=config.get("language_weight", 0.2),
                era_weight=config.get("era_weight", 0.2),
                hygiene_weight=config.get("hygiene_weight", 0.2),
                use_amoo=config.get("use_amoo", True),
                amoo_curvature_weight=config.get("amoo_curvature_weight", 0.1),
                icf_spearman_weight=config.get("spearman_weight", 10.0),
                icf_spearman_reg_strength=config.get("spearman_reg_strength", 1.0),
                ranking_reg_strength=config.get("ranking_reg_strength", 1.0),
            )
            self.use_research_aligned_loss = False
            self.use_unified_loss = True
        else:
            # Legacy CombinedLoss (ICF-only, backward compatible)
            self.criterion = CombinedLoss(
                use_neural_ndcg=config.get("use_neural_ndcg", False),
                neural_ndcg_weight=config.get("neural_ndcg_weight", 0.5),
                use_listwise_ranking=config.get("use_listwise_ranking", False),
                listwise_method=config.get("listwise_method", "lambdarank"),
                listwise_weight=config.get("listwise_weight", 1.0),
                rank_weight=config.get(
                    "rank_weight", 0.01
                ),  # Use config value (default 0.01 for magnitude normalization)
                use_spearman=config.get("use_spearman", False),
                spearman_weight=config.get("spearman_weight", 10.0),
                spearman_reg_strength=config.get("spearman_reg_strength", 1.0),
                track_components=True,
            )
            self.use_research_aligned_loss = False
            self.use_unified_loss = False

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_pairs = config.get("n_pairs", 16)
        self.min_diff = config.get("min_diff", 0.05)
        self.use_weighted_sampling = config.get("use_weighted_sampling", True)
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)

        # Distillation support (optional)
        self.use_distillation = config.get("use_distillation", False) and HAS_DISTILLATION
        self.teacher_model = None
        self.distillation_loss_fn = None
        self.distilled_model_wrapper = None

        if self.use_distillation:
            try:
                teacher_model_name = config.get("teacher_model_name", "all-MiniLM-L6-v2")
                teacher_model_type = config.get("teacher_model_type", "sentence-transformers")

                # Device will be set later when model is moved to device
                device = torch.device("cpu")  # Will be updated when model moves to GPU

                self.teacher_model = LanguageModelTeacher(
                    model_name=teacher_model_name,
                    model_type=teacher_model_type,
                    device=device,
                    use_word_frequency_head=config.get("teacher_use_icf_head", False),
                )

                self.distillation_loss_fn = DistillationLoss(
                    temperature=config.get("distillation_temperature", 3.0),
                    alpha=config.get("distillation_alpha", 0.5),
                    beta=config.get("distillation_beta", 0.1),
                    use_feature_distillation=config.get("use_feature_distillation", False),
                    feature_projection_dim=config.get("feature_projection_dim", None),
                    use_dynamic_temperature=config.get(
                        "use_dynamic_temperature", True
                    ),  # Enable by default
                    base_temperature=config.get("distillation_temperature", 3.0),
                    min_temperature=config.get("min_temperature", 2.0),
                    max_temperature=config.get("max_temperature", 10.0),
                )

                # Wrap student model with distillation
                self.distilled_model_wrapper = DistilledICFModel(
                    student_model=self.model,
                    teacher_model=self.teacher_model,
                )

                print(f"✅ Distillation enabled: teacher={teacher_model_name}")
            except Exception as e:
                print(f"⚠️  Failed to initialize distillation: {e}")
                print("   Continuing without distillation")
                self.use_distillation = False

        # For validation metrics
        self.validation_predictions: list = []
        self.validation_targets: list = []
        self.validation_words: list = []  # Store words for diagnostic analysis

    def forward(self, x, return_all: bool = False):
        """Forward pass with optional multi-task support."""
        if self.use_multi_task_model:
            return self.model(x, return_all=return_all)
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        # Support multiple batch formats:
        # 1. Dict with 'words' (distillation)
        # 2. Dict with 'icf_targets' (multi-task)
        # 3. Tuple (legacy single-task)
        # 4. Tuple with 3 elements (legacy with words)

        words = None
        if isinstance(batch, dict):
            if "words" in batch:
                # Distillation batch
                byte_tensors = batch["byte_tensors"]
                icf_targets = batch["icf_targets"]
                words = batch["words"]
            elif "icf_targets" in batch:
                # Multi-task batch
                byte_tensors = batch["byte_tensors"]
                icf_targets = batch["icf_targets"]
            else:
                # Fallback: assume legacy format
                byte_tensors, icf_targets = batch
        else:
            # Legacy tuple format
            if len(batch) == 3:
                # Tuple with words: (byte_tensors, icf_targets, words)
                byte_tensors, icf_targets, words = batch
            else:
                # Standard tuple: (byte_tensors, icf_targets)
                byte_tensors, icf_targets = batch

        # Get predictions (multi-task or single-task)
        if self.use_multi_task_model:
            model_outputs = self(byte_tensors, return_all=True)
            predictions = model_outputs.get("icf", self(byte_tensors))  # Fallback to ICF
        else:
            predictions = self(byte_tensors)

        # Distillation: get teacher predictions if enabled
        teacher_predictions = None
        teacher_features = None
        student_features = None

        if self.use_distillation and words is not None and len(words) > 0 and words[0]:
            # Get teacher predictions and features
            with torch.no_grad():
                distilled_outputs = self.distilled_model_wrapper(
                    byte_tensors=byte_tensors,
                    words=words,
                    return_features=self.distillation_loss_fn.use_feature_distillation,
                )
                teacher_predictions = distilled_outputs.get("teacher_predictions")
                if self.distillation_loss_fn.use_feature_distillation:
                    teacher_features = distilled_outputs.get("teacher_features")
                    # Extract student features from model
                    try:
                        _, features_dict = self.model(byte_tensors, return_features=True)
                        student_features = features_dict.get("feature_activations")
                    except (TypeError, AttributeError):
                        pass

        # Use distillation loss if enabled and teacher predictions available
        if self.use_distillation and teacher_predictions is not None:
            # Distillation loss combines supervised + distillation + feature alignment
            loss, loss_components = self.distillation_loss_fn(
                student_predictions=predictions,
                teacher_predictions=teacher_predictions,
                ground_truth=icf_targets,
                student_features=student_features,
                teacher_features=teacher_features,
            )

            # Log distillation loss components
            self.log(
                "train_loss_supervised",
                loss_components["supervised_loss"],
                on_step=True,
                on_epoch=True,
            )
            self.log(
                "train_loss_distillation",
                loss_components["distillation_loss"],
                on_step=True,
                on_epoch=True,
            )
            if loss_components["feature_loss"].item() > 0:
                self.log(
                    "train_loss_feature",
                    loss_components["feature_loss"],
                    on_step=True,
                    on_epoch=True,
                )
            # Log temperature if dynamic
            if "temperature" in loss_components:
                self.log(
                    "train_temperature", loss_components["temperature"], on_step=True, on_epoch=True
                )

        elif self.use_unified_loss:
            # Unified multi-task loss
            icf_mask = batch.get("icf_mask") if isinstance(batch, dict) else None
            if icf_mask is not None:
                icf_mask = icf_mask.to(torch.bool).view(-1)
                icf_idx = torch.where(icf_mask)[0]
            else:
                icf_idx = torch.arange(len(icf_targets), device=icf_targets.device)

            # Subset ICF-supervised rows (clean tokens).
            icf_predictions = predictions[icf_idx] if len(icf_idx) > 0 else None
            icf_targets_sub = icf_targets[icf_idx] if len(icf_idx) > 0 else None

            # Ranking pairs only within the supervised subset.
            pairs = None
            if icf_targets_sub is not None and len(icf_idx) >= 2:
                n_pairs = min(len(icf_targets_sub), self.n_pairs)
                pairs_rel, _pair_target_diffs = generate_ranking_pairs(
                    icf_targets_sub.squeeze(1) if icf_targets_sub.dim() > 1 else icf_targets_sub,
                    n_pairs=n_pairs,
                    min_diff=self.min_diff,
                    use_weighted_sampling=self.use_weighted_sampling,
                )
                if pairs_rel is not None and len(pairs_rel) > 0:
                    pairs = pairs_rel.to(predictions.device)

            # Aux logits / targets.
            language_logits = (
                model_outputs.get("language")
                if self.use_multi_task_model and isinstance(model_outputs, dict)
                else (batch.get("language_logits") if isinstance(batch, dict) else None)
            )
            language_targets = batch.get("language_targets") if isinstance(batch, dict) else None
            era_logits = (
                model_outputs.get("era")
                if self.use_multi_task_model and isinstance(model_outputs, dict)
                else (batch.get("era_logits") if isinstance(batch, dict) else None)
            )
            era_targets = batch.get("era_targets") if isinstance(batch, dict) else None
            hygiene_logits = (
                model_outputs.get("hygiene")
                if self.use_multi_task_model and isinstance(model_outputs, dict)
                else (batch.get("hygiene_logits") if isinstance(batch, dict) else None)
            )
            hygiene_targets = batch.get("hygiene_targets") if isinstance(batch, dict) else None

            # Subset language/era supervision to the ICF-supervised rows by default.
            if language_logits is not None and language_targets is not None and len(icf_idx) > 0:
                language_logits = language_logits[icf_idx]
                language_targets = language_targets[icf_idx]
            else:
                language_logits = None
                language_targets = None

            if era_logits is not None and era_targets is not None and len(icf_idx) > 0:
                era_logits = era_logits[icf_idx]
                era_targets = era_targets[icf_idx]
            else:
                era_logits = None
                era_targets = None

            # Temporal: only compute when both predictions + targets exist and mask selects rows.
            historical_targets_full = (
                batch.get("historical_targets") if isinstance(batch, dict) else None
            )
            historical_mask = batch.get("historical_mask") if isinstance(batch, dict) else None
            historical_predictions = None
            historical_targets = None
            current_predictions = None
            current_targets = None

            if (
                self.use_multi_task_model
                and isinstance(model_outputs, dict)
                and model_outputs.get("temporal") is not None
                and isinstance(historical_targets_full, dict)
                and historical_mask is not None
            ):
                historical_mask = historical_mask.to(torch.bool).view(-1)
                temp_idx = torch.where(
                    historical_mask & (icf_mask if icf_mask is not None else True)
                )[0]
                temporal_pred = model_outputs["temporal"]
                decades = list(self.config.get("temporal_decades", (1800, 1900, 2000)))
                if (
                    len(temp_idx) > 0
                    and temporal_pred.dim() == 2
                    and temporal_pred.size(1) >= len(decades)
                ):
                    current_predictions = predictions[temp_idx]
                    current_targets = icf_targets[temp_idx]
                    historical_predictions = {
                        int(dec): temporal_pred[temp_idx, j].unsqueeze(1)
                        for j, dec in enumerate(decades)
                    }
                    historical_targets = {
                        int(dec): historical_targets_full[int(dec)][temp_idx]
                        for dec in decades
                        if int(dec) in historical_targets_full
                    }

            # Call unified loss with all task data (if available).
            loss_result = self.criterion(
                icf_predictions=icf_predictions,
                icf_targets=icf_targets_sub,
                icf_pairs=pairs,
                language_logits=language_logits,
                language_targets=language_targets,
                era_logits=era_logits,
                era_targets=era_targets,
                hygiene_logits=hygiene_logits,
                hygiene_targets=hygiene_targets,
                current_predictions=current_predictions,
                current_targets=current_targets,
                historical_predictions=historical_predictions,
                historical_targets=historical_targets,
                word_icf_scores=batch.get("word_icf_scores") if isinstance(batch, dict) else None,
                original_embedding=(
                    batch.get("original_embedding") if isinstance(batch, dict) else None
                ),
                reduced_embedding=(
                    batch.get("reduced_embedding") if isinstance(batch, dict) else None
                ),
                target_length=batch.get("target_length") if isinstance(batch, dict) else None,
            )

            # Unified loss returns (total_loss, diagnostics)
            if isinstance(loss_result, tuple):
                loss, diagnostics = loss_result
                # Log task-specific losses if available
                if isinstance(diagnostics, dict) and "task_losses" in diagnostics:
                    for task_name, task_loss in diagnostics["task_losses"].items():
                        self.log(f"train_loss_{task_name}", task_loss, on_step=True, on_epoch=True)
            else:
                loss = loss_result
        else:
            # Legacy CombinedLoss or ResearchAlignedICFLoss (ICF-only)
            # Generate ranking pairs using the vectorized function
            n_pairs = min(len(icf_targets), self.n_pairs)
            pairs, pair_target_diffs = generate_ranking_pairs(
                icf_targets.squeeze(1) if icf_targets.dim() > 1 else icf_targets,
                n_pairs=n_pairs,
                min_diff=self.min_diff,
                use_weighted_sampling=self.use_weighted_sampling,
            )

            if pairs is not None and len(pairs) > 0:
                pairs = pairs.to(predictions.device)
                pair_target_diffs = (
                    pair_target_diffs.to(predictions.device)
                    if pair_target_diffs is not None
                    else None
                )

            # Check if using ResearchAlignedICFLoss
            if self.use_research_aligned_loss:
                # ResearchAlignedICFLoss signature: (predictions, targets, pairs, features)
                features = None
                if self.config.get("use_monotonicity", False):
                    # Extract features for monotonicity constraints if needed
                    features = {}

                loss, loss_components = self.criterion(
                    predictions,
                    icf_targets,
                    pairs=pairs if pairs is not None and len(pairs) > 0 else None,
                    features=features,
                )

                # Log research-aligned loss components
                if isinstance(loss_components, dict):
                    for key, value in loss_components.items():
                        if isinstance(value, torch.Tensor):
                            val = value.item() if value.numel() == 1 else value.mean().item()
                            self.log(f"train_loss_{key}", val, on_step=True, on_epoch=True)
            else:
                # Legacy CombinedLoss
                loss = self.criterion(
                    predictions,
                    icf_targets,
                    pairs=pairs if pairs is not None and len(pairs) > 0 else None,
                    pair_target_diffs=pair_target_diffs,
                )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log loss components if available (for better monitoring)
        # Get component stats from the loss function
        if hasattr(self.criterion, "get_component_stats"):
            stats = self.criterion.get_component_stats()
            if stats:
                self.log("train_huber_ratio", stats.get("huber_ratio", 0), on_epoch=True)
                self.log("train_ranking_ratio", stats.get("ranking_ratio", 0), on_epoch=True)
                if self.criterion.use_spearman:
                    self.log("train_spearman_ratio", stats.get("spearman_ratio", 0), on_epoch=True)
                # Log raw component values for debugging
                self.log("train_huber_mean", stats.get("huber_mean", 0), on_epoch=True)
                self.log("train_ranking_mean", stats.get("ranking_mean", 0), on_epoch=True)
                if self.criterion.use_spearman:
                    self.log("train_spearman_mean", stats.get("spearman_mean", 0), on_epoch=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Apply gradient clipping and enhanced gradient flow analysis."""
        # Research: Gradient clipping prevents explosion in ranking models
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self.clip_grad_norm
            )
            # Log gradient norm for monitoring (research: gradient norms indicate convergence)
            self.log("grad_norm", grad_norm, on_step=True, on_epoch=True, prog_bar=False)

            # Enhanced: Layer-wise gradient analysis (for diagnostic purposes)
            # Log per-layer gradients periodically to detect vanishing/exploding gradients
            if self.global_step % 100 == 0:  # Sample every 100 steps to avoid log spam
                layer_grads = {}
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        layer_grad_norm = param.grad.norm().item()
                        layer_grads[name] = layer_grad_norm

                        # Log per-layer gradients (use sanitized name for TensorBoard)
                        sanitized_name = name.replace(".", "/")
                        self.log(f"grad_norm/{sanitized_name}", layer_grad_norm, on_step=True)

                # Detect vanishing/exploding gradients
                if len(layer_grads) > 0:
                    min_grad = min(layer_grads.values())
                    max_grad = max(layer_grads.values())

                    if min_grad < 1e-6:
                        self.log("grad_vanishing_warning", 1.0, on_step=True)
                    if max_grad > 100:
                        self.log("grad_exploding_warning", 1.0, on_step=True)

                    # Log gradient statistics
                    self.log("grad_norm_min", min_grad, on_step=True)
                    self.log("grad_norm_max", max_grad, on_step=True)

    def validation_step(self, batch, batch_idx):
        # Support both single-task (ICF-only) and multi-task batches
        words = None
        if isinstance(batch, dict) and "icf_targets" in batch:
            # Multi-task batch
            byte_tensors = batch["byte_tensors"]
            icf_targets = batch["icf_targets"]
            words = batch.get("words")  # Try to get words if available
        elif isinstance(batch, dict) and "words" in batch:
            # Distillation batch format
            byte_tensors = batch["byte_tensors"]
            icf_targets = batch["icf_targets"]
            words = batch["words"]
        else:
            # Legacy single-task batch (backward compatible)
            if isinstance(batch, tuple) and len(batch) == 3:
                byte_tensors, icf_targets, words = batch
            else:
                byte_tensors, icf_targets = batch

        # Get predictions (multi-task or single-task)
        if self.use_multi_task_model:
            model_outputs = self(byte_tensors, return_all=True)
            predictions = model_outputs.get("icf", self(byte_tensors))  # Fallback to ICF
        else:
            predictions = self(byte_tensors)

        if self.use_unified_loss:
            # Unified multi-task loss
            icf_mask = batch.get("icf_mask") if isinstance(batch, dict) else None
            if icf_mask is not None:
                icf_mask = icf_mask.to(torch.bool).view(-1)
                icf_idx = torch.where(icf_mask)[0]
            else:
                icf_idx = torch.arange(len(icf_targets), device=icf_targets.device)

            icf_predictions = predictions[icf_idx] if len(icf_idx) > 0 else None
            icf_targets_sub = icf_targets[icf_idx] if len(icf_idx) > 0 else None

            pairs = None
            if icf_targets_sub is not None and len(icf_idx) >= 2:
                n_pairs = min(len(icf_targets_sub), 16)
                pairs_rel, _pair_target_diffs = generate_ranking_pairs(
                    icf_targets_sub.squeeze(1) if icf_targets_sub.dim() > 1 else icf_targets_sub,
                    n_pairs=n_pairs,
                    min_diff=0.05,
                    use_weighted_sampling=False,
                )
                if pairs_rel is not None and len(pairs_rel) > 0:
                    pairs = pairs_rel.to(predictions.device)

            language_logits = (
                model_outputs.get("language")
                if self.use_multi_task_model and isinstance(model_outputs, dict)
                else (batch.get("language_logits") if isinstance(batch, dict) else None)
            )
            language_targets = batch.get("language_targets") if isinstance(batch, dict) else None
            era_logits = (
                model_outputs.get("era")
                if self.use_multi_task_model and isinstance(model_outputs, dict)
                else (batch.get("era_logits") if isinstance(batch, dict) else None)
            )
            era_targets = batch.get("era_targets") if isinstance(batch, dict) else None
            hygiene_logits = (
                model_outputs.get("hygiene")
                if self.use_multi_task_model and isinstance(model_outputs, dict)
                else (batch.get("hygiene_logits") if isinstance(batch, dict) else None)
            )
            hygiene_targets = batch.get("hygiene_targets") if isinstance(batch, dict) else None

            if language_logits is not None and language_targets is not None and len(icf_idx) > 0:
                language_logits = language_logits[icf_idx]
                language_targets = language_targets[icf_idx]
            else:
                language_logits = None
                language_targets = None

            if era_logits is not None and era_targets is not None and len(icf_idx) > 0:
                era_logits = era_logits[icf_idx]
                era_targets = era_targets[icf_idx]
            else:
                era_logits = None
                era_targets = None

            historical_targets_full = (
                batch.get("historical_targets") if isinstance(batch, dict) else None
            )
            historical_mask = batch.get("historical_mask") if isinstance(batch, dict) else None
            historical_predictions = None
            historical_targets = None
            current_predictions = None
            current_targets = None

            if (
                self.use_multi_task_model
                and isinstance(model_outputs, dict)
                and model_outputs.get("temporal") is not None
                and isinstance(historical_targets_full, dict)
                and historical_mask is not None
            ):
                historical_mask = historical_mask.to(torch.bool).view(-1)
                temp_idx = torch.where(
                    historical_mask & (icf_mask if icf_mask is not None else True)
                )[0]
                temporal_pred = model_outputs["temporal"]
                decades = list(self.config.get("temporal_decades", (1800, 1900, 2000)))
                if (
                    len(temp_idx) > 0
                    and temporal_pred.dim() == 2
                    and temporal_pred.size(1) >= len(decades)
                ):
                    current_predictions = predictions[temp_idx]
                    current_targets = icf_targets[temp_idx]
                    historical_predictions = {
                        int(dec): temporal_pred[temp_idx, j].unsqueeze(1)
                        for j, dec in enumerate(decades)
                    }
                    historical_targets = {
                        int(dec): historical_targets_full[int(dec)][temp_idx]
                        for dec in decades
                        if int(dec) in historical_targets_full
                    }

            loss_result = self.criterion(
                icf_predictions=icf_predictions,
                icf_targets=icf_targets_sub,
                icf_pairs=pairs,
                language_logits=language_logits,
                language_targets=language_targets,
                era_logits=era_logits,
                era_targets=era_targets,
                hygiene_logits=hygiene_logits,
                hygiene_targets=hygiene_targets,
                current_predictions=current_predictions,
                current_targets=current_targets,
                historical_predictions=historical_predictions,
                historical_targets=historical_targets,
                word_icf_scores=batch.get("word_icf_scores") if isinstance(batch, dict) else None,
                original_embedding=(
                    batch.get("original_embedding") if isinstance(batch, dict) else None
                ),
                reduced_embedding=(
                    batch.get("reduced_embedding") if isinstance(batch, dict) else None
                ),
                target_length=batch.get("target_length") if isinstance(batch, dict) else None,
            )

            if isinstance(loss_result, tuple):
                loss, diagnostics = loss_result
                if isinstance(diagnostics, dict) and "task_losses" in diagnostics:
                    for task_name, task_loss in diagnostics["task_losses"].items():
                        self.log(f"val_loss_{task_name}", task_loss, on_step=False, on_epoch=True)
            else:
                loss = loss_result
        else:
            # Legacy CombinedLoss (ICF-only)
            # Generate ranking pairs
            n_pairs = min(len(icf_targets), 16)
            pairs, pair_target_diffs = generate_ranking_pairs(
                icf_targets.squeeze(1) if icf_targets.dim() > 1 else icf_targets,
                n_pairs=n_pairs,
                min_diff=0.05,
                use_weighted_sampling=True,
            )

            if pairs is not None and len(pairs) > 0:
                pairs = pairs.to(predictions.device)
                pair_target_diffs = (
                    pair_target_diffs.to(predictions.device)
                    if pair_target_diffs is not None
                    else None
                )

            # Check if using ResearchAlignedICFLoss
            if self.use_research_aligned_loss:
                features = None
                if self.config.get("use_monotonicity", False):
                    features = {}

                loss, loss_components = self.criterion(
                    predictions,
                    icf_targets,
                    pairs=pairs if pairs is not None and len(pairs) > 0 else None,
                    features=features,
                )
            else:
                # Legacy CombinedLoss
                loss = self.criterion(
                    predictions,
                    icf_targets,
                    pairs=pairs if pairs is not None and len(pairs) > 0 else None,
                    pair_target_diffs=pair_target_diffs,
                )

        # Store for metrics computation
        self.validation_predictions.append(predictions.detach().cpu().numpy())
        self.validation_targets.append(icf_targets.detach().cpu().numpy())

        # Store words for diagnostic analysis
        if words is not None:
            if isinstance(words, list):
                self.validation_words.extend(words)
            elif isinstance(words, torch.Tensor):
                try:
                    # Convert tensor to list (assuming it's a batch of strings or indices)
                    if words.dtype == torch.long:
                        # If it's indices, we can't recover words, skip
                        pass
                    else:
                        word_list = words.tolist() if hasattr(words, "tolist") else list(words)
                        self.validation_words.extend(word_list)
                except Exception:
                    pass

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Compute validation metrics at end of epoch with enhanced evaluation."""
        if not self.validation_predictions:
            return

        import numpy as np

        pred_array = np.concatenate(self.validation_predictions).flatten()
        target_array = np.concatenate(self.validation_targets).flatten()

        # Convert to tensors for diagnostic analysis
        pred_tensor = torch.from_numpy(pred_array)
        target_tensor = torch.from_numpy(target_array)

        # Diagnostic analysis (if words available)
        words_for_diagnostics = None
        if self.validation_words and len(self.validation_words) == len(pred_array):
            words_for_diagnostics = self.validation_words

        # Compute standard metrics
        metrics = compute_metrics(pred_array, target_array)

        # Log standard metrics
        self.log("val_spearman_corr", metrics["spearman_corr"], prog_bar=True)
        self.log("val_mae", metrics["mae"], prog_bar=True)
        self.log("val_rmse", metrics["rmse"])

        # Enhanced: Log loss component breakdown if available
        # This addresses the issue: we optimize multiple loss components but only report Spearman
        if hasattr(self.criterion, "forward"):
            # Try to compute loss components on validation set
            try:
                # Create dummy pairs for ranking loss computation
                # (In practice, we'd want to use actual pairs from validation set)
                batch_size = min(64, len(pred_tensor))
                sample_indices = torch.randperm(len(pred_tensor))[:batch_size]
                sample_preds = pred_tensor[sample_indices]
                sample_targets = target_tensor[sample_indices]

                # Compute loss with components
                if isinstance(self.criterion, ResearchAlignedICFLoss):
                    # ResearchAlignedICFLoss returns (loss, components)
                    _, loss_components = self.criterion(
                        sample_preds.unsqueeze(1) if sample_preds.dim() == 1 else sample_preds,
                        (
                            sample_targets.unsqueeze(1)
                            if sample_targets.dim() == 1
                            else sample_targets
                        ),
                        pairs=None,  # Would need actual pairs for ranking loss
                        features=None,
                    )
                    # Log component breakdown
                    for component_name, component_value in loss_components.items():
                        if isinstance(component_value, torch.Tensor):
                            component_value = component_value.item()
                        self.log(f"val_loss_{component_name}", component_value, prog_bar=False)

                    # Log theoretical bound comparison for Spearman
                    theoretical_bound = 0.18  # From CEILING_ANALYSIS.md
                    spearman_ratio = (
                        metrics["spearman_corr"] / theoretical_bound
                        if theoretical_bound > 0
                        else 0.0
                    )
                    self.log("val_spearman_vs_theoretical", spearman_ratio, prog_bar=False)
                    self.log("val_theoretical_bound", theoretical_bound, prog_bar=False)

                    # Log component bounds comparison
                    # From LOSS_COMPONENT_BOUNDS.md (refined thresholds based on empirical data)
                    component_bounds = {
                        "huber": {
                            "good": 0.08,
                            "poor": 0.20,
                            "best": 0.05,
                        },  # Refined: top performers achieve 0.05-0.08
                        "rank": {
                            "good": 0.12,
                            "poor": 0.30,
                            "best": 0.05,
                        },  # Refined: top performers achieve 0.08-0.12
                        "spearman": {
                            "good": 0.82,
                            "poor": 0.90,
                            "best": 0.81,
                        },  # Refined: top performers achieve 0.81-0.82
                        "asymmetric_penalty": {"good": 0.05, "poor": 0.10, "best": 0.02},
                        "monotonicity": {"good": 0.01, "poor": 0.05, "best": 0.0},
                        "quantile": {"good": 0.20, "poor": 0.30, "best": 0.10},
                        # Multi-task bounds (from MULTI_TASK_BOUNDS.md)
                        "language_classification": {
                            "good": 0.7,
                            "poor": 1.0,
                            "best": 0.5,
                        },  # Cross-entropy loss
                        "language_ranking": {"good": 0.1, "poor": 0.2, "best": 0.0},
                        "era_classification": {
                            "good": 1.0,
                            "poor": 1.5,
                            "best": 0.5,
                        },  # Cross-entropy loss
                        "era_ranking": {"good": 0.1, "poor": 0.2, "best": 0.0},
                        "temporal_base": {"good": 0.10, "poor": 0.20, "best": 0.05},  # MSE loss
                        "temporal_consistency": {"good": 0.05, "poor": 0.10, "best": 0.0},
                        "temporal_ranking": {"good": 0.15, "poor": 0.30, "best": 0.05},
                        "text_reduction_regret": {
                            "good": 0.30,
                            "poor": 0.50,
                            "best": 0.15,
                        },  # Cosine distance
                        "text_reduction_path_regret": {
                            "good": 0.40,
                            "poor": 0.60,
                            "best": 0.20,
                        },  # Cumulative path regret
                        "text_reduction_ranking": {"good": 0.1, "poor": 0.2, "best": 0.0},
                    }

                    for component_name, component_value in loss_components.items():
                        if isinstance(component_value, torch.Tensor):
                            component_value = component_value.item()

                        # Compare to bounds
                        if component_name in component_bounds:
                            bounds = component_bounds[component_name]

                            # Determine status (best/good/acceptable/poor)
                            if "best" in bounds and component_value <= bounds["best"]:
                                status_code = 0.0  # best
                            elif component_value <= bounds["good"]:
                                status_code = 1.0  # good
                            elif component_value <= bounds["poor"]:
                                status_code = 2.0  # acceptable
                            else:
                                status_code = 3.0  # poor

                            # Log status (0.0=best, 1.0=good, 2.0=acceptable, 3.0=poor)
                            self.log(
                                f"val_loss_{component_name}_status", status_code, prog_bar=False
                            )

                            # Log ratio to good threshold
                            if bounds["good"] > 0:
                                ratio = component_value / bounds["good"]
                                self.log(
                                    f"val_loss_{component_name}_vs_good", ratio, prog_bar=False
                                )

                            # Log ratio to best threshold (if available)
                            if "best" in bounds and bounds["best"] > 0:
                                ratio_best = component_value / bounds["best"]
                                self.log(
                                    f"val_loss_{component_name}_vs_best", ratio_best, prog_bar=False
                                )

                    # Log component ratios for loss balance monitoring
                    # From LOSS_COMPONENT_BOUNDS.md: Component Ratios section
                    total_loss_val = loss_components.get("total")
                    if total_loss_val is not None:
                        if isinstance(total_loss_val, torch.Tensor):
                            total_loss_val = total_loss_val.item()

                        for comp_name, comp_val in loss_components.items():
                            if comp_name == "total" or comp_name == "reg_strength":
                                continue
                            if isinstance(comp_val, torch.Tensor):
                                comp_val = comp_val.item()

                            if total_loss_val > 0:
                                ratio = comp_val / total_loss_val
                                self.log(f"val_loss_{comp_name}_ratio", ratio, prog_bar=False)
            except Exception:
                # Silently fail if loss component extraction fails
                pass

        # Enhanced: Add ranking metrics (NDCG, MAP, MRR) using rank-eval
        if HAS_RANKING_METRICS:
            try:
                ranking_metrics = compute_ranking_metrics(
                    pred_array, target_array, k_values=[1, 3, 5, 10], use_graded=True
                )
                for metric_name, value in ranking_metrics.items():
                    self.log(f"val_{metric_name}", value, prog_bar=False)
            except Exception:
                # Silently fail if rank-eval not available or computation fails
                pass

        # Enhanced: Add confidence intervals for key metrics
        if HAS_CONFIDENCE_INTERVALS and len(pred_array) >= 10:
            try:
                ci_metrics = compute_metrics_with_ci(pred_array, target_array, n_bootstrap=100)
                # Log CI bounds for Spearman (most important metric)
                if "spearman_corr" in ci_metrics:
                    ci = ci_metrics["spearman_corr"]
                    self.log("val_spearman_ci_lower", ci["ci_lower"], prog_bar=False)
                    self.log("val_spearman_ci_upper", ci["ci_upper"], prog_bar=False)
            except Exception:
                # Silently fail if CI computation fails
                pass

        # Enhanced: Diagnostic analysis with concrete examples
        try:
            from tiny_icf.eval_diagnostic import (
                compute_diagnostic_metrics,
                format_diagnostic_report,
            )

            diagnostics = compute_diagnostic_metrics(
                pred_tensor,
                target_tensor,
                words=words_for_diagnostics,
            )

            # Log diagnostic metrics
            dist_metrics = diagnostics["distances"]
            self.log("val_percent_close_1pct", dist_metrics.get("percent_close_1pct", 0.0))
            self.log("val_percent_close_5pct", dist_metrics.get("percent_close_5pct", 0.0))
            self.log("val_percent_close_10pct", dist_metrics["percent_close_10pct"])
            self.log("val_percent_close_20pct", dist_metrics["percent_close_20pct"])
            self.log("val_percent_close_50pct", dist_metrics["percent_close_50pct"])
            self.log("val_percent_close_abs_01", dist_metrics["percent_close_abs_01"])
            self.log("val_percent_close_abs_05", dist_metrics["percent_close_abs_05"])
            self.log("val_percent_close_abs_10", dist_metrics["percent_close_abs_10"])
            self.log("val_mean_squared_error", dist_metrics["mean_squared_error"])
            self.log("val_median_absolute_error", dist_metrics["median_absolute_error"])
            self.log("val_mean_relative_error", dist_metrics["mean_relative_error"])
            self.log("val_max_absolute_error", dist_metrics["max_absolute_error"])

            # Log interesting case counts
            cases = diagnostics["interesting_cases"]
            self.log("val_num_close_calls", len(cases["close_calls"]))
            self.log("val_num_false_positives", len(cases["false_positives"]))
            self.log("val_num_false_negatives", len(cases["false_negatives"]))
            self.log("val_num_worst_offenders", len(cases["worst_offenders"]))
            self.log("val_num_ranking_errors", len(cases["ranking_errors"]))

            # Log ranking quality metrics
            if "ranking_quality" in diagnostics:
                ranking = diagnostics["ranking_quality"]
                self.log("val_precision_at_100", ranking.get("precision_at_k", 0.0))
                self.log("val_mrr_at_100", ranking.get("mrr_at_k", 0.0))
                self.log("val_mean_rank_error", ranking.get("mean_rank_error", 0.0))

            # Save diagnostic report to file (every 5 epochs or on best)
            current_epoch = (
                self.current_epoch
                if hasattr(self, "current_epoch")
                else getattr(self.trainer, "current_epoch", 0)
            )
            if current_epoch % 5 == 0 or current_epoch == 0:
                report = format_diagnostic_report(diagnostics, top_n=15)

                # Save to experiment directory if available
                if hasattr(self.trainer, "log_dir") and self.trainer.log_dir:
                    import os
                    import json

                    report_path = os.path.join(
                        self.trainer.log_dir, f"diagnostic_report_epoch_{current_epoch}.txt"
                    )
                    with open(report_path, "w") as f:
                        f.write(report)

                    # Also save JSON for programmatic access
                    json_path = os.path.join(
                        self.trainer.log_dir, f"diagnostic_data_epoch_{current_epoch}.json"
                    )
                    # Convert to JSON-serializable format
                    diagnostic_json = {
                        "distances": dist_metrics,
                        "interesting_cases": {
                            "close_calls": [
                                {"word": w, "pred": float(p), "target": float(t), "error": float(e)}
                                for w, p, t, e in cases["close_calls"][:20]
                            ],
                            "false_positives": [
                                {"word": w, "pred": float(p), "target": float(t), "error": float(e)}
                                for w, p, t, e in cases["false_positives"][:20]
                            ],
                            "false_negatives": [
                                {"word": w, "pred": float(p), "target": float(t), "error": float(e)}
                                for w, p, t, e in cases["false_negatives"][:20]
                            ],
                            "worst_offenders": [
                                {"word": w, "pred": float(p), "target": float(t), "error": float(e)}
                                for w, p, t, e in cases["worst_offenders"][:20]
                            ],
                            "ranking_errors": [
                                {
                                    "word1": w1,
                                    "pred1": float(p1),
                                    "target1": float(t1),
                                    "word2": w2,
                                    "pred2": float(p2),
                                    "target2": float(t2),
                                    "diff": float(d),
                                }
                                for w1, p1, t1, w2, p2, t2, d in cases["ranking_errors"][:20]
                            ],
                        },
                        "error_patterns": {
                            "by_icf_range": {
                                k: {
                                    kk: (
                                        float(vv)
                                        if isinstance(vv, (np.floating, np.integer))
                                        else vv
                                    )
                                    for kk, vv in v.items()
                                }
                                for k, v in diagnostics["error_patterns"]
                                .get("by_icf_range", {})
                                .items()
                            },
                            "by_length": {
                                str(k): {
                                    kk: (
                                        float(vv)
                                        if isinstance(vv, (np.floating, np.integer))
                                        else vv
                                    )
                                    for kk, vv in v.items()
                                }
                                for k, v in diagnostics["error_patterns"]
                                .get("by_length", {})
                                .items()
                            },
                        },
                    }
                    with open(json_path, "w") as f:
                        json.dump(diagnostic_json, f, indent=2)

                    # Log diagnostic report to Aim as text artifact
                    if self._aim_logger is not None:
                        try:
                            import aim

                            run = self._aim_logger.experiment
                            if run is not None:
                                # Log diagnostic report as text
                                # Aim Text objects need to be created properly
                                run.track(
                                    aim.Text(report),
                                    name="diagnostic_report",
                                    step=current_epoch,
                                    context={"subset": "validation", "epoch": current_epoch},
                                )

                                # Log diagnostic summary metrics individually (Aim prefers individual metrics)
                                run.track(
                                    len(cases["worst_offenders"]),
                                    name="diagnostic_worst_offenders_count",
                                    step=current_epoch,
                                    context={"subset": "validation"},
                                )
                                run.track(
                                    len(cases["close_calls"]),
                                    name="diagnostic_close_calls_count",
                                    step=current_epoch,
                                    context={"subset": "validation"},
                                )
                                run.track(
                                    len(cases["false_positives"]),
                                    name="diagnostic_false_positives_count",
                                    step=current_epoch,
                                    context={"subset": "validation"},
                                )
                                run.track(
                                    len(cases["false_negatives"]),
                                    name="diagnostic_false_negatives_count",
                                    step=current_epoch,
                                    context={"subset": "validation"},
                                )
                                run.track(
                                    len(cases["ranking_errors"]),
                                    name="diagnostic_ranking_errors_count",
                                    step=current_epoch,
                                    context={"subset": "validation"},
                                )
                        except Exception as e:
                            if not hasattr(self, "_aim_diagnostic_warned"):
                                print(f"⚠️  Failed to log diagnostic to Aim: {e}")
                                self._aim_diagnostic_warned = True
        except ImportError:
            # Diagnostic module not available, skip
            pass
        except Exception as e:
            # Don't fail validation if diagnostics fail
            if not hasattr(self, "_diagnostic_warned"):
                print(f"⚠️  Diagnostic analysis failed: {e}")
                self._diagnostic_warned = True

        # Enhanced: Log learning rate for monitoring
        if hasattr(self, "optimizers") and self.optimizers() is not None:
            try:
                current_lr = self.optimizers().param_groups[0]["lr"]
                self.log("learning_rate", current_lr, on_epoch=True, prog_bar=False)
            except Exception:
                pass

        # Clear for next epoch
        self.validation_predictions.clear()
        self.validation_targets.clear()
        self.validation_words.clear()
        self.validation_words.clear()

    def configure_optimizers(self):
        # Research: Component-specific learning rates (attention vs MLP)
        # Attention layers converge 2-3x faster, so use lower LR
        # MLP layers converge slower, so use higher LR
        use_component_specific_lr = self.config.get("use_component_specific_lr", False)

        if use_component_specific_lr:
            # Separate parameter groups for conv (feature extraction) vs MLP (prediction)
            # Research: Different components converge at different rates
            # For CNN models: conv layers extract features, MLP layers make predictions
            conv_params = []
            mlp_params = []
            embedding_params = []

            for name, param in self.named_parameters():
                if "emb" in name.lower():
                    embedding_params.append(param)
                elif "conv" in name.lower() or "bn" in name.lower():
                    conv_params.append(param)
                elif "linear" in name.lower() or "head" in name.lower():
                    mlp_params.append(param)
                else:
                    # Default to MLP group for unknown layers
                    mlp_params.append(param)

            # Research: Component-specific LRs for residual ranking models
            # - Embeddings: Lower LR (0.3-0.5×) - often huge and noisy, overfitting risk
            #   Note: 0.1× was too aggressive, causing training failure (0.0316 Spearman)
            # - Conv/Backbone: Moderate LR (0.5-1×) - feature extraction
            # - MLP/Head: Higher LR (1.0-1.2×) - task-specific, needs faster learning
            # For character-level CNNs with residuals: embeddings need regularization but not too much
            emb_lr = self.learning_rate * 0.3  # Moderate lower for embeddings (0.1× was too low)
            conv_lr = self.learning_rate * 0.5  # Moderate for feature extraction
            mlp_lr = self.learning_rate * 1.0  # Standard for prediction head (reference LR)

            optimizer = AdamW(
                [
                    {"params": embedding_params, "lr": emb_lr, "weight_decay": self.weight_decay},
                    {"params": conv_params, "lr": conv_lr, "weight_decay": self.weight_decay},
                    {"params": mlp_params, "lr": mlp_lr, "weight_decay": self.weight_decay},
                ]
            )
        else:
            # Standard single learning rate
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        # Research-based scheduler selection: ReduceLROnPlateau for adaptive scheduling
        # or CosineAnnealingLR with warmup for stable convergence
        scheduler_type = self.config.get(
            "scheduler_type", "cosine"
        )  # 'cosine', 'plateau', or 'cosine_warmup'
        max_epochs = self.config.get("epochs", 150)

        if scheduler_type == "plateau":
            # ReduceLROnPlateau: Adaptive scheduling based on validation metrics
            # Research shows this works well for ranking models with plateau detection
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",  # Maximize Spearman correlation
                factor=0.5,  # Reduce LR by half when plateauing
                patience=8,  # Wait 8 epochs before reducing
                min_lr=1e-6,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_spearman_corr",  # Monitor Spearman for plateau detection
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif scheduler_type == "cosine_warmup":
            # Cosine annealing with proper warmup implementation
            # Research: Warmup stabilizes training, especially with Spearman loss
            # Use epoch-based warmup (simpler and works correctly with LambdaLR)
            warmup_epochs = max(5, max_epochs // 20)  # 5% warmup, minimum 5 epochs

            # Create warmup + cosine scheduler
            # Research: Warmup helps stabilize training, especially with Spearman loss
            # LambdaLR receives epoch number when interval='epoch'
            def warmup_cosine_lambda(epoch):
                if epoch < warmup_epochs:
                    # Linear warmup: lr = base_lr * (epoch / warmup_epochs)
                    return float(epoch + 1) / float(warmup_epochs)
                else:
                    # Cosine annealing after warmup
                    progress = float(epoch - warmup_epochs) / float(
                        max(1, max_epochs - warmup_epochs)
                    )
                    import math

                    return 0.5 * (1 + math.cos(progress * math.pi))

            scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)

            # Use epoch interval (LambdaLR works correctly with epoch numbers)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Update every epoch
                    "frequency": 1,
                },
            }
        else:  # 'cosine' (default)
            # Standard cosine annealing (no warmup)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=1e-6,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
