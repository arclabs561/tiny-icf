import torch


def test_multitask_icf_return_features_includes_aux_logits():
    from tiny_icf.model_multi_task import MultiTaskICF

    model = MultiTaskICF(
        output_tasks=["icf", "language", "era", "temporal", "hygiene"],
        num_languages=10,
        num_eras=5,
        num_hygiene=8,
        temporal_decades=[1800, 1900, 2000],
    )
    x = torch.randint(0, 256, (4, 20), dtype=torch.long)

    y, feats = model(x, return_features=True)
    assert tuple(y.shape) == (4, 1)
    assert isinstance(feats, dict)
    assert "raw_output" in feats
    assert "confidence" in feats
    assert "language_logits" in feats and tuple(feats["language_logits"].shape) == (4, 10)
    assert "era_logits" in feats and tuple(feats["era_logits"].shape) == (4, 5)
    assert "hygiene_logits" in feats and tuple(feats["hygiene_logits"].shape) == (4, 8)
    assert "temporal_logits" in feats and tuple(feats["temporal_logits"].shape) == (4, 3)


def test_multitask_dataset_masks_and_hygiene_labels_collate():
    from tiny_icf.data_multi_task import (
        MultiTaskICFDataset,
        collate_multi_task_batch,
        HYGIENE_TO_INDEX,
    )

    temporal_data = {"hello": {1800: 0.9, 1900: 0.5, 2000: 0.2}}
    samples = [
        ("hello", 0.1, True),
        ("http://example.com", 0.5, False),
    ]
    ds = MultiTaskICFDataset(
        samples,
        max_length=20,
        augment_prob=0.0,
        include_language=True,
        include_era=True,
        include_hygiene=True,
        include_temporal=True,
        temporal_data=temporal_data,
        temporal_decades=[1800, 1900, 2000],
    )

    a = ds[0]
    b = ds[1]
    assert bool(a["icf_mask"].item()) is True
    assert bool(b["icf_mask"].item()) is False
    assert int(a["hygiene_targets"].item()) == HYGIENE_TO_INDEX["clean_word"]
    assert int(b["hygiene_targets"].item()) == HYGIENE_TO_INDEX["url"]
    assert bool(a["historical_mask"].item()) is True
    assert bool(b["historical_mask"].item()) is False

    batch = collate_multi_task_batch([a, b])
    assert tuple(batch["byte_tensors"].shape) == (2, 20)
    assert tuple(batch["icf_targets"].shape) == (2, 1)
    assert tuple(batch["icf_mask"].shape) == (2,)
    assert tuple(batch["hygiene_targets"].shape) == (2,)
    assert "historical_targets" in batch and "historical_mask" in batch
    assert tuple(batch["historical_mask"].shape) == (2,)
    # historical_targets is dict[int, Tensor[batch, 1]]
    assert 1800 in batch["historical_targets"]
    assert tuple(batch["historical_targets"][1800].shape) == (2, 1)


def test_unified_multitask_loss_accepts_hygiene():
    from tiny_icf.loss_unified import UnifiedMultiTaskLoss

    loss_fn = UnifiedMultiTaskLoss(use_amoo=False, hygiene_weight=1.0, icf_weight=0.0)
    hygiene_logits = torch.randn(5, 8)
    hygiene_targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    total, diag = loss_fn(hygiene_logits=hygiene_logits, hygiene_targets=hygiene_targets)
    assert torch.isfinite(total).item()
    assert isinstance(diag, dict)
    assert "task_losses" in diag and "hygiene" in diag["task_losses"]
