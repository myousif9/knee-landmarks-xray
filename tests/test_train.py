import torch

from src.segmentation.train import boundary_loss, dice_from_intersection_denominator


def test_dice_from_intersection_denominator_per_channel():
    intersection = torch.tensor([4.0, 2.0])
    denominator = torch.tensor([8.0, 8.0])

    dice = dice_from_intersection_denominator(intersection, denominator)

    assert torch.allclose(dice, torch.tensor([1.0, 0.5]))


def test_boundary_loss_supports_multichannel_targets():
    pred = torch.zeros((2, 2, 16, 16), dtype=torch.float32)
    target = torch.zeros((2, 2, 16, 16), dtype=torch.float32)
    target[:, 0, 4:10, 4:10] = 1
    target[:, 1, 6:12, 6:12] = 1

    loss = boundary_loss(pred, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
