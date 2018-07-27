import numpy as np
import utils
from torch import nn

def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return (intersection / (union - intersection + epsilon)).mean()


def validation_multi(model: nn.Module, criterion, valid_loader, num_classes):
    model.eval()
    losses = []
    confusion_matrix = np.zeros(
        (num_classes, num_classes), dtype=np.uint32)
    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        output_classes = outputs.data.cpu().numpy().argmax(axis=1)
        target_classes = targets.data.cpu().numpy()
        confusion_matrix += calculate_confusion_matrix_from_arrays(
            output_classes, target_classes, num_classes)

    confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
    valid_loss = np.mean(losses)  # type: float
    ious = {'iou_{}'.format(cls + 1): iou
            for cls, iou in enumerate(calculate_iou(confusion_matrix))}

    dices = {'dice_{}'.format(cls + 1): dice
             for cls, dice in enumerate(calculate_dice(confusion_matrix))}

    average_iou = np.mean(list(ious.values()))
    average_dices = np.mean(list(dices.values()))

    print(
        'Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}'.format(valid_loss, average_iou, average_dices))
    metrics = {'valid_loss': valid_loss, 'iou': average_iou}
    metrics.update(ious)
    metrics.update(dices)
    metrics['valid_loss'] = (metrics['valid_loss']).astype(np.float64)
    metrics['iou'] = (metrics['iou']).astype(np.float64)
    return metrics


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices