import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models import DeepConvNet, EEGNet82
from src.extract import EEGDataset
from src.train_oof_dcn import make_oof_splits


EEG_WEIGHT = 0.44  # selected exclusively on concatenated OOF predictions


def predict(model, data, labels, device):
    output = []
    model.eval()
    with torch.no_grad():
        for x, _ in DataLoader(EEGDataset(data, labels), batch_size=128):
            output.append(model(x.to(device)).cpu())
    return torch.cat(output)


def main():
    archive = np.load(os.path.join(ROOT, 'data', 'processed', 'eeg_dataset.npz'))
    data = archive['data'].astype(np.float32); labels = archive['labels_0indexed']
    splits, test_idx = make_oof_splits(labels, 5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcn_oof, eeg_oof, oof_targets = [], [], []
    dcn_test, eeg_test = [], []
    for fold, (_, val_idx) in enumerate(splits):
        dcn = DeepConvNet(24, 26, 501, temporal_kernel=15, dropout_rate=0.5).to(device)
        dcn.load_state_dict(torch.load(os.path.join(
            ROOT, 'models', 'checkpoints', 'oof_dcn_0_2000',
            f'fold_{fold}_seed_{42 + fold}.pth'), map_location=device))
        eeg = EEGNet82(24, 26, input_time_points=801,
                       temporal_kernel_length=25, dropout_rate=0.3).to(device)
        eeg.load_state_dict(torch.load(os.path.join(
            ROOT, 'models', 'checkpoints', 'oof_eegnet_k25',
            f'fold_{fold}_seed_{142 + fold}.pth'), map_location=device))
        dcn_oof.append(predict(dcn, data[val_idx, :, 50:551], labels[val_idx], device))
        eeg_oof.append(predict(eeg, data[val_idx], labels[val_idx], device))
        oof_targets.append(torch.from_numpy(labels[val_idx]).long())
        dcn_test.append(predict(dcn, data[test_idx, :, 50:551], labels[test_idx], device))
        eeg_test.append(predict(eeg, data[test_idx], labels[test_idx], device))

    dcn_oof = torch.cat(dcn_oof); eeg_oof = torch.cat(eeg_oof)
    oof_targets = torch.cat(oof_targets)
    scale = dcn_oof.std() / eeg_oof.std()
    oof_logits = (1 - EEG_WEIGHT) * dcn_oof + EEG_WEIGHT * scale * eeg_oof
    oof_acc = 100 * (oof_logits.argmax(1) == oof_targets).float().mean().item()
    test_logits = ((1 - EEG_WEIGHT) * torch.stack(dcn_test).mean(0)
                   + EEG_WEIGHT * scale * torch.stack(eeg_test).mean(0))
    test_targets = torch.from_numpy(labels[test_idx]).long()
    test_acc = 100 * (test_logits.argmax(1) == test_targets).float().mean().item()
    print(f'DCN weight: {1 - EEG_WEIGHT:.2f}')
    print(f'EEGNet weight: {EEG_WEIGHT:.2f}')
    print(f'OOF logit scale: {scale:.6f}')
    print(f'OOF accuracy: {oof_acc:.2f}%')
    print(f'Held-out test accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
