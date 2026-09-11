
# train_target_mask_v1.py
#
# Independent V1 training: only the mask -> apply -> classify branch.
# There is no connection to CST_conformer.py / GCC / the main SELD output.
# This is intentional, according to our decision: first test this path
# independently, then connect it to the main SELD model.
#
# The existing DataGenerator is used without modification, so nothing needs
# to be changed in cls_data_generator.py / batch_feature_extraction.py.
#
# Run:
#   python3 train_target_mask_v1.py <task-id> <job-id>
#
# <task-id> must refer to a configuration without RL (for example,
# '6' = mic+gcc+multi_accdoa), because at this stage the STFT window length
# must remain the same fixed value selected during stage one with RL --
# not a variable configuration.


import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import cls_data_generator
import parameters
from target_mask_network import TargetMaskBranch, extract_target_chunk_label


# =====================================================================
# CRITICAL TODO: Before running, change this to the actual index of the
# target class (e.g., Drone) in your custom dataset's 14 classes.
# The current value is only a placeholder. If you do not change it,
# the branch will learn the wrong target.
# =====================================================================
TARGET_CLASS_IDX = 0

# TODO: Configure these according to the split convention of your custom
# dataset (similar to the train_splits/val_splits/test_splits logic in
# train_seldnet.py). Simple placeholder values are used for now.
TRAIN_SPLITS = [1, 2, 3]
VAL_SPLITS = [4]


def run_epoch(data_gen, model, optimizer, device, params, train=True):
    model.train() if train else model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_correct, total_count, nb_batches = 0., 0, 0, 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for feat, label in data_gen.generate():
            feat_t = torch.tensor(feat).float().to(device)     # (B, nb_ch, T, F)
            label_t = torch.tensor(label).float().to(device)   # (B, T_label, 6, 4, nb_classes)
            target = extract_target_chunk_label(label_t, TARGET_CLASS_IDX).to(device)

            if train:
                optimizer.zero_grad()

            logit, mask, bss_mel = model(feat_t)
            loss = criterion(logit, target)

            if train:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                pred = (torch.sigmoid(logit) > 0.5).float()
                total_correct += (pred == target).sum().item()
                total_count += target.numel()

            total_loss += loss.item()
            nb_batches += 1
            if params['quick_test'] and nb_batches == 4:
                break

    avg_loss = total_loss / max(nb_batches, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def main(argv):
    task_id = '6' if len(argv) < 2 else argv[1]   # Default: mic+gcc+multi_accdoa, without RL
    job_id = 1 if len(argv) < 3 else argv[2]

    params = parameters.get_params(task_id)

    if params.get('use_rl_adaptive_feat', False):
        raise ValueError(
            'This script is intended for stage two (fixed window length). '
            'Do not use a task-id with use_rl_adaptive_feat=True here -- '
            'first find the best window length with RL in stage one, then '
            'pre-extract the features using that fixed length and use a '
            'task-id without RL here.'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('TARGET_CLASS_IDX =', TARGET_CLASS_IDX, '(check that this is correct!)')

    data_gen_train = cls_data_generator.DataGenerator(
        params=params, split=TRAIN_SPLITS, shuffle=True, per_file=False
    )
    data_gen_val = cls_data_generator.DataGenerator(
        params=params, split=VAL_SPLITS, shuffle=False, per_file=True
    )

    model = TargetMaskBranch(params).to(device)
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('V1 model params: {:,}'.format(nb_params))

    optimizer = optim.Adam(model.parameters(), lr=params.get('mask_lr', 1e-3))

    nb_epoch = 2 if params['quick_test'] else params.get('mask_nb_epochs', 50)
    patience = params.get('mask_patience', 10)

    best_val_loss = float('inf')
    patience_cnt = 0
    model_dir = params['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = os.path.join(model_dir, 'target_mask_v1_{}_{}_best.pt'.format(task_id, job_id))

    for epoch in range(nb_epoch):
        t0 = time.time()
        train_loss, train_acc = run_epoch(data_gen_train, model, optimizer, device, params, train=True)
        val_loss, val_acc = run_epoch(data_gen_val, model, optimizer, device, params, train=False)
        dt = time.time() - t0

        print('epoch {}: time={:.1f}s train_loss={:.4f} train_acc={:.3f} val_loss={:.4f} val_acc={:.3f}'.format(
            epoch, dt, train_loss, train_acc, val_loss, val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
            print('  -> saved best checkpoint to {}'.format(ckpt_path))
        else:
            patience_cnt += 1
            if patience_cnt > patience:
                print('early stopping at epoch {}'.format(epoch))
                break

    print('Done. Best val_loss = {:.4f}, checkpoint at {}'.format(best_val_loss, ckpt_path))


if __name__ == '__main__':
    main(sys.argv)
