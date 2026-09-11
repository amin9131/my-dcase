# test_target_mask_v1_shapes.py
#
# Lightweight dry-run for V1:
# DataGenerator (without RL) -> TargetMaskBranch
# -> extract_target_chunk_label -> BCEWithLogitsLoss -> backward()
#
# No real audio file or execution of batch_feature_extraction.py is required.

import os
import shutil
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import parameters
import cls_feature_class
import cls_data_generator
from target_mask_network import TargetMaskBranch, extract_target_chunk_label

TARGET_CLASS_IDX = 3  # Arbitrary value for the synthetic test -- in real usage, this must be the target class index in your 14-class dataset


def main():
    tmp_dir = tempfile.mkdtemp(prefix='v1_mask_test_')
    try:
        # ---------------------------------------------------------------
        # STEP 0: Parameters -- task-id='6' means mic+gcc+multi_accdoa,
        # without RL.
        # ---------------------------------------------------------------
        params = parameters.get_params('6')
        params['feat_label_dir'] = tmp_dir
        params['dataset_dir'] = tmp_dir
        params['label_sequence_length'] = 10
        params['quick_test'] = False

        feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])  # = 5
        params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution  # = 50
        params['t_pool_size'] = [feature_label_resolution, 1, 1]

        nb_mel_bins = params['nb_mel_bins']
        nb_ch = 10  # 4 mic + 6 GCC pairs
        nb_label_frames = 30  # Each file has 30 label frames = 3 chunks of 10 frames
        nb_feat_frames = nb_label_frames * feature_label_resolution  # = 150
        filenames = ['fold1_room1_mix001.npy', 'fold1_room1_mix002.npy', 'fold1_room1_mix003.npy']

        # Set batch_size to the total number of chunks across the three files
        # (3 files × 3 chunks = 9), so one batch contains all data from all
        # three files. This avoids depending on the order returned by
        # os.listdir or on the calculation of nb_total_batches, which could
        # otherwise affect whether the negative file appears in the first batch.
        params['batch_size'] = 9

        print('STEP 0: nb_mel_bins={}, nb_ch={}, target_class_idx={}'.format(
            nb_mel_bins, nb_ch, TARGET_CLASS_IDX))

        # ---------------------------------------------------------------
        # STEP 1: Create synthetic feature and label files
        # ---------------------------------------------------------------
        feat_cls = cls_feature_class.FeatureClass(params)

        feat_dir = feat_cls.get_normalized_feat_dir()
        os.makedirs(feat_dir, exist_ok=True)

        # Create the feature using random noise (rather than a constant value)
        # because this time we also want to verify actual learning (loss
        # reduction), not just tensor shapes.
        rng = np.random.default_rng(0)
        for fn in filenames:
            arr = rng.normal(size=(nb_feat_frames, nb_ch * nb_mel_bins)).astype(np.float32)
            np.save(os.path.join(feat_dir, fn), arr)

        label_dir = feat_cls.get_label_dir()
        os.makedirs(label_dir, exist_ok=True)

        # Intentionally keep one file completely negative for the target class
        # and activate the target class with high density in the remaining
        # files. This ensures that the batch contains both label 0 and label 1.
        # With completely random activation and 50% density on small chunks,
        # it is likely that every chunk contains at least one active frame,
        # which would make the test less meaningful.
        for file_idx, fn in enumerate(filenames):
            lbl = np.zeros((nb_label_frames, 6, 4, params['unique_classes']), dtype=np.float32)
            if file_idx > 0:  # Keep the first file completely negative
                active_frames = rng.choice(nb_label_frames, size=nb_label_frames // 2, replace=False)
                for f in active_frames:
                    lbl[f, 0, 0, TARGET_CLASS_IDX] = 1.0  # Activation, slot a0
                    lbl[f, 0, 1:, TARGET_CLASS_IDX] = rng.uniform(-1, 1, size=3)  # Arbitrary x, y, z
            np.save(os.path.join(label_dir, fn), lbl)

        print('STEP 1: {} synthetic feature/label files created (file 0 = all-negative for target class)'.format(len(filenames)))

        # ---------------------------------------------------------------
        # STEP 2: DataGenerator
        # ---------------------------------------------------------------
        data_gen = cls_data_generator.DataGenerator(
            params=params, split=1, shuffle=False, per_file=False, is_eval=False
        )
        feat, label = next(data_gen.generate())
        print('STEP 2: feat shape = {}, label shape = {}'.format(feat.shape, label.shape))

        print('   (filenames_list order as read by DataGenerator: {})'.format(data_gen._filenames_list))
        expected_feat_shape = (params['batch_size'], nb_ch, params['feature_sequence_length'], nb_mel_bins)
        assert feat.shape == expected_feat_shape, \
            'feat shape mismatch: got {}, expected {}'.format(feat.shape, expected_feat_shape)
        expected_label_shape = (params['batch_size'], params['label_sequence_length'], 6, 4, params['unique_classes'])
        assert label.shape == expected_label_shape, \
            'label shape mismatch: got {}, expected {}'.format(label.shape, expected_label_shape)
        print('   ✅ feat/label shapes correct -- matches existing DataGenerator output unchanged')

        # ---------------------------------------------------------------
        # STEP 3: TargetMaskBranch -- forward pass
        # ---------------------------------------------------------------
        x = torch.tensor(feat).float()
        label_t = torch.tensor(label).float()

        model = TargetMaskBranch(params)
        logit, mask, bss_mel = model(x)

        print('STEP 3: logit shape = {}, mask shape = {}, bss_mel shape = {}'.format(
            logit.shape, mask.shape, bss_mel.shape))
        assert logit.shape == (params['batch_size'],), 'logit shape mismatch'
        assert mask.shape == (params['batch_size'], params['feature_sequence_length'], nb_mel_bins), 'mask shape mismatch'
        assert bss_mel.shape == (params['batch_size'], 4, params['feature_sequence_length'], nb_mel_bins), 'bss_mel shape mismatch'
        assert mask.min().item() >= 0.0 and mask.max().item() <= 1.0, 'mask values must be in [0,1] (sigmoid)'
        print('   ✅ forward shapes correct, mask in [0,1]')

        # ---------------------------------------------------------------
        # STEP 4: Chunk-level label extraction
        # ---------------------------------------------------------------
        target = extract_target_chunk_label(label_t, TARGET_CLASS_IDX)
        print('STEP 4: target shape = {}, values = {}'.format(target.shape, target.tolist()))
        assert target.shape == (params['batch_size'],), 'target label shape mismatch'
        assert set(target.unique().tolist()).issubset({0.0, 1.0}), 'target must be binary'

        # Direct and unambiguous verification: manually sum the raw activation
        # values for each chunk and compare them with the output of
        # extract_target_chunk_label. This avoids relying on whether the
        # completely negative file happens to appear in the batch.
        raw_activity_sum = label_t[:, :, :, 0, TARGET_CLASS_IDX].sum(dim=(1, 2))
        expected_target = (raw_activity_sum > 0).float()
        assert torch.equal(target, expected_target), \
            'extract_target_chunk_label mismatch: got {}, expected {} (raw sums: {})'.format(
                target.tolist(), expected_target.tolist(), raw_activity_sum.tolist())
        print('   ✅ chunk-level target extraction verified against raw label sums (not just shape)')

        # ---------------------------------------------------------------
        # STEP 5: BCE loss + backward -- verify that the gradient actually
        # reaches the mask convolution layers.
        # ---------------------------------------------------------------
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(logit, target)
        print('STEP 5: initial loss = {:.4f}'.format(loss.item()))

        # Before backward, make sure the gradient is None/zero.
        first_mask_conv_weight = model.mask_net.conv1[0].weight
        assert first_mask_conv_weight.grad is None, 'grad should be None before backward'

        loss.backward()
        assert first_mask_conv_weight.grad is not None, 'no gradient reached mask conv1 -- coupling broken!'
        grad_norm = first_mask_conv_weight.grad.norm().item()
        assert grad_norm > 0, 'gradient on mask conv1 is exactly zero -- coupling broken!'
        print('   ✅ gradient reached mask_net.conv1, norm = {:.6f}'.format(grad_norm))
        print('   ✅ confirms mask -> apply -> classify coupling is wired correctly (not parallel/decoupled)')

        # ---------------------------------------------------------------
        # STEP 6: A few synthetic training steps -- verify that the loss
        # actually decreases.
        # ---------------------------------------------------------------
        model2 = TargetMaskBranch(params)
        optimizer = optim.Adam(model2.parameters(), lr=1e-3)
        losses = []
        for step in range(30):
            optimizer.zero_grad()
            logit2, _, _ = model2(x)
            loss2 = criterion(logit2, target)
            loss2.backward()
            optimizer.step()
            losses.append(loss2.item())

        print('STEP 6: loss over 30 steps: start={:.4f}, end={:.4f}'.format(losses[0], losses[-1]))
        assert losses[-1] < losses[0], 'loss did not decrease -- something is wrong with the learning path'
        print('   ✅ loss decreases with training on synthetic data -- gradient path is learnable')

        # ---------------------------------------------------------------
        # STEP 7: Parameter count -- verify that the model is lightweight
        # enough for a GTX1650 4GB.
        # ---------------------------------------------------------------
        nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mask_params = sum(p.numel() for p in model.mask_net.parameters() if p.requires_grad)
        cls_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
        print('STEP 7: total params = {:,} (mask_net={:,}, classifier={:,})'.format(
            nb_params, mask_params, cls_params))

        print('\n=================================================')
        print('All steps passed. V1 mask->apply->classify branch is wired correctly.')
        print('=================================================')

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
