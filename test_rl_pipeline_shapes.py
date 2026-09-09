
# test_rl_pipeline_shapes.py
#
# Lightweight dry-run to test the chain:
# DataGenerator (multi-config) -> RLAdaptiveWrapper (gather)
# -> MSELoss_ADPIT (per-sample) -> PPORolloutBuffer/ppo_update
# without requiring real audio files or running batch_feature_extraction.py

import os
import shutil
import tempfile
import numpy as np
import torch

import parameters
import cls_feature_class
import cls_data_generator
import CST_conformer
import seldnet_model
import rl_feature_selector as rlfs


def main():
    tmp_dir = tempfile.mkdtemp(prefix='rl_pipeline_test_')
    try:
        # ---------------------------------------------------------------
        # STEP 0: Parameters (task-id=8 -> mic+gcc+multi_accdoa+RL enabled)
        # ---------------------------------------------------------------
        params = parameters.get_params('8')
        params['feat_label_dir'] = tmp_dir
        params['dataset_dir'] = tmp_dir           # Only needed for path construction; it is not read
        params['batch_size'] = 2
        params['label_sequence_length'] = 10
        params['quick_test'] = False

        # Recalculate derived parameters (because we modified them after get_params)
        feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])  # = 5
        params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
        params['t_pool_size'] = [feature_label_resolution, 1, 1]

        num_configs = len(params['feat_win_configs'])
        nb_mel_bins = params['nb_mel_bins']
        nb_ch = 10  # 4 mic + 6 GCC pairs -- fixed value in the current pipeline
        nb_label_frames = 20
        nb_feat_frames = nb_label_frames * feature_label_resolution  # = 100
        filenames = ['fold1_room1_mix001.npy', 'fold1_room1_mix002.npy']

        print('STEP 0: Parameters are ready. num_configs={}, nb_mel_bins={}'.format(num_configs, nb_mel_bins))

        # ---------------------------------------------------------------
        # STEP 1: Create synthetic feature files (one constant value per config)
        # ---------------------------------------------------------------
        feat_cls = cls_feature_class.FeatureClass(params)  # Only used for path-naming methods

        for cfg_id in range(num_configs):
            d = feat_cls.get_normalized_feat_dir(cfg_id=cfg_id)
            os.makedirs(d, exist_ok=True)
            for fn in filenames:
                arr = np.full((nb_feat_frames, nb_ch * nb_mel_bins), fill_value=float(cfg_id), dtype=np.float32)
                np.save(os.path.join(d, fn), arr)

        label_dir = feat_cls.get_label_dir()
        os.makedirs(label_dir, exist_ok=True)
        for fn in filenames:
            lbl = (np.random.rand(nb_label_frames, 6, 4, params['unique_classes']).astype(np.float32) * 0.1)
            np.save(os.path.join(label_dir, fn), lbl)

        print('STEP 1: Synthetic files created ({} configs x {} files)'.format(num_configs, len(filenames)))

        # ---------------------------------------------------------------
        # STEP 2: DataGenerator -- test multi-config stacking
        # ---------------------------------------------------------------
        data_gen = cls_data_generator.DataGenerator(
            params=params, split=1, shuffle=False, per_file=False, is_eval=False
        )
        feat, label = next(data_gen.generate())
        print('STEP 2: feat shape = {}, label shape = {}'.format(feat.shape, label.shape))

        expected_feat_shape = (params['batch_size'], num_configs, nb_ch,
                                params['feature_sequence_length'], nb_mel_bins)
        assert feat.shape == expected_feat_shape, \
            'feat shape mismatch: got {}, expected {}'.format(feat.shape, expected_feat_shape)
        print('   ✅ feat shape is correct')

        # ---------------------------------------------------------------
        # STEP 3: RLAdaptiveWrapper -- test forward + gather correctness
        # ---------------------------------------------------------------
        data_in, data_out = data_gen.get_data_sizes()
        backbone = CST_conformer.CST_former(data_in, data_out, params)
        model = rlfs.RLAdaptiveWrapper(backbone, num_configs, hidden=params['rl_hidden_dim'])

        x = torch.tensor(feat).float()
        target = torch.tensor(label).float()

        doa, action, log_prob, value, entropy, state = model(x, prev_action=None, greedy=False)
        print('STEP 3: doa shape = {}, expected = {}'.format(doa.shape, tuple(data_out)))
        assert doa.shape == torch.Size(data_out), 'doa shape mismatch'
        print('   ✅ backbone output shape is correct')

        # Verify gather: the selected value must be exactly equal to the selected config number
        for b in range(x.shape[0]):
            expected_val = float(action[b].item())
            selected_slice = x[b, action[b]]
            actual_val = selected_slice.mean().item()
            assert abs(actual_val - expected_val) < 1e-5, \
                'gather mismatch in sample {}: expected {}, got {}'.format(b, expected_val, actual_val)
        print('   ✅ gather works correctly: selected value exactly matches the action')

        # ---------------------------------------------------------------
        # STEP 4: MSELoss_ADPIT -- test per-sample output
        # ---------------------------------------------------------------
        criterion = seldnet_model.MSELoss_ADPIT()
        loss, loss_per_sample = criterion(doa, target)
        print('STEP 4: loss = {:.4f}, loss_per_sample shape = {}'.format(loss.item(), loss_per_sample.shape))
        assert loss_per_sample.shape[0] == params['batch_size'], 'loss_per_sample length must equal batch_size'
        print('   ✅ per-sample loss is correct')

        # ---------------------------------------------------------------
        # STEP 5: jitter + reward + PPORolloutBuffer + ppo_update
        # ---------------------------------------------------------------
        jitter = rlfs.jitter_penalty(action, None, num_configs)
        reward = -loss_per_sample.detach() - params['jitter_penalty_coef'] * jitter

        buffer = rlfs.PPORolloutBuffer()
        buffer.add(state, action, log_prob, value, reward)

        # Add a second synthetic transition (simulating the next batch)
        # so that the buffer contains more than one sample
        doa2, action2, log_prob2, value2, entropy2, state2 = model(x, prev_action=action.detach(), greedy=False)
        loss2, loss_per_sample2 = criterion(doa2, target)
        jitter2 = rlfs.jitter_penalty(action2, action.detach(), num_configs)
        reward2 = -loss_per_sample2.detach() - params['jitter_penalty_coef'] * jitter2
        buffer.add(state2, action2, log_prob2, value2, reward2)

        print('STEP 5: buffer size before update = {}'.format(len(buffer)))

        rl_optimizer = torch.optim.Adam(model.actor_critic.parameters(), lr=params['rl_lr'])
        rlfs.ppo_update(
            model.actor_critic, buffer, rl_optimizer,
            clip_eps=params['ppo_clip_eps'], epochs=2, minibatch_size=2,
            ent_coef=params['ppo_ent_coef'], vf_coef=params['ppo_vf_coef'],
            gamma=params['ppo_gamma'], lam=params['ppo_lambda']
        )
        assert len(buffer) == 0, 'buffer should be empty after ppo_update'
        print('   ✅ ppo_update executed without errors and the buffer was cleared')

        print('\n=================================================')
        print('All steps passed successfully. The shape pipeline is healthy.')
        print('=================================================')

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()