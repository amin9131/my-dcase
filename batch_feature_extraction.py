# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameters
import sys


def main(argv):
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once. 

    # use parameter set defined by user
    task_id = '8' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    feat_win_configs = params.get('feat_win_configs', None)

    if feat_win_configs:
        # Multi-resolution mode: extract + normalize once per candidate window length,
        # each dumped to its own suffixed folder (see get_*_dir(cfg_id) in cls_feature_class.py)
        print('Extracting multi-resolution features for RL config candidates: {} ms'.format(feat_win_configs))
        for cfg_id in range(len(feat_win_configs)):
            print('\n--- Feature config {}: {} ms window ---'.format(cfg_id, feat_win_configs[cfg_id]))
            dev_feat_cls.extract_all_feature(cfg_id=cfg_id)
            dev_feat_cls.preprocess_features(cfg_id=cfg_id)
    else:
        # Original single-resolution behaviour, unchanged
        dev_feat_cls.extract_all_feature()
        dev_feat_cls.preprocess_features()

    # Extract labels — independent of window length, so this runs only once regardless of mode
    dev_feat_cls.extract_all_labels()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)