# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameters
import sys


def main(argv):
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once.

    task_id = '8' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params)

    feat_win_configs = params.get('feat_win_configs', None)

    if feat_win_configs:

        # Multi-resolution mode
        print(
            'Extracting multi-resolution features for RL config candidates: {} ms'
            .format(feat_win_configs)
        )

        for cfg_id in range(len(feat_win_configs)):

            print(
                '\n--- Feature config {}: {} ms window ---'
                .format(cfg_id, feat_win_configs[cfg_id])
            )

            # ---------------------------------------------------------
            # Extract features
            # ---------------------------------------------------------
            dev_feat_cls.extract_all_feature(cfg_id=cfg_id)

            # If user pressed Q, stop the ENTIRE program.
            if dev_feat_cls.is_stop_requested():
                print('\n========================================')
                print('STOP requested by user.')
                print('Stopping entire program.')
                print('Normalization will NOT start.')
                print('Remaining configurations will NOT run.')
                print('Labels will NOT be extracted.')
                print('========================================')
                return

            # ---------------------------------------------------------
            # Normalize features
            # ---------------------------------------------------------
            dev_feat_cls.preprocess_features(cfg_id=cfg_id)

            # Check again after normalization
            if dev_feat_cls.is_stop_requested():
                print('\n========================================')
                print('STOP requested by user.')
                print('Stopping entire program.')
                print('Remaining configurations will NOT run.')
                print('Labels will NOT be extracted.')
                print('========================================')
                return

    else:

        # Original single-resolution behaviour
        dev_feat_cls.extract_all_feature()

        if dev_feat_cls.is_stop_requested():
            print('\n========================================')
            print('STOP requested by user.')
            print('Stopping entire program.')
            print('Normalization will NOT start.')
            print('Labels will NOT be extracted.')
            print('========================================')
            return

        dev_feat_cls.preprocess_features()

        if dev_feat_cls.is_stop_requested():
            print('\n========================================')
            print('STOP requested by user.')
            print('Stopping entire program.')
            print('Labels will NOT be extracted.')
            print('========================================')
            return

    # ---------------------------------------------------------
    # Extract labels
    # ---------------------------------------------------------
    dev_feat_cls.extract_all_labels()

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)