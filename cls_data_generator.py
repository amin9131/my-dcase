#
# Data generator for training the SELDnet
#

import os
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random


class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False
    ):
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()

        # --- RL multi-resolution support ---
        self._feat_win_configs = params.get('feat_win_configs', None)
        self._multi_config_mode = bool(self._feat_win_configs) and params.get('use_rl_adaptive_feat', False)
        self._num_configs = len(self._feat_win_configs) if self._multi_config_mode else 1
        if self._multi_config_mode:
            self._feat_dirs = [self._feat_cls.get_normalized_feat_dir(cfg_id=c) for c in range(self._num_configs)]
        else:
            self._feat_dirs = [self._feat_cls.get_normalized_feat_dir()]
        self._feat_dir = self._feat_dirs[0]   # نگه داشته شده برای backward-compat (لاگ‌گیری و شناسایی فایل‌ها)
        # --- end RL support ---
        
        self._multi_accdoa = params['multi_accdoa']

        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._nb_classes = self._feat_cls.get_nb_classes()

        self._circ_buf_feat = None
        self._circ_buf_label = None

        self._get_filenames_list_and_feat_label_sizes()

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
        if self._is_eval:
            label_shape = None
        else:
            if self._multi_accdoa is True:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3*3)
            else:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3)
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        print('Computing some stats about the dataset')
        max_frames, total_frames, temp_feat = -1, 0, []
        for filename in os.listdir(self._feat_dir):
            if int(filename[4]) in self._splits: # check which split the file belongs to
                self._filenames_list.append(filename)
                    
                temp_feat = np.load(os.path.join(self._feat_dir, filename))
                total_frames += (temp_feat.shape[0] - (temp_feat.shape[0] % self._feature_seq_len))
                if temp_feat.shape[0]>max_frames:
                    max_frames = temp_feat.shape[0]
  
        if len(temp_feat)!=0:
            self._nb_frames_file = max_frames if self._per_file else temp_feat.shape[0]
            self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins
        else:
            print('Loading features failed')
            exit()

        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
            if self._multi_accdoa is True:
                self._num_track_dummy = temp_label.shape[-3]
                self._num_axis = temp_label.shape[-2]
                self._num_class = temp_label.shape[-1]
            else:
                self._label_len = temp_label.shape[-1]
            self._doa_len = 3 # Cartesian

        if self._per_file:
            self._batch_size = int(np.ceil(max_frames/float(self._feature_seq_len)))
            print('\tWARNING: Resetting batch size to {}. To accommodate the inference of longest file of {} frames in a single batch'.format(self._batch_size, max_frames))
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor(total_frames / (self._batch_size*self._feature_seq_len)))

        self._feature_batch_seq_len = self._batch_size*self._feature_seq_len
        self._label_batch_seq_len = self._batch_size*self._label_seq_len
        return

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)

        # یک circular buffer به ازای هر config ویژگی (طول لیست = 1 در حالت غیر-RL، برای سازگاری با گذشته)
        self._circ_buf_feat_list = [deque() for _ in range(self._num_configs)]
        self._circ_buf_label = deque()

        file_cnt = 0
        if self._is_eval:
            for i in range(self._nb_total_batches):
                while len(self._circ_buf_feat_list[0]) < self._feature_batch_seq_len:
                    for cfg_idx, feat_dir in enumerate(self._feat_dirs):
                        temp_feat = np.load(os.path.join(feat_dir, self._filenames_list[file_cnt]))

                        for row in temp_feat:
                            self._circ_buf_feat_list[cfg_idx].append(row)

                        if self._per_file:
                            extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                            extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6
                            for row in extra_feat:
                                self._circ_buf_feat_list[cfg_idx].append(row)

                    file_cnt = file_cnt + 1

                feat = self._pop_batch_multi_config()
                yield self._collapse_config_dim(feat)

        else:
            for i in range(self._nb_total_batches):

                while len(self._circ_buf_feat_list[0]) < self._feature_batch_seq_len:
                    # لیبل رو اول می‌خونیم چون crop کردن feature به طول اون وابسته‌ست
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))
                    if not self._per_file:
                        temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                    temp_mul = temp_label.shape[0] // self._label_seq_len

                    for cfg_idx, feat_dir in enumerate(self._feat_dirs):
                        temp_feat = np.load(os.path.join(feat_dir, self._filenames_list[file_cnt]))
                        if not self._per_file:
                            temp_feat = temp_feat[:temp_mul * self._feature_seq_len, :]

                        for f_row in temp_feat:
                            self._circ_buf_feat_list[cfg_idx].append(f_row)

                        if self._per_file:
                            feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                            extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6
                            for f_row in extra_feat:
                                self._circ_buf_feat_list[cfg_idx].append(f_row)

                    # لیبل مستقل از config است، فقط یک‌بار push می‌شه
                    for l_row in temp_label:
                        self._circ_buf_label.append(l_row)
                    if self._per_file:
                        label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                        if self._multi_accdoa is True:
                            extra_labels = np.zeros((label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class))
                        else:
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))
                        for l_row in extra_labels:
                            self._circ_buf_label.append(l_row)

                    file_cnt = file_cnt + 1

                feat = self._pop_batch_multi_config()
                feat = self._collapse_config_dim(feat)

                if self._multi_accdoa is True:
                    label = np.zeros((self._label_batch_seq_len, self._num_track_dummy, self._num_axis, self._num_class))
                    for j in range(self._label_batch_seq_len):
                        label[j, :, :, :] = self._circ_buf_label.popleft()
                else:
                    label = np.zeros((self._label_batch_seq_len, self._label_len))
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()

                label = self._split_in_seqs(label, self._label_seq_len)
                if self._multi_accdoa is True:
                    pass
                else:
                    mask = label[:, :, :self._nb_classes]
                    mask = np.tile(mask, 3)
                    label = mask * label[:, :, self._nb_classes:]

                yield feat, label

    def _pop_batch_multi_config(self):
        """
        یک batch کامل رو از circular buffer هر config جدا pop می‌کنه، به شکل
        (n_seqs, ch, seq_len, mel_bins) در می‌آره، و روی محور جدید config استک می‌کنه
        -> خروجی: (n_seqs, num_configs, ch, seq_len, mel_bins)
        """
        feat_per_cfg = []
        for cfg_idx in range(self._num_configs):
            feat_c = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
            for j in range(self._feature_batch_seq_len):
                feat_c[j, :] = self._circ_buf_feat_list[cfg_idx].popleft()
            feat_c = np.reshape(feat_c, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))
            feat_c = self._split_in_seqs(feat_c, self._feature_seq_len)
            feat_c = np.transpose(feat_c, (0, 2, 1, 3))  # (n_seqs, ch, seq_len, mel_bins)
            feat_per_cfg.append(feat_c)
        return np.stack(feat_per_cfg, axis=1)  # (n_seqs, num_configs, ch, seq_len, mel_bins)

    def _collapse_config_dim(self, feat):
        """
        Backward compatibility: وقتی feat_win_configs تنظیم نشده (حالت غیر-RL)،
        محور config حذف می‌شه تا شکل خروجی دقیقاً مثل قبل از این تغییر
        (n_seqs, ch, seq_len, mel_bins) بمونه و کد فعلی train_seldnet.py/مدل‌ها بشکنه نه.
        وقتی حالت چند-config فعاله، محور config نگه داشته می‌شه تا RL wrapper ازش استفاده کنه.
        """
        if self._multi_config_mode:
            return feat
        return feat[:, 0]

    def _split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:  # for multi-ACCDOA with ADPIT
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2], data.shape[3]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()
    
    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict)
