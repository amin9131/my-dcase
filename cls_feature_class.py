# Contains routines for labels creation, features extraction and normalization
#


import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa
plot.switch_backend('agg')
import shutil
import math
import wave
import contextlib
import threading
import msvcrt
import time


def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


class FeatureClass:
    def __init__(self, params, is_eval=False):
        """

        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)

        # ---------new add---------:
        self._feat_win_configs_ms = params.get('feat_win_configs', None)   # مثلا [10, 20, 40, 80]
        if self._feat_win_configs_ms is not None:
            self._win_len_options = [int(self._fs * (w_ms / 1000.0)) for w_ms in self._feat_win_configs_ms]
            self._nfft_options = [self._next_greater_power_of_2(w) for w in self._win_len_options]
        else:
            self._win_len_options = None
            self._nfft_options = None
        # ^^^^^^^^^^^^new add^^^^^^^^^^^^^^^

        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)

        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = 4

        self._multi_accdoa = params['multi_accdoa']
        self._use_salsalite = params['use_salsalite']
        if self._use_salsalite and self._dataset=='mic':
            # Initialize the spatial feature constants
            self._lower_bin = int(np.floor(params['fmin_doa_salsalite'] * self._nfft / float(self._fs)))
            self._lower_bin = np.max((1, self._lower_bin))
            self._upper_bin = int(np.floor(np.min((params['fmax_doa_salsalite'], self._fs//2)) * self._nfft / float(self._fs)))


            # Normalization factor for salsalite
            c = 343
            self._delta = 2 * np.pi * self._fs / (self._nfft * c)
            self._freq_vector = np.arange(self._nfft//2 + 1)
            self._freq_vector[0] = 1
            self._freq_vector = self._freq_vector[None, :, None]  # 1 x n_bins x 1 

            # Initialize spectral feature constants
            self._cutoff_bin = int(np.floor(params['fmax_spectra_salsalite'] * self._nfft / float(self._fs)))
            assert self._upper_bin <= self._cutoff_bin, 'Upper bin for doa featurei {} is higher than cutoff bin for spectrogram {}!'.format()
            self._nb_mel_bins = self._cutoff_bin-self._lower_bin 
        else:
            self._nb_mel_bins = params['nb_mel_bins']
            # self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T
            self._mel_wts = self._compute_mel_wts(self._nfft)
        # Sound event classes dictionary
        self._nb_unique_classes = params['unique_classes']

        self._filewise_frames = {}
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._keyboard_thread = None
        self._user_stop_requested = False

    # def _keyboard_listener(self):
    #     """
    #     Listen for keyboard commands in a background thread.

    #     P -> pause / resume
    #     Q -> stop
    #     """

    #     while not self._stop_event.is_set():

    #         if msvcrt.kbhit():

    #             key = msvcrt.getwch().lower()

    #             if key == 'p':

    #                 if not self._pause_event.is_set():

    #                     self._pause_event.set()

    #                     print('\n')
    #                     print('==================================================')
    #                     print('PAUSE requested')
    #                     print('Current file will finish.')
    #                     print('Extraction will pause before the next file.')
    #                     print('Press P to resume or Q to stop.')
    #                     print('==================================================')

    #                 else:

    #                     self._pause_event.clear()

    #                     print('\n')
    #                     print('==================================================')
    #                     print('RESUMED')
    #                     print('==================================================')

    #             elif key == 'q':

    #                 self._user_stop_requested = True
    #                 self._stop_event.set()

    #                 print('\n')
    #                 print('==================================================')
    #                 print('STOP requested')
    #                 print('Current file will finish.')
    #                 print('Program will stop before the next file.')
    #                 print('==================================================')

    #         time.sleep(0.1)

    def is_stop_requested(self):
        return self._user_stop_requested     
    
    def _keyboard_listener(self):
        """
        Test keyboard listener.
        """

        print('Keyboard listener started.')

        while not self._stop_event.is_set():

            if msvcrt.kbhit():

                key = msvcrt.getwch()

                print(
                    '\nKEY RECEIVED:',
                    repr(key)
                )

                if key.lower() == 'p':

                    if not self._pause_event.is_set():

                        self._pause_event.set()

                        print('>>> PAUSE')

                    else:

                        self._pause_event.clear()

                        print('>>> RESUME')

                elif key.lower() == 'q':

                    self._user_stop_requested = True
                    self._stop_event.set()

                    print('>>> STOP')

            time.sleep(0.1)

    def _wait_if_paused(self):
        """
        Wait here while extraction is paused.

        P -> resume
        Q -> stop
        """

        while self._pause_event.is_set():

            if self._stop_event.is_set():
                raise KeyboardInterrupt

            time.sleep(0.2)

        if self._stop_event.is_set():
            raise KeyboardInterrupt

    def _wait_for_pause_or_continue(self):
        """
        Wait while the extraction is paused.

        This function is called between files.
        """

        self._check_pause()

    def _compute_mel_wts(self, nfft):
        return librosa.filters.mel(sr=self._fs, n_fft=nfft, n_mels=self._nb_mel_bins).T

    def get_frame_stats(self):

        if len(self._filewise_frames)!=0:
            return

        print('Computing frame stats:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))
        for sub_folder in os.listdir(self._aud_dir):
            loc_aud_folder = os.path.join(self._aud_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                with contextlib.closing(wave.open(os.path.join(loc_aud_folder, wav_filename),'r')) as f: 
                    audio_len = f.getnframes()
                nb_feat_frames = int(audio_len / float(self._hop_len))
                nb_label_frames = int(audio_len / float(self._label_hop_len))
                self._filewise_frames[file_name.split('.')[0]] = [nb_feat_frames, nb_label_frames]
        return

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input, _nb_frames, win_len=None, nfft=None):
        win_len = self._win_len if win_len is None else win_len
        nfft = self._nfft if nfft is None else nfft
        _nb_ch = audio_input.shape[1]
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=nfft, hop_length=self._hop_len,
                                        win_length=win_len, window='hann')
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T

    def _get_mel_spectrogram(self, linear_spectra, mel_wts=None):
        mel_wts = self._mel_wts if mel_wts is None else mel_wts
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        W = linear_spectra[:, :, 0]
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        E = self._eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1))/3.0 )
        
        I_norm = I/E[:, :, np.newaxis]
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0,2,1)), self._mel_wts), (0,2,1))
        foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv

    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def _get_salsalite(self, linear_spectra):
        # Adapted from the official SALSA repo- https://github.com/thomeou/SALSA
        # spatial features
        phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        phase_vector = phase_vector / (self._delta * self._freq_vector)
        phase_vector = phase_vector[:, self._lower_bin:self._cutoff_bin, :]
        phase_vector[:, self._upper_bin:, :] = 0
        phase_vector = phase_vector.transpose((0, 2, 1)).reshape((phase_vector.shape[0], -1))

        # spectral features
        linear_spectra = np.abs(linear_spectra)**2
        for ch_cnt in range(linear_spectra.shape[-1]):
            linear_spectra[:, :, ch_cnt] = librosa.power_to_db(linear_spectra[:, :, ch_cnt], ref=1.0, amin=1e-10, top_db=None)
        linear_spectra = linear_spectra[:, self._lower_bin:self._cutoff_bin, :]
        linear_spectra = linear_spectra.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        
        return np.concatenate((linear_spectra, phase_vector), axis=-1) 

    def _get_spectrogram_for_file(self, audio_filename, win_len=None, nfft=None):
        audio_in, fs = self._load_audio(audio_filename)
        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        nb_label_frames = int(len(audio_in) / float(self._label_hop_len))
        self._filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]
        audio_spec = self._spectrogram(audio_in, nb_feat_frames, win_len=win_len, nfft=nfft)
        return audio_spec

    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)
        se_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        x_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                for active_event in active_event_list:
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[2]
                    y_label[frame_ind, active_event[0]] = active_event[3]
                    z_label[frame_ind, active_event[0]] = active_event[4]

        label_mat = np.concatenate((se_label, x_label, y_label, z_label), axis=1)
        return label_mat

    # OUTPUT LABELS
    def get_adpit_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
        """

        se_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))  # [nb_frames, 6, max_classes]
        x_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
                active_event_list_per_class = []
                for i, active_event in enumerate(active_event_list):
                    active_event_list_per_class.append(active_event)
                    if i == len(active_event_list) - 1:  # if the last
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]

                    elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                        active_event_list_per_class = []

        label_mat = np.stack((se_label, x_label, y_label, z_label), axis=2)  # [nb_frames, 6, 4(=act+XYZ), max_classes]
        return label_mat

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------

    def extract_file_feature(self, _arg_in, cfg_id=None):
        _file_cnt, _wav_path, _feat_path = _arg_in

        if cfg_id is not None and self._win_len_options is not None:
            win_len = self._win_len_options[cfg_id]
            nfft = self._nfft_options[cfg_id]
            mel_wts = self._compute_mel_wts(nfft)
        else:
            win_len, nfft, mel_wts = None, None, self._mel_wts

        spect = self._get_spectrogram_for_file(_wav_path, win_len=win_len, nfft=nfft)

        if not self._use_salsalite:
            mel_spect = self._get_mel_spectrogram(spect, mel_wts=mel_wts)

        feat = None
        if self._dataset == 'foa':
            foa_iv = self._get_foa_intensity_vectors(spect)
            feat = np.concatenate((mel_spect, foa_iv), axis=-1)
        elif self._dataset == 'mic':
            if self._use_salsalite:
                feat = self._get_salsalite(spect)
            else:
                gcc = self._get_gcc(spect)
                feat = np.concatenate((mel_spect, gcc), axis=-1)
        else:
            print('ERROR: Unknown dataset format {}'.format(self._dataset))
            exit()

        
        if feat is not None:
            print('{}: {}, {}'.format(_file_cnt, os.path.basename(_wav_path), feat.shape))
            feat = feat.astype(np.float32)
            np.save(_feat_path, feat)
            
    def extract_all_feature(self, cfg_id=None):
        self._pause_event.clear()
        self._stop_event.clear()
        self._user_stop_requested = False

        self._feat_dir = self.get_unnormalized_feat_dir(cfg_id)
        create_folder(self._feat_dir)

        start_s = time.time()

        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir,
            self._desc_dir,
            self._feat_dir
        ))

        print('\nKeyboard control:')
        print('    P = pause / resume')
        print('    Q = stop')
        print()

        # -------------------------------------------------------------
        # Reset control flags for this extraction run
        # -------------------------------------------------------------
        self._pause_event.clear()
        self._stop_event.clear()

        # -------------------------------------------------------------
        # Start keyboard listener
        # -------------------------------------------------------------
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_listener,
            daemon=True
        )

        self._keyboard_thread.start()

        try:

            if self._is_eval:

                for file_cnt, file_name in enumerate(
                        os.listdir(self._aud_dir)):

                    # -------------------------------------------------
                    # Wait here if P was pressed
                    # -------------------------------------------------
                    self._wait_if_paused()

                    # -------------------------------------------------
                    # Stop before starting a new file
                    # -------------------------------------------------
                    if self._stop_event.is_set():
                        print('\nExtraction stopped.')
                        break

                    wav_filename = '{}.wav'.format(
                        file_name.split('.')[0]
                    )

                    wav_path = os.path.join(
                        self._aud_dir,
                        wav_filename
                    )

                    feat_path = os.path.join(
                        self._feat_dir,
                        '{}.npy'.format(
                            wav_filename.split('.')[0]
                        )
                    )

                    # -------------------------------------------------
                    # Skip already extracted feature
                    # -------------------------------------------------
                    if os.path.exists(feat_path):

                        print(
                            '{}: {}, already exists -> skip'.format(
                                file_cnt,
                                os.path.basename(wav_path)
                            )
                        )

                        continue

                    # -------------------------------------------------
                    # Extract current file
                    # -------------------------------------------------
                    # print(
                    #     '\n{}: extracting {}'.format(
                    #         file_cnt,
                    #         os.path.basename(wav_path)
                    #     )
                    # )

                    self.extract_file_feature(
                        (file_cnt, wav_path, feat_path),
                        cfg_id=cfg_id
                    )

                    # -------------------------------------------------
                    # Check stop after current file
                    # -------------------------------------------------
                    if self._stop_event.is_set():

                        print(
                            '\nCurrent file finished.'
                        )

                        print(
                            'Stopping before the next file.'
                        )

                        break

            else:

                for sub_folder in os.listdir(self._aud_dir):

                    # -------------------------------------------------
                    # Check stop between folders
                    # -------------------------------------------------
                    if self._stop_event.is_set():
                        break

                    loc_aud_folder = os.path.join(
                        self._aud_dir,
                        sub_folder
                    )

                    for file_cnt, file_name in enumerate(
                            os.listdir(loc_aud_folder)):

                        # ---------------------------------------------
                        # Wait if paused
                        # ---------------------------------------------
                        self._wait_if_paused()

                        # ---------------------------------------------
                        # Stop before starting next file
                        # ---------------------------------------------
                        if self._stop_event.is_set():
                            break

                        wav_filename = '{}.wav'.format(
                            file_name.split('.')[0]
                        )

                        wav_path = os.path.join(
                            loc_aud_folder,
                            wav_filename
                        )

                        feat_path = os.path.join(
                            self._feat_dir,
                            '{}.npy'.format(
                                wav_filename.split('.')[0]
                            )
                        )

                        # ---------------------------------------------
                        # Skip existing feature
                        # ---------------------------------------------
                        if os.path.exists(feat_path):

                            print(
                                '{}: {}, already exists -> skip'.format(
                                    file_cnt,
                                    os.path.basename(wav_path)
                                )
                            )

                            continue

                        # ---------------------------------------------
                        # Extract current file
                        # ---------------------------------------------
                        # print(
                        #     '\n{}: extracting {}'.format(
                        #         file_cnt,
                        #         os.path.basename(wav_path)
                        #     )
                        # )

                        self.extract_file_feature(
                            (file_cnt, wav_path, feat_path),
                            cfg_id=cfg_id
                        )

                        # ---------------------------------------------
                        # Check stop after current file
                        # ---------------------------------------------
                        if self._stop_event.is_set():

                            print(
                                '\nCurrent file finished.'
                            )

                            print(
                                'Stopping before the next file.'
                            )

                            break

                    if self._stop_event.is_set():
                        break

        except KeyboardInterrupt:

            print('\n')
            print('==================================================')
            print('Feature extraction stopped.')
            print('Already completed .npy files are preserved.')
            print('Run extraction again to continue.')
            print('==================================================')

        finally:
            if self._keyboard_thread is not None:
                self._keyboard_thread.join(timeout=1.0)
        # -------------------------------------------------------------
        # Final status
        # -------------------------------------------------------------
        if self._stop_event.is_set():
            print('\nFeature extraction stopped.')
        else:
            print('\nFeature extraction finished.')

        print(
            'Elapsed time: {:.2f} seconds'.format(
                time.time() - start_s
            )
        )

        if self._user_stop_requested:
            print('\nFeature extraction stopped.')
        else:
            print('\nFeature extraction finished.')

    def preprocess_features(self, cfg_id=None):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir(cfg_id)
        self._feat_dir_norm = self.get_normalized_feat_dir(cfg_id)
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file(cfg_id)
        spec_scaler = None

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        else:
            print('Estimating weights for normalizing feature files:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                spec_scaler.partial_fit(feat_file)
                del feat_file
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            feat_file = spec_scaler.transform(feat_file)
            np.save(
                os.path.join(self._feat_dir_norm, file_name),
                feat_file
            )
            del feat_file

        print('normalized files written to {}'.format(self._feat_dir_norm))

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self.get_frame_stats()
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)
        for sub_folder in os.listdir(self._desc_dir):
            loc_desc_folder = os.path.join(self._desc_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_desc_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                
                nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]
                desc_file_polar = self.load_output_format_file(os.path.join(loc_desc_folder, file_name))
                desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
                if self._multi_accdoa:
                    label_mat = self.get_adpit_labels_for_file(desc_file, nb_label_frames)
                else:
                    label_mat = self.get_labels_for_file(desc_file, nb_label_frames)
                print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
                np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file):
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 5:  # frame, class idx, source_id, polar coordinates(2) # no distance data, for example in synthetic data fold 1 and 2
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            if len(_words) == 6: # frame, class idx, source_id, polar coordinates(2), distance 
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            elif len(_words) == 7: # frame, class idx, source_id, cartesian coordinates(3), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
        _fid.close()
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
                _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), 0))
        _fid.close()

    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames/float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt+self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict

    def regression_label_format_to_output_format(self, _sed_labels, _doa_labels):
        """
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _nb_classes = self._nb_unique_classes
        _is_polar = _doa_labels.shape[-1] == 2*_nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2*_nb_classes]
            _z = _doa_labels[:, 2*_nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append([_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append([_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class], _z[_frame_ind, _tmp_class]])
        return _output_dict

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:

                    ele_rad = tmp_val[3]*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
        return out_dict

    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                    # in degrees
                    azimuth = np.arctan2(y, x) * 180 / np.pi
                    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                    r = np.sqrt(x**2 + y**2 + z**2)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], azimuth, elevation])
        return out_dict

    # ------------------------------- Misc public functions -------------------------------

    def get_normalized_feat_dir(self, cfg_id=None):
        suffix = '_cfg{}'.format(cfg_id) if cfg_id is not None else ''
        return os.path.join(
            self._feat_label_dir,
            '{}_norm{}'.format('{}_salsa'.format(self._dataset_combination) if (self._dataset=='mic' and self._use_salsalite) else self._dataset_combination, suffix)
        )

    def get_unnormalized_feat_dir(self, cfg_id=None):
        suffix = '_cfg{}'.format(cfg_id) if cfg_id is not None else ''
        return os.path.join(
            self._feat_label_dir,
            '{}{}'.format('{}_salsa'.format(self._dataset_combination) if (self._dataset=='mic' and self._use_salsalite) else self._dataset_combination, suffix)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir,
               '{}_label'.format('{}_adpit'.format(self._dataset_combination) if self._multi_accdoa else self._dataset_combination)
        )

    def get_normalized_wts_file(self, cfg_id=None):
        suffix = '_cfg{}'.format(cfg_id) if cfg_id is not None else ''
        return os.path.join(self._feat_label_dir, '{}_wts{}'.format(self._dataset, suffix))

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return self._nb_unique_classes

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_mel_bins(self):
        return self._nb_mel_bins


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)

def delete_and_create_folder(folder_name):
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)

