import os
import glob
import random
import librosa
import numpy as np
import os.path as osp
import nlpaug.flow as naf
import matplotlib.pyplot as plt
import nlpaug.augmenter.audio as naa
import librosa.display as librosa_display

from playsound import playsound

class Audioaug():
    def __init__(self, dbname=None, sample_rate=None, mode='simple_audio_aug'):
        self.path = osp.dirname(osp.abspath(__file__))
        self.sample_rate          = sample_rate
        self.background_noise     = []
        self.noise_count          = 0
        self.max_ratio            = 0.1
        self.background_noise_dir = osp.join(self.path, '..', '..', 'DB', dbname, '_background_noise_')
        
        if osp.isdir(self.background_noise_dir):
            noise_wave_list = glob.glob(osp.join(self.background_noise_dir, '*.wav'))
            self.background_noise = [librosa.load(i) for i in noise_wave_list]
            self.noise_count = len(self.background_noise)-1
        
        if mode=='simple_audio_aug':
            self.aug = naf.Sometimes([
            naa.NoiseAug(noise_factor=1000),
            naa.LoudnessAug(loudness_factor=(0, 4)),
            naa.PitchAug(sampling_rate=sample_rate, pitch_range=(-2,4)),
            naa.ShiftAug(sampling_rate=sample_rate,shift_max=0.2),
            naa.SpeedAug(speed_range=(0.1,0.4)), 
        ])
            self.albu = self.sample_audio_aug
        else:
            print('{} is an invalid Augmentation Mode! '.format(mode))
            self.albu = None

    def do_sometimes(self, n):
        if n < 1:
            return False
        if n==1:
            return True
        rn = random.randint(0, n)
        if rn==0:
            return True
        else:
            return False

    def sample_audio_aug(self, audio):
        augmented_data = self.aug.augment(audio)
        return augmented_data

    def apply(self, data):
        if not self.albu is None:
            if self.do_sometimes(4):
                data = self.albu(data)
            if self.do_sometimes(3):
                rn = self.get_one_noise(noiselen=len(data))
                if not rn is None:
                    data += (self.max_ratio * rn)
        return data

    def get_one_noise(self, noiselen=8000):
        noise_ = None
        if len(self.background_noise):
            selected_noise = self.background_noise[random.randint(0, self.noise_count)][0]
            start_idx = random.randint(0, len(selected_noise)- 1 - noiselen)
            noise_ = selected_noise[start_idx:(start_idx + noiselen)]
        return noise_
    
    '''
    TODO
    def plot_write_play(self, plot, augmented_data,audio_file):
        augmented_data = augmented_data.astype(np.int16)
        if plot:
            librosa_display.waveplot(augmented_data, sr=sample_rate, alpha=0.5)
            librosa_display.waveplot(data, sr=sample_rate, color='r', alpha=0.25)

            plt.tight_layout()
            plt.show()
        
        filename, file_extension = os.path.splitext(audio_file)
        op_filename = os.path.join('out',audio_file)
        write_audio(augmented_data, op_filename, sample_rate) 
        if playback == True:
            playsound(op_filename)
    '''


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file', type=str, default='sample.wav', help='audio_file', required=False)
    parser.add_argument('--clip_duration', type=float, default= 1.5, help='clip_duration', required=False)
    parser.add_argument('--sample_rate', type=int, default=16000, help='')
    parser.add_argument('--aug_method', type=str, default='loudness_aug', help='')
    parser.add_argument('--playback', type=bool, default=True, help='')
    args = parser.parse_args()
   
    audio_file = args.audio_file
    clip_duration = args.clip_duration 
    sample_rate = args.sample_rate
    aug_method = args.aug_method
    playback = args.playback
    plot=False
    
    desired_nsamples = int(clip_duration*sample_rate)
    fs, data = read_audio(audio_file)
    augmented_data = apply_augmentation(data, sample_rate)
    plot_write_play(plot,augmented_data,sample_rate,audio_file)