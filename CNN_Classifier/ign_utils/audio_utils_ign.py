import glob
import random
import librosa
import shutil
import numpy as np
import soundfile as sf
import time
import os.path as osp
from scipy.io import wavfile
from scipy.io.wavfile import read, write
from playsound import playsound
import math
from multiprocessing import Pool
import multiprocessing
from multiprocessing import Process, Lock

from functools import partial
try:
    from .array_utils import moving_window
    from .general_utils import create_directory_safe, do_sometimes
except:
    from array_utils import moving_window
    from general_utils import create_directory_safe, do_sometimes
import os


class Create_Custom_DB():
    def __init__(self, dbname=None):
        self.path = osp.dirname(osp.abspath(__file__))
        from general_utils import create_directory_safe,mkdir_safe, get_timestamp
        self.dbname=dbname
        self.src_dir = osp.join(self.path, '..', 'DB', self.dbname, 'train')
        self.val_dir = osp.join(self.path, '..', 'DB', self.dbname, 'val')
        create_directory_safe(self.src_dir)
        create_directory_safe(self.val_dir)
        self.classlist = []
        self.class_dict = {0:'fan_on', 1:'fan_off', 2:'breeze_mode', 3:'otto_mode', 4:'turbo_mode', 5:'unknown', 6:'silence'}
        self.trainfiles = glob.glob(osp.join(self.src_dir, '**', '*' + '.wav'), recursive=True)
        print('Total Files: ', len(self.trainfiles))
    
    def add_audiosamples(self, num_cycles=1):
        for i in range (num_cycles):
            print('Available classes: ', self.class_dict)
            labelval = input('Enter Class label val from dict, number should be between 0, {}\n'.format(len(self.class_dict)-1))
            label = self.class_dict[int(labelval)]
            class_dir = osp.join(self.src_dir, label)
            if label not in self.classlist:
                self.classlist.append(label)
                create_directory_safe(class_dir)
            record_audio(duration=1, sr=8000, filename=osp.join(class_dir, '{}_{}_{}.wav'.format(label, str(i), str(random.randint(0, 1000)))))
        self.trainfiles = glob.glob(osp.join(self.src_dir, '**', '*' + '.wav'), recursive=True)

    def create_val_split(self, fraction=0.2):
        print('Val split fraction: {} %'.format(fraction * 100))
        valfiles = random.sample(self.trainfiles, int(len(self.trainfiles)*fraction))
        for filep in valfiles:
            classname = filep.split('/')[-2]
            create_directory_safe(osp.join(self.val_dir, classname))
            shutil.move(filep, osp.join(self.val_dir, classname, osp.basename(filep)))
        print('Moved {} files to Val dir'.format(len(valfiles)))

def plot_audio(signal):
    import matplotlib.pyplot as plt
    
    #import pdb;pdb.set_trace()
    #print('min max',np.min(signal), np.max(signal))
    
    plt.figure(1)
    plt.title("Signal Wave...")
    plt.plot(signal)
    plt.show(block=False)
    plt.pause(0.001)
    
def ReadAudio(filename, normalize = False):
    """
    Read wave file as mono.

    Args:
        - filename (str) : wave file / path.

    Returns:
        tuple of sampling rate and audio data
    """
    fs, sig = read(filename=filename)
    #import pdb;pdb.set_trace()
    if (sig.ndim == 1):
        samples = sig
    else:
        samples = sig[:, 0]
        
    if normalize == True:
        samples = (samples/32768).astype('float32') 
           
    return samples, fs 


def ReadAudio(filename, sr = None, normalize = False):
    sr = sr
    if sr is None:
        fs, samples = read(filename=filename)
  
        if normalize == True:
            samples = (samples/32768).astype('float32')
    else:
        samples, fs = librosa.load(filename, sr=sr)
       
        if normalize == False:
            samples = (samples*32768).astype(np.int16)
    
    if (samples.ndim != 1):
       # import pdb;pdb.set_trace()       
        samples = samples[:, 0]
   #import pdb;pdb.set_trace() 
    samples = np.squeeze(samples)
           
    return samples, fs

def WriteAudio_with_int16(data, filename='temp.wav', sr=16000 ):
    data = convert_to_int16(data)
    return write(filename=filename, rate=sr, data=data)    
   
def WriteAudio(data, filename = 'temp.wav', sr = 16000 ):
    """
    Read wave file as mono.
    Args:
        - filename (str) : path to save resulting wave file to.
        - data            (array) : signal/audio array.
        - sr               (int) : sampling rate.

    Returns:
        tuple of sampling rate and audio data.
    """
    return write(filename=filename, rate=sr, data=data) 

def generate_random_noise_signal():
    sr = 8000 # sample rate
    T = random.randint(1, 2)    # seconds
    t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
    freq = random.randint(150, 600)
    x = 0.5*np.sin(2*np.pi*freq*t)# pure sine wave at 220 Hz
    wn = np.random.randn(len(x))
    data_wn = x + 0.009*wn
    return data_wn

def record_audio(duration=1.5, sr=16000, folder_path='test_samples'):
    import sounddevice as sd
    mydata = sd.rec(int(sr * duration), samplerate=sr,channels=1, blocking=True)
    
    sd.wait()
    filename = 'temp.wav'
    sf.write(filename, mydata, sr)
    return np.squeeze(mydata)
    
def play_audio(data,sr=16000):
    WriteAudio(data, filename = 'temp.wav', sr = 16000 )
    playsound('temp.wav')
    
def play_audio_sa(data,sr=16000):
    import simpleaudio as sa
    play_obj =sa.play_buffer(audio_data=data, num_channels=1, bytes_per_sample=2, sample_rate=sr)
    play_obj.wait_done()

def cal_rms(signal):
    '''
    calculate RMS value of a signal
    '''
    # abs_signal = abs(signal)
    # major_signal = abs_signal[abs_signal > threshold]
    # signal = signal.astype(np.float32)
    return np.sqrt(np.mean(np.square(signal), axis=-1))

def cal_db(amplitude):
    '''
    Calculate power signal value in decibel
    '''
    return 20 * math.log10(amplitude)

def cal_adjusted_rms(clean_rms, snr):
    # import pdb;pdb.set_trace()
    '''
    given a signal and snr value, calculate rms value of adjusted signal
    '''
    a = float(snr) / 20  # SNR/20 = log10 (Aspeech/ANoise)
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def get_adjusted_noise(speech, noise, snr):
    '''
    adjust signal based on snr value 
    '''
    try:
        speech = extract_peak_signal(speech, threshold=500)
    except:
        pass
    speech_rms = cal_rms(speech)
    adjusted_noise_rms = cal_adjusted_rms(speech_rms, snr)
    noise_rms = cal_rms(noise)
    adjusted_noise = noise * (adjusted_noise_rms / noise_rms) 
    # print('DB: LEVEL OF NOISE AD', get_db_level(adjusted_noise))
    # import pdb;pdb.set_trace()
    return adjusted_noise

def extract_peak_signal(signal, threshold=500):
    abs_signal = abs(speech)
    sel_ = moving_window(abs_signal, window=201, stride=100)
    max_sel = np.max(sel_, axis=1)
    max_sel = max_sel.repeat(100, axis=0)
    max_sel = np.squeeze(max_sel)
    sp_signal = abs_signal[max_sel>threshold]
    return sp_signal

def get_db_level(signal, isnoise=True):
    '''
    given a signal, get decibel strength
    '''
    if not isnoise:
        try:
            signal = extract_peak_signal(signal, threshold=500)
        except:
            pass
        signal_rms = cal_rms(signal)
    else:
        signal_rms = cal_rms(signal)
    
    # print('signal_rms: ', signal_rms)
    db_level = cal_db(signal_rms)
    return db_level, signal_rms

def convert_to_int16(signal):
    return signal.astype(np.int16)

def adjust_db_level(signal, db_level):
    '''
    adjust a signal to given db level
    '''
    clippedFlag = False
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    
    db, signal_rms = get_db_level(signal, isnoise=False)
    # print('Actual db level: ', db)
    target_rmsval = 10**(db_level/20)

    # print('target_rmsval: ', target_rmsval)
    db_ratio = target_rmsval/signal_rms
    ad_signal = (signal*db_ratio)
    final_db, __ = get_db_level(ad_signal, isnoise=False)
    
    
    if min_int16 > min(ad_signal) or max(ad_signal) > max_int16:
        # print('adDB: ', final_db)
        # print('max: ', max_int16)
        # print('min: ', min_int16)
        # print('maxval: ', max(ad_signal))
        # print('minval: ', min(ad_signal))
        clippedFlag = True
        # print('clippedFlag: ', clippedFlag)
        max_val = max(abs(max(ad_signal)), abs(min(ad_signal)))
        ad_signal /= max_val
        ad_signal=ad_signal*max_int16
    return ad_signal, clippedFlag

def clip_sanity_check(signal):
    '''
    adjust signal comparing min_mx int16 vals
    '''
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if signal.max(axis=0) > max_int16 or signal.min(axis=0) < min_int16:
        if signal.max(axis=0) >= abs(signal.min(axis=0)): 
            reduction_rate = max_int16 / signal.max(axis=0)
        else:
            reduction_rate = min_int16 / signal.min(axis=0)
        # print('reduction_rate: ', reduction_rate)
        signal = signal * (reduction_rate)
    return signal


def get_mixed_signal(speech, sp_level, noise, snr):
    speech = speech.astype(np.float32)
    noise = noise.astype(np.float32)
    speech_len = len(speech)
    noise_len = len(noise)
    start_pt = random.randint(0, (noise_len-speech_len)-1)
    end_pt = start_pt + speech_len
    noise = noise[start_pt:end_pt]
    # noise = noise[:speech_len]
    ad_speech, clippedFlag = adjust_db_level(speech, sp_level)
    adjusted_noise = get_adjusted_noise(ad_speech, noise, snr)
    mixed_signal = adjusted_noise + ad_speech
    mixed_signal = clip_sanity_check(mixed_signal)
    return mixed_signal.astype(np.int16)
    
def embedd_noise( dstfolder, split_len, noiselist, snr_val, total, completed_count, f):
    # print('Processing file', f)  
    #global lock, split_len, noiselist, snr_val
    min_snr = snr_val - 10
    max_snr = snr_val + 10
    
    data, sr = ReadAudio(f, sr=16000)
    train_val_folder = f.split('/')[split_len]
    basename = osp.basename(f)
    dirname = osp.dirname(f).split(train_val_folder)[-1]
    newpath = osp.join(dstfolder, train_val_folder + dirname, basename)
    create_directory_safe(osp.dirname(newpath))
    if train_val_folder == 'val':
        snr = snr_val
        speech_level = 80
    else:
        snr = random.randint(min_snr, max_snr)
        # snr = 20
        speech_level = random.randint(50, 80)
        # speech_level = 80
    if do_sometimes(1) or train_val_folder == 'val':
        noisepath = random.choice(noiselist)
        noise, sr = ReadAudio(noisepath, sr=16000)
        data = get_mixed_signal(data, speech_level, noise, snr)
        WriteAudio_with_int16(data, filename=newpath, sr=16000)
    else:
        shutil.copy(f, newpath)
    #lock.acquire()
    completed_count+=1
    print('Completed: {}/{}, SNR: {}'.format(completed_count, total, snr), '\r', end="")
    #lock.release()
    
def create_noise_embeddings(srcfolder, dstfolder, noisepath, snr_val):
    bg_noise_available = False
    split_len = len(srcfolder.split('/'))
    noiselist = glob.glob(osp.join(noisepath, '*'))
    
    if len(noiselist):
        bg_noise_available = True
    else:
        return False

    all_files = glob.glob(osp.join(srcfolder, '**', '*.wav'), recursive=True)
    total = len(all_files)
    completed_count = 0 
    print('Audio Embedding process running..\n')
    pool = multiprocessing.Pool()
    #l = multiprocessing.Lock()
    #lock = Lock()
    func = partial(embedd_noise, dstfolder, split_len, noiselist, snr_val , total, completed_count)
    
    print("Number of processors: ", multiprocessing.cpu_count())

    #for f in all_files:
    with Pool(20) as pool:
        pool.map(func, all_files)
        #embedd_noise(f, split_len, noiselist, snr_val )
    

if __name__ == '__main__':
    # n = generate_random_noise_signal()
    #cd = Create_Custom_DB(dbname='custom')
    #cd.add_audiosamples(num_cycles=10)
    #cd.create_val_split()
    filename = 'data/sample.wav'
    
    sr = 16000
    #filename='rec.wav'
    duration=1.5
    
    if 0:
        data = record_audio(duration=duration, sr=sr, filename=filename)
        play_audio(data,sr)
    
    if 1:
        data,sr = ReadAudio(filename)
        
        if 1:
            play_audio(data,sr)
        
        if 1:
            play_audio_sa(data,sr)
        
        if 0:
            plot_audio(data)
        
        
    
    
    
    
    
    
    
    
    
