from operator import truediv
import warnings
warnings.simplefilter(action='ignore')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
graph = tf.get_default_graph()
from flask import Flask, render_template, request, redirect, session, url_for, flash
import csv
import pandas as pd
import numpy as np
import os
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import joblib
from sklearn import preprocessing
from keras.models import load_model
import io
import sed_vis
import dcase_util
from pydub import AudioSegment
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
# flask need some secret key to use session
app.secret_key = "b'\xcapZ\x14~`\xb9\x8e\xa3\xa9\xa5>'"

PATH_MODEL = 'sed_128_2_Adam_model.h5'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
model = load_model(PATH_MODEL)

sr = 48000           # sampling rate of the incoming signal
nfft = 2048          # number of fast Fourier transform (FFT) component
win_len = nfft       # windows of length
hop_len_s = 0.02     # hop length of 20ms
hop_len = int(sr * hop_len_s) # 960
frame_res = sr / float(hop_len)
frames_1_sec = int(frame_res) # 50 frames in a second, 50 x 60s = 3000 frames
nb_mel_bands = 40    # number of Mel bands to generate
seq_len = 128
nb_ch = 2

__class_labels = {
    'knock': 0,
    'drawer': 1,
    'clearthroat': 2,
    'phone': 3,
    'keysDrop': 4,
    'speech': 5,
    'keyboard': 6,
    'pageturn': 7,
    'cough': 8,
    'doorslam': 9,
    'laughter': 10
}

@app.route('/predict-eval', methods = ['GET', 'POST'])
def predictEval():
    error_notFound = ''
    error_NotMatch = ''
    target_audio = os.path.abspath(os.path.join(BASE_DIR , "static/audio"))
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            error_notFound = "Upload the audio first"
            return render_template('home.html' , error=error_notFound)
        elif file and audio_ext_validator(file.filename):
            filename = file.filename
            audio_filename = os.path.basename(filename)
            file.save(os.path.join(target_audio , filename))
            AUDIO_PATH = os.path.join(target_audio , filename)
            if audio_ch_validator(AUDIO_PATH):
                predictions = predict(AUDIO_PATH, model)
                json_df = predictions.to_json()

                plt_path = os.path.abspath(os.path.join(BASE_DIR, "static/plt", audio_filename))
                wave_png_path = plt_path + "_wave.png"
                wave_filename = os.path.basename(wave_png_path)
                mbe_png_path = plt_path + "_mbe.png"
                mbe_filename = os.path.basename(mbe_png_path)

                vis_png_path = visualize_result(AUDIO_PATH)
                vis_filename = os.path.basename(vis_png_path)

                session['audio_filename'] = audio_filename
                session['wave_filename'] = wave_filename
                session['mbe_filename'] = mbe_filename
                session['vis_filename'] = vis_filename
                session['predictions'] = json_df
            else:
                error_NotMatch = "The audio has to be 2 channels"
                os.remove(AUDIO_PATH)
        else:
            error_NotMatch = "The audio is not match with the requirement (wav extension only)"
        if(len(error_NotMatch) == 0):
            return redirect(url_for('success', filename=audio_filename, wave_img=wave_filename, mbe_img=mbe_filename, vis_img=vis_filename, tables=json_df))
        else:
            return render_template('home.html' , error=error_NotMatch)
    return render_template('home.html')

@app.route("/success")
def success():
    audio_filename = session.get('audio_filename')
    wave_filename = session.get('wave_filename')
    mbe_filename = session.get('mbe_filename')
    vis_filename = session.get('vis_filename')
    json_df = session.get('predictions')
    predictions = pd.read_json(json_df)
    return render_template('success.html', filename=audio_filename, wave_img=wave_filename, mbe_img=mbe_filename, vis_img=vis_filename, tables=[predictions.to_html()])

def audio_ext_validator(filename):
    ext = filename.split(".")[-1]
    valid = ext in ["wav"]
    return valid

def audio_ch_validator(file_path):
    audio = wave.open(file_path, "r")
    num_channels = int(audio.getnchannels())
    return num_channels == 2

def predict(filename, model):
    y, sr = load_audio(filename, mono=False, fs=48000)
    audio_filename = os.path.basename(filename)
    csv_path  = "./static/results/{}_result.csv".format(audio_filename)
    fid = open(csv_path, 'w', newline='')
    fid_writer = csv.writer(fid, delimiter=';')
    # Extract features
    mbe = None
    for ch in range(y.shape[0]):
        mbe_ch = extract_mbe(y[ch, :], sr, nfft, hop_len, nb_mel_bands).T
        if mbe is None:
            mbe = mbe_ch
        else:
            mbe = np.concatenate((mbe, mbe_ch), 1)

    plt_path = os.path.abspath(os.path.join(BASE_DIR, "static/plt", audio_filename))
    wave_png_path = plt_path + "_wave.png"
    fig_wave = plt.Figure()
    ax_wave = fig_wave.add_subplot(111)
    librosa.display.waveshow(y, sr=sr, ax=ax_wave)
    fig_wave.savefig(wave_png_path)

    mbe_png_path = plt_path + "_mbe.png"
    fig_mbe = plt.Figure()
    ax_mbe = fig_mbe.add_subplot(111)
    db_mbe = librosa.amplitude_to_db(mbe_ch.T, ref=np.max)
    librosa.display.specshow(db_mbe , sr=sr, hop_length=hop_len, x_axis='time', y_axis='linear', ax=ax_mbe)
    fig_mbe.savefig(mbe_png_path)

    normalized_features_wts_file = os.path.abspath(os.path.join(BASE_DIR , "static/scaler_stereo.save"))
    spec_scaler = joblib.load(normalized_features_wts_file)

    # Scale features
    feat = spec_scaler.transform(mbe)

    # Split into sequencse and multichannels
    feat = split_in_seqs(feat, seq_len)
    feat = split_multi_channels(feat, nb_ch)
    with graph.as_default():
        pred_file = model.predict(feat)
    pred_file_threshold = (pred_file > 0.5).astype(int)
    pred_file_threshold = reshape_3Dto2D(pred_file_threshold)

    fid_writer.writerow(["start_time", "end_time", "labels"])
    for col in range(pred_file_threshold.shape[1]):
    # For each class, identify all onset-offset times of its instances
        region_list = contiguous_regions(pred_file_threshold[:, col])
        for region in region_list:
            # Only 2 decimal after comma
            onset = "{:.2f}".format(region[0]*hop_len_s)
            offset = "{:.2f}".format(region[1]*hop_len_s)

            #Write predict into CSV file
            fid_writer.writerow([onset, offset, list(__class_labels.keys())[col]])
    fid.close()
    df = pd.read_csv(csv_path, sep=";")
    df_sort = df.sort_values(by="start_time", ascending=True)
    prediction_result = df_sort.reset_index(drop=True)
    prediction_result.to_csv(csv_path, index=False, header=True)
    return prediction_result


def visualize_result(filename):
    audio_file = os.path.basename(filename)
    target_mono = os.path.abspath(os.path.join(BASE_DIR , "static/audio"))
    output_mono = os.path.join(target_mono, '{}_mono.wav'.format(audio_file))
    output_estimated = "./static/results/{}_result.ann".format(audio_file)

    sound = AudioSegment.from_wav(filename)
    sound = sound.set_channels(1)
    sound.export(output_mono, format="wav")

    csv_path  = "./static/results/{}_result.csv".format(audio_file)

    df_estimated = pd.read_csv(csv_path , sep=",")
    df_estimated.to_csv(output_estimated, index=None, header=False, sep='\t')

    # Load audio signal first
    audio_container = dcase_util.containers.AudioContainer().load(output_mono)

    # Load event lists
    estimated_event_list = dcase_util.containers.MetaDataContainer().load(output_estimated)

    event_lists = {'estimated': estimated_event_list}

    # Visualize the data
    vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,
                                                    audio_signal=audio_container.data,
                                                    sampling_rate=audio_container.fs,
                                                    publication_mode=True)
    plt_path = os.path.abspath(os.path.join(BASE_DIR, "static/plt", audio_file))
    vis_png_path = plt_path + "_vis.png"

    vis.save(vis_png_path)
    return vis_png_path

## Utils functions
# function for load the audio file
def load_audio(filename, mono=True, fs=44100):
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None

# function for feature extraction
def extract_mbe(_y, _sr, _nfft, _hop_len, _nb_mel):
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_hop_len, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    return np.log(np.dot(mel_basis, spec))

# function for reshape array from 3D to 2D
def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])

# functions that is used for preprocessing data before feed the model
def split_in_seqs(data, subdivs):
    if len(data.shape) == 1:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, 1))
    elif len(data.shape) == 2:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1]))
    elif len(data.shape) == 3:
        if data.shape[0] % subdivs:
            data = data[:-(data.shape[0] % subdivs), :, :]
        data = data.reshape((data.shape[0] // subdivs, subdivs, data.shape[1], data.shape[2]))
    return data

def split_multi_channels(data, num_channels):
    in_shape = data.shape
    if len(in_shape) == 3:
        hop = in_shape[2] // num_channels
        tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
        for i in range(num_channels):
            tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
    else:
        print("ERROR: The input should be a 3D matrix but it seems to have dimensions ", in_shape)
        exit()
    return tmp

def contiguous_regions(activity_array):
    # Find the changes in the activity_array
    change_indices = np.diff(activity_array).nonzero()[0]

    # Shift change_index with one, focus on the frame after the change.
    change_indices += 1

    if activity_array[0]:
      # If the first element of activity_array is True add 0 at the beginning
      change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
      # If the last element of activity_array is True, add the length of the array
      change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))

@app.route('/')
def index():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)