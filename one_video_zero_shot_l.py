import sys
sys.path.append('../')
import tempfile, os, subprocess
from scipy.io import wavfile
from mreserve.preprocess import (
    make_spectrogram,
    video_to_segments_zero_shot,
    extract_frames_from_video,
    preprocess_video,
    encoder,
    MASK,
)
from mreserve.modeling import PretrainedMerlotReserve
import jax
import jax.numpy as jnp
import numpy as np

'''
Adapted from video_to_segments_zero_shot in mreserve.preprocess 

'''
def video_to_segments_moma(video_id, time_interval=1.0, times=None):
    """
    Load and process the video into a list of segments, each one having
        * frame
        * spectrogram
        * end_time

    """

    video_fn = f'/gpfs/loomis/project/cpsc490/cpsc490_lkt22/merlot_reserve/MOMA/MOMA-Dataset/raw_videos/{video_id}.mp4'

    ##############################################
    # 1. extract frames
    ##############################################
    frames = extract_frames_from_video(video_fn, times=[t['mid_time'] for t in times], use_multithreading=True)


    # Turn this into a unified list
    for i, (frame_i, ts_i) in enumerate(zip(frames, times)):
        ts_i['frame'] = frame_i
        ts_i['spectrogram'] = np.zeros([3, 60, 65], dtype=np.float32) # dummy zero audio
        ts_i['idx'] = i
    return times


def video_to_segments_moma_av(video_id, time_interval=1.0, times=None):
    """
    Load and process the video into a list of segments, each one having
        * frame
        * spectrogram
        * end_time

    """
    video_fn = f'/gpfs/loomis/project/cpsc490/cpsc490_lkt22/merlot_reserve/MOMA/MOMA-Dataset/raw_videos/{video_id}.mp4'
    audio_input_fn = f'/gpfs/loomis/project/cpsc490/cpsc490_lkt22/merlot_reserve/MOMA-audio/{video_id}.m4a'

    ##############################################
    # 0. Start the process for extracting audio
    ##############################################
    temp_folder = tempfile.TemporaryDirectory()
    audio_fn = os.path.join(temp_folder.name, 'audio.wav')
    ffmpeg_process = subprocess.Popen(['ffmpeg', '-y', '-i', audio_input_fn, '-ac', '1', '-ar', '22050',
                                       audio_fn], stdout=-1, stderr=-1, text=True)

    ##############################################
    # 1. extract frames
    ##############################################
    frames = extract_frames_from_video(video_fn, times=[t['mid_time'] for t in times], use_multithreading=True)

    ##############################################
    # 2. Finish extracting audio
    ##############################################
    try:
        stdout, stderr = ffmpeg_process.communicate(None, timeout=500.0)
    except subprocess.TimeoutExpired:
        ffmpeg_process.kill()
        # stdout, stderr = subprocess.TimeoutExpired.communicate()
        raise ValueError("couldnt convert in time")
    except:  # Keyboardinterrupt
        ffmpeg_process.kill()
        raise
    ffmpeg_process.kill()

    sr, waveform = wavfile.read(audio_fn, mmap=True)
    waveform = waveform.astype('float32')
    waveform /= max(np.abs(waveform).max(), 1.0)

    # Pad to max time just in case
    desired_final_frame = int(sr * max([t['end_time'] for t in times]))
    if waveform.size < desired_final_frame:
        waveform = np.concatenate([waveform, np.zeros(desired_final_frame - waveform.size, dtype=np.float32)], 0)

    # Process each segment. here i'm always using a playback_speed of 1 (aka no fast forwarding).
    spectrograms = []
    total_audio_len = sr * 5.0
    for ts_group in times:
        rest_time = 5.0 - (ts_group['end_time'] - ts_group['start_time'])
        if rest_time > 0:
            start_idx = int(sr * ts_group['start_time'])
            end_idx = int(sr * ts_group['end_time'])
            wav_ts = waveform[start_idx:end_idx]
            left_pad = int((total_audio_len - len(wav_ts)) / 2)
            right_pad = int(total_audio_len - len(wav_ts) - left_pad)            
            wav_ts = np.concatenate([np.zeros(left_pad, dtype=np.float32), wav_ts, np.zeros(right_pad, dtype=np.float32)], 0)
        else:
            start_idx = int(sr * (ts_group['mid_time']-2.5))
            end_idx = int(sr * (ts_group['mid_time']+2.5))
            wav_ts = waveform[start_idx:end_idx]
        
        spectrograms.append(make_spectrogram(wav_ts, playback_speed=1, sr=sr))
    temp_folder.cleanup()

    # Turn this into a unified list
    for i, (frame_i, spec_i, ts_i) in enumerate(zip(frames, spectrograms, times)):
        ts_i['frame'] = frame_i
        ts_i['spectrogram'] = spec_i
        ts_i['idx'] = i
    return times

def video_to_segments_moma_a(video_id, time_interval=1.0, times=None):
    """
    Load and process the video into a list of segments, each one having
        * frame
        * spectrogram
        * end_time

    """
    audio_input_fn = f'/gpfs/loomis/project/cpsc490/cpsc490_lkt22/merlot_reserve/MOMA-audio/{video_id}.m4a'

    ##############################################
    # 0. Start the process for extracting audio
    ##############################################
    temp_folder = tempfile.TemporaryDirectory()
    audio_fn = os.path.join(temp_folder.name, 'audio.wav')
    ffmpeg_process = subprocess.Popen(['ffmpeg', '-y', '-i', audio_input_fn, '-ac', '1', '-ar', '22050',
                                       audio_fn], stdout=-1, stderr=-1, text=True)
    ##############################################
    # 2. Finish extracting audio
    ##############################################
    try:
        stdout, stderr = ffmpeg_process.communicate(None, timeout=500.0)
    except subprocess.TimeoutExpired:
        ffmpeg_process.kill()
        # stdout, stderr = subprocess.TimeoutExpired.communicate()
        raise ValueError("couldnt convert in time")
    except:  # Keyboardinterrupt
        ffmpeg_process.kill()
        raise
    ffmpeg_process.kill()

    sr, waveform = wavfile.read(audio_fn, mmap=True)
    waveform = waveform.astype('float32')
    waveform /= max(np.abs(waveform).max(), 1.0)

    # Pad to max time just in case
    desired_final_frame = int(sr * max([t['end_time'] for t in times]))
    if waveform.size < desired_final_frame:
        waveform = np.concatenate([waveform, np.zeros(desired_final_frame - waveform.size, dtype=np.float32)], 0)

    # Process each segment. here i'm always using a playback_speed of 1 (aka no fast forwarding).
    spectrograms = []
    total_audio_len = sr * 5.0
    for ts_group in times:
        rest_time = 5.0 - (ts_group['end_time'] - ts_group['start_time'])
        if rest_time > 0:
            start_idx = int(sr * ts_group['start_time'])
            end_idx = int(sr * ts_group['end_time'])
            wav_ts = waveform[start_idx:end_idx]
            left_pad = int((total_audio_len - len(wav_ts)) / 2)
            right_pad = int(total_audio_len - len(wav_ts) - left_pad)            
            wav_ts = np.concatenate([np.zeros(left_pad, dtype=np.float32), wav_ts, np.zeros(right_pad, dtype=np.float32)], 0)
        else:
            start_idx = int(sr * (ts_group['mid_time']-2.5))
            end_idx = int(sr * (ts_group['mid_time']+2.5))
            wav_ts = waveform[start_idx:end_idx]
        
        spectrograms.append(make_spectrogram(wav_ts, playback_speed=1, sr=sr))
    temp_folder.cleanup()

    # Turn this into a unified list
    for i, (spec_i, ts_i) in enumerate(zip(spectrograms, times)):
        ts_i['frame'] = np.zeros([192, 168, 3], dtype=np.float32) # dummy zero audio
        ts_i['spectrogram'] = spec_i
        ts_i['idx'] = i
    return times

#### BEGIN ACTUAL SCRIPT

zero_shot_flavor = sys.argv[1].lower()
video_id = sys.argv[2]

# create list of options 
# options_file = '/gpfs/loomis/project/cpsc490/cpsc490_lkt22/merlot_reserve/MOMA/MOMA-Dataset/annotations/anns/sact_cnames.txt'
# options_file = '/gpfs/loomis/project/cpsc490/cpsc490_lkt22/merlot_reserve/MOMA/MOMA-Dataset/annotations/anns/act_cnames.txt'
# with open(options_file) as f:
#     options = [opt.strip() for opt in f.readlines()]
options = ['clean table', 'grind coffee', 'drink coffee', 'go hiking', 'play ping pong', 'brush teeth', 'play tennis', 'play badminton', 'play soccer', 'play table tennis']

# This handles loading the model and getting the checkpoints.
grid_size = (18, 32)
model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)

# these values copied from action kitchen values 
# this is fixed for action anticipation tasks.
tau_a = 1.0
# this has to be divided by 5.
num_observed_segments = 6
time_interval = 5.0 # 2.0
tau_o = time_interval * num_observed_segments# 12.0

# add eight segments of five seconds long or however long time interval is
times = []
# st = 96.806
st = 0
for i in range(8):
    et = st + time_interval
    times.append({'start_time': st, 'end_time': et, 'mid_time': round((st + et) / 2.0, 2)})
    st = et

print(times)

# break video into segments length time_interval
# since we are not specifying times I do not think this will work 
if zero_shot_flavor == 'moma-v':
    video_segments = video_to_segments_moma(video_id, time_interval=time_interval, times=times)
elif zero_shot_flavor == 'moma-av':
    video_segments = video_to_segments_moma_av(video_id, time_interval=time_interval, times=times)
elif zero_shot_flavor == 'moma-a':
    video_segments = video_to_segments_moma_a(video_id, time_interval=time_interval, times=times)
else:
    video_path = video_id
    video_segments = video_to_segments_zero_shot(video_path, time_interval=time_interval, times=times)

# do not use text as input, use audio?
for i in range(0,7):
    video_segments[i]['use_text_as_input'] = False

# not sure what this is doing 
# if num_observed_segments == 6:
#     video_segments[6]['frame'] *= 0
#     video_segments[7]['frame'] *= 0

# setting up the last frame as the masked token 
# we do want to use text as input for the last frame
video_segments[7]['text'] = '<|MASK|>'
video_segments[7]['use_text_as_input'] = True

# preprocess video 
video_pre = preprocess_video(video_segments, output_grid_size=grid_size, verbose=False)

# Now we embed the entire video and extract the text. result is  [seq_len, H]. we extract a hidden state for every
# MASK token
out_h = model.embed_video(**video_pre)
out_h = out_h[video_pre['tokens'] == MASK]

label_space = model.get_label_space(options)

# dot product the <|MASK|> tokens and the options together
logits = 100.0 * jnp.einsum('bh,lh->bl', out_h, label_space)

# TODO: if action list gets longer we want to only print top results
# loop through results 
for i, logits_i in enumerate(logits):
    #print(f"Idx {i}", flush=True)
    probs = jax.nn.softmax(logits_i, -1)
    for idx_i in jnp.argsort(-probs):
        p_i = probs[idx_i]
        print("{:.1f} {}".format(p_i * 100.0, options[idx_i], flush=True))
