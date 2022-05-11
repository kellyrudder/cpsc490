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
import json
import csv

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

def load_json():
    save_path = '/gpfs/loomis/project/cpsc490/cpsc490_lkt22/merlot_reserve/MOMA/MOMA-Dataset/annotations/anns/video_anns.json'
    with open(save_path, "r") as f:
        return json.load(f)
        
def add_to_csv(outfile_path, video_id, start_time, ground_truth, gt_ranking, gt_score):
    f = open(outfile_path, 'a')
    writer = csv.writer(f)
    
    # add row to csv file
    writer.writerow([video_id, start_time, ground_truth, gt_ranking, gt_score])
    
    f.close()

def add_csv_header(outfile_path):
    f = open(outfile_path, 'w')
    writer = csv.writer(f)
    
    # add header
    writer.writerow(['video_id', 'start_time', 'ground_truth', 'gt_ranking', 'gt_score'])
    
    f.close()

#### BEGIN ACTUAL SCRIPT

zero_shot_flavor = sys.argv[1].lower()
outfile_path = f'lance_experiment.{zero_shot_flavor}.csv'

add_csv_header(outfile_path)

video_anns = load_json()

# create list of options 
options_file = '/gpfs/loomis/project/cpsc490/cpsc490_lkt22/merlot_reserve/MOMA/MOMA-Dataset/annotations/anns/sact_cnames.txt'
#options_file = '/gpfs/loomis/project/cpsc490/cpsc490_lkt22/merlot_reserve/MOMA/MOMA-Dataset/annotations/anns/act_cnames.txt'
with open(options_file) as f:
    options = [opt.strip() for opt in f.readlines()]
# options = ['clean table', 'grind coffee', 'drink coffee', 'go hiking', 'play ping pong', 'brush teeth', 'play tennis', 'play badminton', 'play soccer', 'play table tennis']
#options = ['babysitting', 'barber service', 'coronation', 'dining service', 'group excercise', 'hospital service', 'hotel service', 'instructing student(s)', 'marriage proposal', 'making a presentation', 'making a transaction', 'playing boardgames', 'playing ball games', 'playing frisbee', 'salon and spa service', 'security screening', 'riding a bike']
#print(options)

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

video_ids = ['ADzHI5-IRxw', 'sP5SmDftmQo', 'o_4N6grOqg4', '7NdW-cCSMCQ', 'xW7auZjGTGk', 'f6DNByuheFM', 'rwSFVVInHtU', 'ZK2QWk0QZEs', 'bsIIpEoghJY', '1eMlUb9vnZk', 'q5NC7eI_KMw', 'cOlsNR6Nvsw', 'Q9Wb0sc5Ft8', 'Ny2S4YURXBA', 'urcmaFVQnF4', 'tGNuv-_Gulc', '63jl5F82OmI', 'FQAGEY4greA', 'jtaP7IFhUvw', 'QT5vlkGM810', 'P5h1sIAvQqU', 'avjhKvfcNt8', 'zSKpxB3rOB8', 'hexthue-KoA', 'woRcSIidK0I', '6-yc-IHhTOE', 'Kkj6y9TtnRM', '0jwhe-zdwcg', 'qwa_QJD4UcU', 'Nk9KYtLzWeI', 'NHhKKOQrstk', '1HBBE_Uz_9U', 'fv-EGsNvNzc', 'oPZMZJBFYEU', 'Yl_FJAOcFgQ', 'yNMqbL8B224', 'lgrz-QBrqqc', 'Bt51tCbV7TQ', 'pU7VLIIog5E', 'uNHsZMSnyE8', 'j6UyNq_TwGA', 'kYCRY9mjL9s', 'IXl2SPerSZ4', 'DnfoqsLwzVQ', 'IqkLiAHYHr8', 'MIEneFA508o', 'uqiMw7tQ1Cc', '5pkm_SYkcbI', 'jDOG8Vk9OOQ', 'Mt54VpVip-Y', 'DX-FIfLb19A', 'Sgn6Os4YSW0', 'taBGfxFQZr4', 'W4SM_0s5Bro', 'WsZ6NEijlfI', '6EqI5V8AUp8', 'PQ_dGsSBpvA', 'dmaIyXwzKlo', '1MAJE6fHISY', 'Sou1PwEqPH8', 'csjlN8-282g', 'PewZ-F1jFkA', 'm_b23McIYs0', 'qVDPNpFw7Hs', 'Sx5-QS0jHDM', 'jkXPQ4cxliw', 'ZghaYwUtkOE', 'QMBAc13_V74', 'XVwEkzEmDCc', '03OPcn80Uzs', 'dZW6J4bKPG8', '6iZ4ekgV0h0', 'ADGXpBO1bhI', 'ULczFFENOy8', '5Oqa1Fag5eQ', 'p_Og7_L2m-s', 'imPQ3lysWnU', 'VTFLQEKLG44', 'eU2ihJW5tyE', 'u2qCGsWoRSc', 'mOt9-6q29sQ', 'V8eLdbKXGzk', 'vc4hG_8t9nA', '3grFZJ7qzjk', 'fz10rpnrmQo', 'QBCwXulQJ8c', '-STH2nfCIk8', 'Qki2u8aKTCM', 'OzZhMlNe01I', '66jMNxcw0A4', 'ylA5EjfEXU8', 'riwvTctlQTM', 'GvioWi5cg4k', '61RjiFidiTw', '0hpX2WK130s', 'XBB2HrS5CXc', 'ZpAlOwKu8fg', 'wqmzwVrkTU4', 'qQat1ohBpVA', 'cIuGwaBFX8I', 'oFsNrG3WI3U', 'AB2GDn6O-wQ', 'u9pi-O_0TVM', 'KpllAjxOIUU', 'jkhA42kcYMA', '4BTb8fEA8K4', 'K5T42WfUcnM', '7bDfcqHnMPw', 'XIeUV11mTt8', '3BGYn6AYdd8', 'DvQ982Cw4uw', 'dMae8xZll3A', 'B5iTAJDC7-w', 'BR9zq5xrTF0', 'XNgOISE1Jxw', 'pKxgfkIlFrM', 'g7M9hCYlLXg', '3GDwHI5hAbM', '5E3EbV0wG0U', '4dIr_cY03Js', 'uhd5GVXBGoM', 'yrFQCOcTlFY', 'RtpRuXnIfXQ', 'ZuyMxKpGKTo', 'fScIX4mkKdo', 'eIqBFOX5YTo', 'lvi3kstJNWw', 'TP1nghTPnKQ', 'dcTddt9VcQw', 'aPxpuwl2IrI', 'IpObqdrS7i0', 'xpSm9bt9Uqc', 'mGwHo3A9hF4', 'MwJjtutSO9Y', 't4YXfwEa4hM', 'zNZ0JLHGxAY', 'rByTJDgBz5g', '-bBJIR6TYF8', 'bBQ_w1TEVi8', 'Pq_k6J7jmu0', 'sAzW9Y6RIEc', 'XC78OR1Hr0E', 'YdB1HMCldJY', 'mfwVN77OTN0', 'iM7imKiDSHg', 'ekYpxgEpLZI', 'inKSvLk7kiI', 'UDIvWdwVZMg', 'Bt-R5UWOGuo', 'lgUyW5aGC2U', '3ZrurhxII-M', 'X1Fbp5WIXCE', 'Cq7JT2zKUSA', '8Ht7rJk1U3Q', 'AW2hoXw8x6E', 'WKGAs9uRlsI', 'Cx9YTJInZ30', '4y21uwFUgkE', '0SLGvlDVkRA', 's4TEW7g753Q', '8n1bxuPWBnQ', 'MqnwOuBns68', 'UPgqo19V7y0', 'lNWHS9Kok8Q', 'DN2RyN5crYE', 'bbtfIzh5Db4', 'mBe3WqWTZDE', '3pQqFQMI0Sc', 'f-W522lJtBQ', 'Cvlk2wgVWPs', 'Pal0AcfUDy0', 'n3udMsz8bic', 'EkKOB8OxEuI', 'dHCLDtSNIQA', 'ym8GquLESdU', 'a6pJPzmwRmc', 'QB_5n-fhthU', 'Egy6I9XB5u8', '6kU_miFjids', 'sloQbuYOmGo', '1cUp0UIVbMw', 'SFrc1EvQ80k', 'Pj8zn7lSZFA', 'N8ULvtu5mpA', 'qr58ALUFa28', '2b3WaNLkmys', 'Z-6erGcT-qw', 'D6VQDNIZH7U', 'eZI12l1MEQQ', 'acmjfct3T1U', 'KOdxrn7e4M8', 'QXzoJRVk5wI', '3DmkDCnV3jg', 'n2nfelAeLGU', 'ZWU-2O2xeiE', '8_HFyosIeAg', 'v1_mJEtfk-c', '06T40MI75Nk', 'MMvEDIHfDlc', 'Jbu1v0RQe6M', 'Qs_sbBxYHCE', 'VnEdtDq9nOI', '5iqYTMEH5TA', 'FKbJsPPoAxw', 'QTrqGwj3XpA', 'l0G0ZO627jo', 'HhUays2ehyI', 'Sd2WaOI93bM', 'RpxegTIqaDE', 'Y83BauQCdG8', '38df09Oa5aY', 'gPY-jQwu2m0', '0Q3IvgfzeMM', 'tcmEvvWC8A0', 'j9G5mPOk8YE', 'sNS67ZJQVDA', '3oM-sNFpvjc', 'NlwmO2m02Iw', 'mOlQg5THqhE', 'kKXCHsd7cWs', 'GZVD0gQNf-I', 'otGws4tXmtY', 'YRBEIF_g2YI', 'SUqSWmeEexg', 'JRFoLTUsoK4', 'GsvToVVwzSk', '0M1QrMIa7cg', '19ueOWcwVNs', 'FXj6ELi4BVg', 'H7YYKi2qaxQ', '40n56DmQ7L4', 'aEMbPiYq6oA', 'm9TEdeOTgAE', 'gT1ALBLPUm4', 'pDv2MiCsC2k', 'C7FmHROuB_o', '7LVMKNsk88Q', '8vJnhDFwXS0', '2AiOgOnvYvo', 'cJLxGSR5eCU', '5c4UIDf7pzc', 'YDZqpg0cmtU', 'Lw40a0opKGY', 'azU6upenuJY', 'Iwpi1Lm6dFo', 'WQq1ojubvkY', 'iHOCbRSbrqU', 'l7EDyUg-LqE', 'vsBPjaLy4fs', 'BnuKKbGe124', '9EvnLgxpYqY', 'E6twzRRJEkY', 'xpyrefzvTpI', '6OlPHcmbSos', 'T-_GTt-msfg', 'wyqfYJX23lg', 'Ir0-y3dg3HU', '-0brhBye9gY', 'Zku1tCcztkA', 'PTGV-jv3mPY', 'Ew54eu7e3fs', 'd8SK35XYEVU', 'iVpaczaAIuQ', '0gafzS6o84M', 'ynDyv6LDRTM', 'Rkthf1_yOj8', 'X7t6ZzNVTDA', 'iajYa15YcDE', 'Spwg5KkBvHE', '_AeBSDuyXUo', 'vjPjausO-KI', 'g0eY0nTJuaA', 'dGXHxnRca-Q', 'RULU1m9_008', 'EPtM0NaTZ7I', 'NQjXuJoA62c', '0YvYn75WjJQ', 'f4KV7APYUO0', 'mERHrv97PIM', 'YkXVy-FAz-U', 't-JbV39qe1M', '7-ohT9KKezY', 'gKhHGFrVQek', 'ALv5HWicU20', '7SEst7XLNIU', 'X82de9Mc6a0', 'O1xwTOddDV8', 'K2C-Qj7r-GU', '51VupvTTSno', 'obEYhjTAFE0', '7qM-DRn10SE', 'vXKFg15o-to', 'TmQua7gUGQA', '_gvvlBYeiZw', '-jVXNEvUJss', 'lNQexzgzvw4', 'VQoMkvvqsgA', 'P2dAsl9qojY', 'QEYVU4VW7wI', 'QmtPlbxB7rA', 'imxo-gi-aLY', 'rwkBvz_i870', 'zkGr_dDIa-0', 'u3wQgc6eS9k', '9xUnQp4xP2I', 'HrTJKybzpxo', 'BoGi6JUwbfI', 'AYVNf_IGqTE', 'g6U6EC_ivBo', 'A9wTx2Pq6L4', 'KlGodrW9Kj4', 'A6NYrdheBfU', 'pSNUJU1QlNk', 'hOFI_ftLMyY', 't_-lPLpbPoM', 'dVSyiwKRpE4', '6PET-GvhFx0', 'mBtay3_Ux7c', '42lBjrzTXj8', 's_SlN1r0_tk', 'cYKOWdiINZI', 'S8r7h-WgEDs', 'SpK1owmRemQ', 'prThf-aAmpo', 'cErvWqz8jWE', '5ciccElb4KU', 'p6SNCvIN4EI', 'W-Ksdkh6j4k', 'CGcVy0dLF3o', 'Wo-lFBv4I9s', 'oquXSDpTegw', 'IJt0_IKPHww', 'z_Xka42e9jE', 'jnTotsG6fBI', '3u26zxtPF5A', '1CKFDu-IMLM', 'EJXMF8h5e2Q', 'iKLoQ_-JKLo', 'Uoncl8ype-M', 'AA6nXQMyfhU', 'GNqrY6yDGsg', 'kgt95I-tAQ0', 'UhxuncuRWPU', 'hkudK8drvh8', 'tTymZqKG2vM', 'E8jrFkPJ4II', 'dLqpkyc9vhg', 'YkMMqOUNyKk', 'nXW0mWcS1II', '8SZJO1Jhr4o', 'FBEoHtVFdMg', '5ww_TetJXiE', 'W9Y9A1-b1bU', 'NkOxfoVOl2g', 'ViTkDmRvWGE', 'wbo3KFFK9K8', 'FME7qDYPnyk', 'NrmMk1Myrxc', 'K3V8qEshV8k', 'ADkm4qkXAj8', 'VTl-q2_1h2w']
for video_id in video_ids[:30]:

    try:
    
        # get json data
        json_data = load_json()
        # get list of subactivities
        video_subactivities = json_data[video_id]['subactivity']
        
        # loop through video's subactivities performing prediction on each 
        for activity in video_subactivities:
            # if not enough time for activity, continue to next activity 
            if activity['end'] < 40.0:
                continue

            # add eight segments of five seconds long or however long time interval is
            times = []
            # the start time is the end of the activity's completion - 40 seconds 
            st = activity['end'] - 40.0
            for i in range(8):
                et = st + time_interval
                times.append({'start_time': st, 'end_time': et, 'mid_time': round((st + et) / 2.0, 2)})
                st = et

            # break video into segments length time_interval
            # since we are not specifying times I do not think this will work 
            if zero_shot_flavor in ['moma-v', 'moma-scramble', 'moma-scramble2']:
                video_segments = video_to_segments_moma(video_id, time_interval=time_interval, times=times)
            elif zero_shot_flavor in ['moma-av', 'moma-a-scramblev', 'moma-a-scramblev2']:
                video_segments = video_to_segments_moma_av(video_id, time_interval=time_interval, times=times)
            elif zero_shot_flavor == 'moma-a':
                video_segments = video_to_segments_moma_a(video_id, time_interval=time_interval, times=times)
            else:
                video_path = video_id
                video_segments = video_to_segments_zero_shot(video_path, time_interval=time_interval, times=times)

            # do not use text as input, use audio?
            # we may want this, not sure. TODO: revisit
            for i in range(0,7):
                video_segments[i]['use_text_as_input'] = False
                
            # scramble frame order
            if zero_shot_flavor == 'moma-scramble':
                frame_holder = video_segments[4]
                video_segments[4] = video_segments[7]
                video_segments[7] = video_segments[2]
                video_segments[2] = video_segments[0]
                video_segments[0] = video_segments[5]
                video_segments[5] = video_segments[1]
                video_segments[1] = video_segments[3]
                video_segments[3] = video_segments[6]
                video_segments[6] = frame_holder
            elif zero_shot_flavor == 'moma-scramble2':
                frame_holder = video_segments[0]
                video_segments[0] = video_segments[5]
                video_segments[5] = video_segments[2]
                video_segments[2] = video_segments[7]
                video_segments[7] = video_segments[6]
                video_segments[6] = video_segments[1]
                video_segments[1] = video_segments[4]
                video_segments[4] = video_segments[3]
                video_segments[3] = frame_holder
            elif zero_shot_flavor == 'moma-a-scramblev':
                perm = [2, 5, 7, 1, 6, 0, 3, 4]
                video_frames = [seg['frame'] for seg in video_segments]
                for i in range(7):
                    video_segments[i]['frame'] = video_frames[perm[i]]
            elif zero_shot_flavor == 'moma-a-scramblev2':
                perm = [3, 6, 5, 4, 1, 0, 7, 2]
                video_frames = [seg['frame'] for seg in video_segments]
                for i in range(7):
                    video_segments[i]['frame'] = video_frames[perm[i]]

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
    
            ground_truth = activity['class']
            print(f"VIDEO: {video_id}, gt: {ground_truth}")
    
            # loop through results 
            for i, logits_i in enumerate(logits):
                #print(f"Idx {i}", flush=True)
                probs = jax.nn.softmax(logits_i, -1)
                rank_counter = 1
                for idx_i in jnp.argsort(-probs):
                    p_i = probs[idx_i]
                    print("{:.1f} {}".format(p_i * 100.0, options[idx_i]))
                    # if this is the ground truth, save its rank and numeric value 
                    if options[idx_i] == ground_truth:
                        add_to_csv(outfile_path, video_id, st, ground_truth, rank_counter, p_i * 100.0)
                    rank_counter += 1
                #print('-------------------------------------------------------------')
                    

    except Exception as e:
        print(f"error on video: {video_id}")
        print(e.message)
        print(e.args)
