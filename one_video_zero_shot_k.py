import sys
sys.path.append('../')
from mreserve.preprocess import video_to_segments_zero_shot, preprocess_video, encoder, MASK
from mreserve.modeling import PretrainedMerlotReserve
import jax
import jax.numpy as jnp
import csv

# these values copied from action kitchen values 
# this is fixed for action anticipation tasks.
tau_a = 1.0
# this has to be divided by 5.
num_observed_segments = 6
time_interval = 5.0 # 2.0
tau_o = time_interval * num_observed_segments# 12.0

# This handles loading the model and getting the checkpoints.
grid_size = (18, 32)
model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)

# add eight segments of five seconds long or however long time interval is
times = []
st = 26.0
for i in range(8):
    et = st + time_interval
    times.append({'start_time': st, 'end_time': et, 'mid_time': round((st + et) / 2.0, 2)})
    st = et

video_path = 'probe_videos/youtube/baseball-highlights.mp4'
# video_path = 'pmjPjZZRhNQ.mp4'
# break video into segments length time_interval
# since we are not specifying times I do not think this will work 
video_segments = video_to_segments_zero_shot(video_path, time_interval=time_interval, times=times)

# do not use text as input, use audio?
for i in range(0,7):
    video_segments[i]['use_text_as_input'] = False
    
# for shuffling video order
#frame_holder = video_segments[0]
#video_segments[0] = video_segments[2]
#video_segments[2] = video_segments[4]
#video_segments[4] = video_segments[1]
#video_segments[1] = video_segments[5]
#video_segments[5] = video_segments[3]
#video_segments[3] = frame_holder

# not sure what this is doing 
if num_observed_segments == 6:
    video_segments[6]['frame'] *= 0
    video_segments[7]['frame'] *= 0

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

# create list of options 
#options = ['play baseball', 'watch softball', 'attend cricket', 'hail taxi', 'mow lawn', 'run track', 'plan wedding', 'explore city', 'eat burger', 'make hotdog']
options = ['hit baseball', 'catch ball', 'run home', 'throw strike', 'build snowman', 'plant tree', 'sing song', 'cut hair', 'buy hat', 'wave ax']
#options = ['hold pot', 'eat seeds', 'add rice', 'make popcorn', 'make dessert', 'watch movie', 'taste food', 'hold popcorn']
#options = ['play tennis', 'play raquetball', 'hold paddle', 'swing raquet', 'make food', 'drink water', 'hold door', 'slay dragon']
#options = ['play tennis', 'play raquetball', 'make popcorn', 'make dessert', 'swing racquet', 'drink water', 'hold popcorn', 'slay dragon']

# get list of options fom actions.csv file
action_to_ids = {}
with open('zero_shot_ek/data/ek100/actions.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for _, row in enumerate(spamreader):
        if _ == 0:
            continue
        verb, noun = row[3].split(' ')
        if ':' in noun:
            noun_split = noun.split(':')
            noun = ' '.join(noun_split[::-1])

        action = verb + ' ' + noun
        action_to_ids[action] = row[0]

ids_to_action = {v:k for k, v in action_to_ids.items()}
#options = action_list = [v for _, v in ids_to_action.items()]

# generate label space with options
label_space = model.get_label_space(options)

# dot product the <|MASK|> tokens and the options together
logits = 100.0 * jnp.einsum('bh,lh->bl', out_h, label_space)

# TODO: if action list gets longer we want to only print top results
num_to_print = 10
num_printed = 0
# loop through results 
for i, logits_i in enumerate(logits):
    #print(f"Idx {i}", flush=True)
    probs = jax.nn.softmax(logits_i, -1)
    for idx_i in jnp.argsort(-probs):
        p_i = probs[idx_i]
        if num_printed < num_to_print:
            num_printed = num_printed + 1
            print("{:.1f} {}".format(p_i * 100.0, options[idx_i], flush=True))
