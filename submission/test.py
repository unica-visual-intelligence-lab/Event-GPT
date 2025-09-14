import os, cv2, random, argparse
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from model import EventClassifier

NUM_FRAMES_INPUT = 16  # Number of frames to sample from each video
PATH_TO_MODEL = 'model.pth'
FPS_OUT = 4  # Output frames per second
MAX_FRAMES = 20000  # Maximum number of frames to process in a video
parser = argparse.ArgumentParser(description='Test')

parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")



args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_to_idx = {'No event': 0, 'Fire': 1, 'Smoke': 2}
model = EventClassifier(num_labels=len(class_to_idx)).to(device)
model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=device))

header = ['video_name', 'current_second', 'No_event', 'Fire', 'Smoke']


def sample_or_pad(frames):
    """ Return exactly NUM_FRAMES_INPUT frames: random sample or pad with repeats/zeros. """
    if len(frames) >= NUM_FRAMES_INPUT:
        return random.sample(frames, NUM_FRAMES_INPUT)
    pad_count = NUM_FRAMES_INPUT - len(frames)
    if frames:
        return frames + [frames[-1]] * pad_count
    return [np.zeros((224,224,3), dtype=np.uint8)] * NUM_FRAMES_INPUT

model.eval()
fps_out = FPS_OUT
print("Processing videos from:", args.videos)
print("Saving results to:", args.results)
print("Found videos:", len(os.listdir(args.videos)))
# safe version of preds = (probs > 0.5).int().cpu().tolist()
def safe_preds(probs):
    """ Convert probabilities to binary predictions, ensuring correct shape. 
    encapsulate also in try-except to handle unexpected input shapes.
    """
    #probs shape should be (1,3)

    try:
        probs_clone = probs.clone()
        detached_probs = probs_clone.detach().cpu()
        if detached_probs.dim() == 1:
            return [0] * len(class_to_idx)  # If it's a single value, return zero vector
        
        #get probs greater than 0.5
        preds = (detached_probs > 0.5).int()
        preds = preds.tolist()  # Convert to list
        #print(f"Preds shape: {len(preds)}")
        #print(f"Preds content: {preds}")

        return preds
    except Exception as e:
        print(f"Error in safe_preds: {e}")
        return   [0] * len(class_to_idx)
with torch.no_grad():
    for vid in tqdm(os.listdir(args.videos)):
        vid = os.path.join(args.videos, vid)
        df = pd.DataFrame(columns=header)
        current_row = {'video_name': vid, 'current_second': [], 'No_event': [], 'Fire': [], 'Smoke': []}
        cap = cv2.VideoCapture(vid)
        

        if not cap.isOpened(): continue


        frame_buf, prev_kvs = [], None
        idx_in, idx_out = -1, -1
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        fcount = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if fcount >= 20000: break  # Limit to MAX_FRAMES
            #print(f"Frame {fcount} read")
            if not ret: break
            idx_in += 1; fcount += 1
            out_due = int(idx_in / fps_in * fps_out)
            if out_due > idx_out:
                idx_out += 1
                frame_buf.append(frame)

            if len(frame_buf) >= NUM_FRAMES_INPUT:
                batch = sample_or_pad(frame_buf)
                frame_buf = []
            else:
                continue

            batch = [cv2.resize(f, (224,224)) for f in batch]
            preds, prev_kvs = model(batch, old_past_key_values=prev_kvs)
            # Append the sigmoid output
            probs = torch.sigmoid(preds)
            raw_preds = preds
            preds = safe_preds(probs)
            #if preds is a single value, make it a list and convert to one-hot encoding
            if isinstance(preds, int):
                empty_vector = [0] * len(class_to_idx)
                empty_vector[preds] = 1
                preds = empty_vector
                #rouding the current second to the nearest second
            current_row['current_second']= int(round(fcount / fps_in, 0))
            preds= preds[0]
            if isinstance(preds, int):
                preds = [0] * len(class_to_idx)
            #if is anything other than a list, make it a list
            if not isinstance(preds, list):
                preds =  [0] * len(class_to_idx)
            current_row['No_event'] = preds[0]
            current_row['Fire']= preds[1]
            current_row['Smoke']= preds[2]
            df = pd.concat([df, pd.DataFrame([current_row])], ignore_index=True)
            
        # Handle any remaining frames in the buffer

        if frame_buf:
            batch = sample_or_pad(frame_buf)
            batch = [cv2.resize(f, (224,224)) for f in batch]
            preds, prev_kvs = model(batch, old_past_key_values=prev_kvs)
            # Append the sigmoid output
            probs = torch.sigmoid(preds)
            raw_preds = preds
            preds =  safe_preds(probs)
            
            #if preds is a single value, make it a list and convert to one-hot encoding
            if isinstance(preds, int):
                empty_vector = [0] * len(class_to_idx)
                empty_vector[preds] = 1
            preds= preds[0]
            if isinstance(preds, int):
                preds = [0] * len(class_to_idx)
            #if is anything other than a list, make it a list
            if not isinstance(preds, list):
                preds =  [0] * len(class_to_idx)
            
            current_row['current_second'] = int(round(fcount / fps_in, 0))
            current_row['No_event'] = preds[0]
            current_row['Fire'] = preds[1]
            current_row['Smoke'] = preds[2]
            df = pd.concat([df, pd.DataFrame([current_row])], ignore_index=True)
        cap.release()
        
        DIM_KERNEL = 3  # number of consecutive zeroes required

        # Get raw 'No_event' as binary array
        no_event = df['No_event'].values  # assuming values are 0 or 1
        #if all predictions are zero, put no event to 1
        fires = df['Fire'].values  # assuming values are 0 or 1
        smokes = df['Smoke'].values  # assuming values are 0 or 1
        current_seconds = df['current_second'].values  # assuming values are seconds
        for i in range(len(no_event)):
            if no_event[i] == 0 and fires[i] == 0 and smokes[i] == 0:
                no_event[i] = 1
        
        # Find indices where fires or smokes are detected with a minimum of DIM_KERNEL consecutive ones
        fire_event_indices = []
        for i in range(len(no_event) - DIM_KERNEL + 1):
            if all(no_event[i:i + DIM_KERNEL] == 0) and (fires[i] == 1 or smokes[i] == 1):
                fire_event_indices.append(i)
        smoke_event_indices = []
        for i in range(len(no_event) - DIM_KERNEL + 1):
            if all(no_event[i:i + DIM_KERNEL] == 0) and smokes[i] == 1:
                smoke_event_indices.append(i)

        # Print the results
        print(f"Video: {vid}")
        print(f"Total frames: {fcount}, FPS input: {fps_in}, FPS output: {fps_out}")
        print(f"Total fire events detected: {len(fire_event_indices)}")
        print(f"Total smoke events detected: {len(smoke_event_indices)}")
        

        # Create results folder if needed
        if not os.path.exists(args.results):
            os.makedirs(args.results)

        # Open file to write result
        f = open(args.results + os.path.basename(vid) + '.txt', 'w')

        if len(fire_event_indices) == 0 and len(smoke_event_indices) == 0:
            print(f"No fire event detected in video {vid}.")
        else:
            first_fire_event = fire_event_indices[0] if fire_event_indices else None
            first_smoke_event = smoke_event_indices[0] if smoke_event_indices else None
            event_second = None
            if first_fire_event is not None:
                event_second = current_seconds[first_fire_event]
            elif first_smoke_event is not None:
                event_second = current_seconds[first_smoke_event]
            if event_second is None:
                print(f"No valid fire or smoke event detected in video {vid}.")
            event_second = max(event_second-FPS_OUT,0)  # Ensure the second is not negative
            print(f"Fire event detected in video {vid} at frame index {first_fire_event}, second: {event_second}")
            f.write(str(event_second))
