import os, cv2, random, argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
#get also f1    
from sklearn.metrics import f1_score
import albumentations as A
from helpers import corresponding_label_to_video, get_rtf_text
from model import EventClassifier
from collections import deque

NUM_FRAMES_INPUT = 16
LIMIT_TRAINING_FRAMES = 10000

def parse_args():
    parser = argparse.ArgumentParser(description='Train on fire videos')
    parser.add_argument("--videos", type=str, default='E:/2025_ICIAP_FIRE/dataset', help="Dataset folder")
    parser.add_argument("--labels", type=str, default='E:/2025_ICIAP_FIRE/GT', help="Labels folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=1, help="Output FPS for processing")
    parser.add_argument("--output", type=str, default='output', help="Output folder for results")
    parser.add_argument("--use_augmentations", type=bool, default=True, help="Use data augmentations during training")
    return parser.parse_args()


def load_and_split(videos_path, labels_path, seed, split_ratio=0.8):
    vids, lbls = corresponding_label_to_video(videos_path, labels_path)
    combined = list(zip(vids, lbls))
    random.seed(seed); random.shuffle(combined)
    vids, lbls = zip(*combined)
    idx = int(len(vids) * split_ratio)
    return vids[:idx], lbls[:idx], vids[idx:], lbls[idx:]


def sample_or_pad(frames):
    """ Return exactly NUM_FRAMES_INPUT frames: random sample or pad with repeats/zeros. """
    if len(frames) >= NUM_FRAMES_INPUT:
        return random.sample(frames, NUM_FRAMES_INPUT)
    pad_count = NUM_FRAMES_INPUT - len(frames)
    if frames:
        return frames + [frames[-1]] * pad_count
    return [np.zeros((224,224,3), dtype=np.uint8)] * NUM_FRAMES_INPUT
def train_augmentations():
    return A.Compose([
       
        #A.HorizontalFlip(p=0.5),
        #A.RandomRotate90(p=0.5),
        #A.RandomCrop(224, 224, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        #more light augmentations
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise( p=0.2),
        
    ], p=1.0,
    additional_targets={
        'image_i1': 'image',
        'image_i2': 'image',
        'image_i3': 'image',
        'image_i4': 'image',
        'image_i5': 'image',
        'image_i6': 'image',
        'image_i7': 'image',
        'image_i8': 'image',
        'image_i9': 'image',
        'image_i10': 'image',
        'image_i11': 'image',
        'image_i12': 'image',
        'image_i13': 'image',
        'image_i14': 'image',
        'image_i15': 'image',
    })


def train_one_epoch(model, optimizer,  train_videos, train_labels, 
                    class_to_idx, fps_out, device, global_stats,use_augmentations=True):
    model.train()
    pbar = tqdm(zip(train_videos, train_labels), total=len(train_videos), desc="Training", unit="video")
    for vid, lbl in pbar:
        cap = cv2.VideoCapture(vid)
        if not cap.isOpened(): continue

        timestart, cls_event = get_rtf_text(lbl)
        
        base_class_idx = []
        for cls in cls_event:
            if cls in class_to_idx:
                base_class_idx.append(class_to_idx[cls])
            else:
                print(f"Warning: Class '{cls}' not found in class_to_idx. Using 'No event' instead.")
                base_class_idx.append(class_to_idx["No event"])

        frame_buf, prev_kvs = [], None
        fcount, idx_in, idx_out = 0, -1, -1
        fps_in = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened() and fcount < LIMIT_TRAINING_FRAMES:
            ret, frame = cap.read()
            if not ret: break
            idx_in += 1; fcount += 1
            global_stats['frames'] += 1

            out_due = int(idx_in / fps_in * fps_out)
            if out_due > idx_out:
                idx_out += 1
                frame_buf.append(frame)

            if len(frame_buf) >= NUM_FRAMES_INPUT:
                batch = sample_or_pad(frame_buf)
            else:
                batch = None

            if batch is not None:
                cls_idx = base_class_idx
                if (fcount / fps_in) < timestart:
                    cls_idx = class_to_idx["No event"]
                if use_augmentations:
                    aug = train_augmentations()
                    #apply the same augmentation to all frames in the batch
                    images = aug(image=batch[0], image_i1=batch[1], image_i2=batch[2],
                                 image_i3=batch[3], image_i4=batch[4], image_i5=batch[5],
                                 image_i6=batch[6], image_i7=batch[7], image_i8=batch[8],
                                 image_i9=batch[9], image_i10=batch[10], image_i11=batch[11],
                                 image_i12=batch[12], image_i13=batch[13], image_i14=batch[14],
                                 image_i15=batch[15])
                    batch = [images[f'image_i{i}'] for i in range(1,NUM_FRAMES_INPUT)]
                    #add the first frame to the batch
                    batch.insert(0, images['image'])
                    
                else:
                    batch = [cv2.resize(f, (224,224)) for f in batch]
                target = torch.tensor(cls_idx, dtype=torch.long).unsqueeze(0).to(device)
                #to one hot
                target = torch.nn.functional.one_hot(target, num_classes=len(class_to_idx)).float()
                if target.dim() > 2:
                    target = target.squeeze(0)
                #sum on the batch
                target = target.sum(dim=0, keepdim=True)
                preds, prev_kvs, loss = model(batch, old_past_key_values=prev_kvs, labels =target)
                prev_kvs = prev_kvs.detach()

                preds =torch.sigmoid(preds)
                #print(f"Preds: {preds}, Target: {target}")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                global_stats['losses'].append(loss.item())
                frame_buf = []

                avg_loss = np.mean(global_stats['losses']) if global_stats['losses'] else 0.0
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'frames': global_stats['frames']})

        if frame_buf:
            batch = sample_or_pad(frame_buf)
            cls_idx = base_class_idx
            if (fcount / fps_in) < timestart:
                cls_idx = class_to_idx["No event"]
            if use_augmentations:
                aug = train_augmentations()
                #apply the same augmentation to all frames in the batch
                images = aug(image=batch[0], image_i1=batch[1], image_i2=batch[2],
                             image_i3=batch[3], image_i4=batch[4], image_i5=batch[5],
                             image_i6=batch[6], image_i7=batch[7], image_i8=batch[8],
                             image_i9=batch[9], image_i10=batch[10], image_i11=batch[11],
                             image_i12=batch[12], image_i13=batch[13], image_i14=batch[14],
                             image_i15=batch[15])
                batch = [images[f'image_i{i}'] for i in range(1,NUM_FRAMES_INPUT)]
                #add the first frame to the batch
                batch.insert(0, images['image'])
            else:
                batch = [cv2.resize(f, (224,224)) for f in batch]
            
            target = torch.tensor(cls_idx, dtype=torch.long).unsqueeze(0).to(device)
            target = torch.nn.functional.one_hot(target, num_classes=len(class_to_idx)).float()
            if target.dim() > 2:
                target = target.squeeze(0)
            target = target.sum(dim=0, keepdim=True)
            preds, prev_kvs, loss_val = model(batch, old_past_key_values=prev_kvs, labels=target)
            optimizer.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_stats['losses'].append(loss_val.item())
            avg_loss = np.mean(global_stats['losses'])
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'frames': global_stats['frames']})

        cap.release()


def evaluate(model, val_videos, val_labels, class_to_idx, fps_out, device,best_f1=0.0):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for vid, lbl in tqdm(zip(val_videos, val_labels), total=len(val_videos), desc="Validating"):
            cap = cv2.VideoCapture(vid)
            if not cap.isOpened(): continue

            timestart, cls_event = get_rtf_text(lbl)
            
         
            base_idx = []
            for cls in cls_event:
                if cls in class_to_idx:
                    base_idx.append(class_to_idx[cls])
                else:
                    print(f"Warning: Class '{cls}' not found in class_to_idx. Using 'No event' instead.")
                    base_idx.append(class_to_idx["No event"])

            frame_buf, prev_kvs = [], None
            idx_in, idx_out = -1, -1
            fps_in = cap.get(cv2.CAP_PROP_FPS)
            fcount = 0

            while cap.isOpened():
                ret, frame = cap.read()
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

                cls_idx = base_idx
                if (fcount / fps_in) < timestart:
                    cls_idx = class_to_idx["No event"]
                gt_vector = [0] * len(class_to_idx)
                if isinstance(cls_idx, int):
                    cls_idx = [cls_idx]
                for label in cls_idx:
                    gt_vector[label] = 1
                batch = [cv2.resize(f, (224,224)) for f in batch]
                preds, prev_kvs = model(batch, old_past_key_values=prev_kvs)
                # Append the sigmoid output
                probs = torch.sigmoid(preds)
                raw_preds = preds
                preds = (probs > 0.5).int().cpu().tolist()
                #if preds is a single value, make it a list and convert to one-hot encoding
                if isinstance(preds, int):
                    empty_vector = [0] * len(class_to_idx)
                    empty_vector[preds] = 1
                    preds = empty_vector
                # Append the predictions and ground truth vectors
                all_preds.append(preds)
                all_trues.append(gt_vector)
                #print(f"Preds: {all_preds[-1]}, Ground Truth: {all_trues[-1]}, Raw Preds: {raw_preds}")
            # Handle any remaining frames in the buffer

            if frame_buf:
                batch = sample_or_pad(frame_buf)
                cls_idx = base_idx
                if (fcount / fps_in) < timestart:
                    cls_idx = class_to_idx["No event"]
                gt_vector = [0] * len(class_to_idx)
                if isinstance(cls_idx, int):
                    cls_idx = [cls_idx]
                for label in cls_idx:
                    gt_vector[label] = 1
                batch = [cv2.resize(f, (224,224)) for f in batch]
                preds, prev_kvs = model(batch, old_past_key_values=prev_kvs)
                # Append the sigmoid output
                probs = torch.sigmoid(preds)
                raw_preds = preds
                preds = (probs > 0.5).int().cpu().tolist()
                
                #if preds is a single value, make it a list and convert to one-hot encoding
                if isinstance(preds, int):
                    empty_vector = [0] * len(class_to_idx)
                    empty_vector[preds] = 1
                # Append the predictions and ground truth vectors
                all_preds.append(preds)
                all_trues.append(gt_vector)
                #print(f"Preds: {all_preds[-1]}, Ground Truth: {all_trues[-1]}, Raw Preds: {raw_preds}")
            cap.release()
        #print("All Predictions:")

    #calculate f1 score as a multi-label classification
    #print("All Predictions and Ground Truths:")
    #print(all_preds)
    #print("All Trues:")
    #print(all_trues)

 
    
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    #now they are[[[0, 1, 0]], .. i nstead of [[0, 1, 0], [0, 1, 0], ...]
    all_preds = [item[0] for item in all_preds]
    #print("All Predictions after flattening:")
    #print(all_preds)

    f1 = f1_score(all_trues, all_preds, average='macro')
    print(classification_report(all_trues, all_preds, target_names=list(class_to_idx.keys())))
    if best_f1 < f1:
        print(f"New best F1 score: {f1:.4f} (previous: {best_f1:.4f})")
        best_f1 = f1
        # Save the model
        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(model.state_dict(), os.path.join('output', 'event_classifier.pth'))
    #return f1
    return best_f1
    



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx = {'No event': 0, 'Fire': 1, 'Smoke': 2}
    best_f1 = 0.0

    train_v, train_l, val_v, val_l = load_and_split(args.videos, args.labels, args.seed)
    model = EventClassifier(num_labels=len(class_to_idx)).to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    

    global_stats = {'frames': 0, 'losses': deque(maxlen=1000)}
    TRAINING = True
    print("Using augmentations:", args.use_augmentations)
    for epoch in range(1, 21):
        print(f"\n=== Epoch {epoch}/20 ===")
        if TRAINING:
            model.train()
            train_one_epoch(model, optimizer,
                            train_v, train_l,
                            class_to_idx, args.fps, device,
                            global_stats,
                            use_augmentations=args.use_augmentations)

        avg_loss = np.mean(global_stats['losses']) if global_stats['losses'] else 0.0
        print(f"[Epoch {epoch}] Completed training. Cumulative Avg Loss: {avg_loss:.4f}")
        model.eval()
        best_f1 = evaluate(model, val_v, val_l, class_to_idx, args.fps, device, best_f1)

   

if __name__ == "__main__":
    main()
