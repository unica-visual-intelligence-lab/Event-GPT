from striprtf.striprtf import rtf_to_text


import os


def get_rtf_text(file_path):
    #open file has the format int,string
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as file:
        rtf_content = file.read()   
    # Convert RTF to plain text
    text_content = rtf_to_text(rtf_content)
    if not text_content:
        return 0, ['No event']
    #its a int,string format, so we need to split it
    timestart = text_content.split(',')[0].strip()
    class_event= text_content.split(',')[1:]
    #merge the rest of the string if there are multiple parts
    class_event = ','.join(class_event).strip()
    #if the class event has a ,
    if ',' in class_event:
        class_event = ['Fire', 'Smoke']
    if class_event == 'smoke':
        class_event = ['Smoke']
    #if the class event is not an array, make it an array
    if not isinstance(class_event, list):
        class_event = [class_event]
    return int(timestart), class_event


def corresponding_label_to_video(videos_path, labels_path):
    """
    Returns a dictionary mapping video file names to their corresponding label file names.
    """
    
    #get full paths of video files
    video_files = []
    for subfolder in os.listdir(videos_path):
        video_folder = os.path.join(videos_path, subfolder)
        if not os.path.exists(video_folder):
            continue
        for file in os.listdir(video_folder):
            if file.endswith('.mp4'):
                video_files.append(os.path.join(video_folder, file))

    #get full paths of label files
    label_files = []
    for subfolder in os.listdir(labels_path):
        label_folder = os.path.join(labels_path, subfolder)
        if not os.path.exists(label_folder):
            continue
        for file in os.listdir(label_folder):
            if file.endswith('.rtf'):
                label_files.append(os.path.join(label_folder, file))

    
    #sort the video files and label files
    video_files.sort()
    label_files.sort()
    print(f"Found {len(video_files)} video files and {len(label_files)} label files.")
    if len(video_files) != len(label_files):
        raise ValueError("The number of video files and label files do not match.")
    #print the first 5 video files and label files
    #print("First 5 video files:", video_files[:5])
    #print("First 5 label files:", label_files[:5])
    return video_files, label_files

if __name__ == "__main__":
    # Example usage
    file_path = 'E:/2025_ICIAP_FIRE/GT/GT_TRAINING_SET_CL1/Video0.rtf'  # Replace with your RTF file path
    try:
        timestamp, event_class = get_rtf_text(file_path)
        print(f"Timestamp: {timestamp}, Event Class: {event_class}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

