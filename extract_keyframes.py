import subprocess
import json
import sys

def get_keyframes(video_path):
    """
    Extracts keyframe timestamps from an MP4 video using ffprobe.
    Requires FFmpeg to be installed on the system.
    """
    cmd = [
        'ffprobe',
        '-loglevel', 'error',
        '-skip_frame', 'nokey',
        '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time',
        '-of', 'json',
        video_path
    ]
    try:
        output = subprocess.check_output(cmd)
        data = json.loads(output)
        keyframes = [float(frame['pkt_pts_time']) for frame in data['frames']]
        return keyframes
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

def save_to_file(keyframes, output_file):
    with open(output_file, 'w') as f:
        for timestamp in keyframes:
            f.write(f"{timestamp}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_video.mp4> <output.txt>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_file = sys.argv[2]
    
    keyframes = get_keyframes(video_path)
    if keyframes:
        save_to_file(keyframes, output_file)
        print(f"Keyframes saved to {output_file}")
    else:
        print("No keyframes found or error occurred.")
