import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import pickle
from tqdm import tqdm
import traceback

# --- MediaPipe Pose Solution Helper ---
mp_pose = mp.solutions.pose

class MediaPipePoseEstimator:
    """Handles pose estimation using MediaPipe Pose."""
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initializes MediaPipe Pose."""
        print("Initializing MediaPipe Pose Estimator...")
        try:
            self.pose = mp_pose.Pose(
                static_image_mode=False, # Process video frames
                model_complexity=model_complexity,
                enable_segmentation=False, # Segmentation not needed for keypoints
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)
            print(f"MediaPipe Pose initialized with complexity={model_complexity}, "
                  f"min_detect_conf={min_detection_confidence}, min_track_conf={min_tracking_confidence}")
        except Exception as e:
            print(f"Error initializing MediaPipe Pose: {e}")
            print("Please ensure the 'mediapipe' library is installed correctly.")
            raise

    def estimate_pose_4_features(self, image):
        """
        Processes an image (BGR format) and extracts 33 landmarks
        [x, y, z, visibility] - 4 features.
        Returns a NumPy array (33, 4) or None if detection fails.
        """
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Make read-only for performance
            results = self.pose.process(image_rgb)
            # image_rgb.flags.writeable = True # Not strictly necessary to change back

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Extract [x, y, z, visibility] -> shape (33, 4)
                keypoints_4_np = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                    dtype=np.float32
                )
                # Basic shape check
                if keypoints_4_np.shape == (33, 4):
                    return keypoints_4_np
                else:
                    print(f"Warning: MediaPipe returned unexpected landmark shape: {keypoints_4_np.shape}. Expected (33, 4).")
                    return None
            else:
                return None # No pose detected
        except Exception as e:
            print(f"Error during MediaPipe pose estimation: {e}")
            # traceback.print_exc() # Uncomment for detailed debugging if needed
            return None

    def close(self):
        """Release MediaPipe Pose resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
            print("MediaPipe Pose resources released.")

# --- Helper Functions ---

def parse_ground_truth(video_path):
    annotation_dir = r"C:\Users\johna\Desktop\CS3264proj\Lecture_room\Lecture room\Annotation_files"
    video_name = os.path.basename(video_path)
    gt_file = os.path.join(annotation_dir, video_name.replace('.avi', '.txt'))
    print(gt_file)
    try:
        with open(gt_file, 'r') as f:
            lines = f.readlines()
            if not lines:
                print(f"Empty annotation file: {gt_file}")
                return 0
            try:
                fall_start = int(lines[0].strip())
                fall_end = int(lines[1].strip()) if len(lines) > 1 else None
            except (ValueError, IndexError):
                print(f"No valid fall start/end frame in {gt_file}")
                return 0
            if fall_start == 0:
                return 0
            return 1
    except FileNotFoundError:
        print(f"Annotation file not found: {gt_file}")
        return 0
    
def derive_label_from_filename(video_path):
    """Derives label (0 for ADL, 1 for Fall) from filename."""
    filename = os.path.basename(video_path)
    # Robust splitting: handle potential multiple hyphens or underscores
    parts = filename.lower().replace('_', '-').split('-')
    if len(parts) >= 1:
        label_str = parts[0]
        if label_str == "fall": return 1
        elif label_str == "adl": return 0
        # Add other labels if necessary
        # elif label_str == "walk": return 2
        else:
            print(f"Warning: Unknown label prefix '{parts[0]}' in filename: {filename}. Defaulting to 0 (ADL). Consider adding specific handling if needed.")
            return 0 # Default or raise error depending on strictness needed
            # raise ValueError(f"Unknown label '{parts[0]}' in filename: {filename}. Expected 'fall' or 'adl'.")
    else:
        raise ValueError(f"Filename '{filename}' format error. Cannot derive label.")


# --- MODIFIED sample_frames to also return FPS ---
def sample_frames(video_path, num_frames_to_sample):
    """
    Samples frames evenly from a video.
    Returns a tuple: (list_of_frames, fps)
    """
    frames = []
    fps = 0.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], fps # Return empty list and 0 fps

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        print(f"Warning: Video {video_path} reported 0 frames.")
        cap.release()
        return [], fps

    if fps <= 0:
        print(f"Warning: Video {video_path} reported FPS <= 0 ({fps}). Cannot reliably calculate velocity based on time. Will calculate velocity based on frame index difference.")
        # We will handle delta_t calculation later based on this fps value

    # Frame sampling logic
    if total_frames <= num_frames_to_sample:
        # Take all frames if video is shorter than desired samples
        frame_indices = np.arange(total_frames)
    else:
        # Sample evenly across the video
        frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

    read_success_count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            read_success_count += 1
        # else: # Optional: Log if a specific sampled frame failed to read
        #     print(f"Warning: Failed to read frame index {idx} from {video_path}")

    cap.release()

    if read_success_count == 0 and total_frames > 0:
         print(f"Error: Failed to read ANY frames from {video_path} despite it reporting {total_frames} total frames.")
         return [], fps # Return empty list

    # If fewer frames were read than sampled (e.g., due to read errors at the end)
    if len(frames) != len(frame_indices):
         print(f"Warning: Read {len(frames)} frames, expected {len(frame_indices)} based on sampling from {video_path}")

    return frames, fps

# --- Main Pre-processing Function ---
# --- MODIFIED to calculate velocity and store 7 features per frame ---
def create_pose_data_pickle_7_features(video_dir, output_pickle_file, num_frames=500, pose_estimator_config=None):
    """
    Processes videos, extracts 4 pose features [x, y, z, vis], calculates
    3 velocity features [vx, vy, vz] per landmark, and saves the data
    to a pickle file in the required format for TemporalGaitGCN.

    Pickle Structure per video:
    {
        'video_path': str,
        'label': int,
        'frames_data': [
            {'keypoints': np.array(33, 4), 'velocities': np.array(33, 3)}, # Frame 0
            {'keypoints': np.array(33, 4), 'velocities': np.array(33, 3)}, # Frame 1
            ...
        ]
    }

    Args:
        video_dir (str): Path to the directory containing video files.
        output_pickle_file (str): Path to save the output .pkl file.
        num_frames (int): Target number of frames to sample per video.
        pose_estimator_config (dict, optional): Configuration for MediaPipePoseEstimator.
    """
    if not os.path.isdir(video_dir):
        print(f"Error: Video directory not found: {video_dir}")
        return False

    # Find video files
    video_paths = sorted(glob.glob(os.path.join(video_dir, '*.mp4'))) + \
                  sorted(glob.glob(os.path.join(video_dir, '*.avi'))) + \
                  sorted(glob.glob(os.path.join(video_dir, '*.mov'))) # Add common extensions
    
    if not video_paths:
        print(f"Error: No compatible video files (.mp4, .avi, .mov) found in '{video_dir}'.")
        return False
    print(f"Found {len(video_paths)} potential video files.")

    if pose_estimator_config is None:
        pose_estimator_config = {'model_complexity': 1} # Default config

    pose_estimator = MediaPipePoseEstimator(**pose_estimator_config)
    all_processed_data = []
    skipped_videos = 0
    videos_with_no_pose = 0

    print(f"Starting pre-processing for {len(video_paths)} videos...")
    for video_path in tqdm(video_paths, desc="Processing Videos"):
        try:
            label = parse_ground_truth(video_path)
        except ValueError as e:
            print(f"Skipping video {os.path.basename(video_path)}: {e}")
            skipped_videos += 1
            continue

        # Sample frames AND get FPS
        sampled_frames, fps = sample_frames(video_path, num_frames)
        if not sampled_frames:
            print(f"Skipping video {os.path.basename(video_path)}: No frames could be sampled.")
            skipped_videos += 1
            continue

        # Determine time delta for velocity calculation
        if fps > 0:
            delta_t = 1.0 / fps
            using_fps_for_velocity = True
        else:
            delta_t = 1.0 # Calculate velocity as change per frame index difference
            using_fps_for_velocity = False
            # Optional: Log this decision per video if desired

        # --- Store frame data (keypoints & velocities) for this video ---
        video_frames_data_list = []
        prev_keypoints_4_np = None # Store the previous frame's keypoints for velocity calculation

        for frame in sampled_frames:
            # Estimate Pose -> NumPy array [x, y, z, vis] or None
            current_keypoints_4_np = pose_estimator.estimate_pose_4_features(frame)

            # Only proceed if pose estimation was successful for this frame
            if isinstance(current_keypoints_4_np, np.ndarray) and current_keypoints_4_np.shape == (33, 4):
                # Calculate velocity if we have previous frame's data
                if prev_keypoints_4_np is not None:
                    # Calculate difference in x, y, z coordinates
                    delta_coords = current_keypoints_4_np[:, :3] - prev_keypoints_4_np[:, :3]
                    velocities_np = delta_coords / delta_t
                else:
                    # First valid frame, velocity is zero
                    velocities_np = np.zeros((33, 3), dtype=np.float32)

                # Store the keypoints and calculated velocities for this frame
                frame_dict = {
                    'keypoints': current_keypoints_4_np,
                    'velocities': velocities_np
                }
                video_frames_data_list.append(frame_dict)

                # Update previous keypoints for the next iteration
                prev_keypoints_4_np = current_keypoints_4_np
            # else:
                # Pose estimation failed for this frame. We simply don't append it.
                # This maintains consistency but might shorten sequences if detection is patchy.
                # We also don't update prev_keypoints_4_np, so the next velocity calculation
                # will use the *last successful* frame's data. If this gap is large,
                # the velocity might be inaccurate. Alternatively, reset prev_keypoints to None here?
                # Let's stick to using last *successful* frame for now.
                # prev_keypoints_4_np = None # Option: Reset if current frame fails

        # Only add video data if *at least one frame* had successful pose estimation
        if video_frames_data_list:
            # --- Append data in the new required format ---
            all_processed_data.append({
                'video_path': video_path, # Store relative path for reference
                'label': label,
                'frames_data': video_frames_data_list # List of dicts
            })
            # Log velocity calculation method for the first video processed
            if len(all_processed_data) == 1:
                print(f"Velocity Calculation Note: Using {'FPS ({:.2f})'.format(fps) if using_fps_for_velocity else 'Frame Index Difference'} for delta_t.")
        else:
            # print(f"Skipping video {os.path.basename(video_path)}: No valid pose data extracted after sampling.")
            videos_with_no_pose += 1
            skipped_videos += 1 # Count this as skipped

    # Clean up pose estimator
    pose_estimator.close()

    print(f"\nPre-processing complete. Successfully processed and generated data for {len(all_processed_data)} videos.")
    if videos_with_no_pose > 0:
         print(f"Note: {videos_with_no_pose} videos were skipped because no pose landmarks could be detected in any sampled frame.")
    remaining_skips = skipped_videos - videos_with_no_pose
    if remaining_skips > 0:
        print(f"Additionally skipped {remaining_skips} videos due to other errors (sampling, label derivation).")


    if not all_processed_data:
        print("Error: No data was processed successfully. Pickle file will not be created.")
        return False

    # Save the processed data
    try:
        print(f"Saving processed data (7 features: x,y,z,vis + vx,vy,vz) to {output_pickle_file}...")
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_pickle_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        with open(output_pickle_file, 'wb') as f:
            pickle.dump(all_processed_data, f)
        print("Data saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")
        traceback.print_exc()
        return False

# --- Make the script runnable ---
if __name__ == "__main__":
    # --- Configuration ---
    VIDEO_SOURCE_DIR = 'Lecture_room/Lecture room/Video'
    NUM_FRAMES_TO_SAMPLE = 50      # <<< Set desired number of frames per video
    # --- MODIFIED: Default output filename reflects content ---
    OUTPUT_PICKLE_NAME = f'{os.path.basename(VIDEO_SOURCE_DIR)}_ur_fall_{NUM_FRAMES_TO_SAMPLE}_frames_7_features.pkl' # Sensible default name
    OUTPUT_DIR = 'preprocessed_data' # Optional: Save pickle to a sub-directory
    OUTPUT_PICKLE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_PICKLE_NAME)

    # Optional: Customize MediaPipe settings
    MP_CONFIG = {
        'model_complexity': 1,          # 0, 1, or 2. Higher complexity -> more accurate but slower.
        'min_detection_confidence': 0.5,# Higher value -> fewer false detections but might miss poses.
        'min_tracking_confidence': 0.5 # Higher value -> tracker more likely to lose target if confidence drops.
    }

    print("--- Starting Pose Data Pickle Creation (7 Features: XYZ+Vis + Vxyz) ---")
    print(f"Video Source Directory: '{VIDEO_SOURCE_DIR}'")
    print(f"Frames to Sample per Video: {NUM_FRAMES_TO_SAMPLE}")
    print(f"Output Pickle Path: '{OUTPUT_PICKLE_PATH}'")
    print(f"MediaPipe Config: {MP_CONFIG}")
    print("-" * 60)


    success = create_pose_data_pickle_7_features(
        video_dir=VIDEO_SOURCE_DIR,
        output_pickle_file=OUTPUT_PICKLE_PATH,
        num_frames=NUM_FRAMES_TO_SAMPLE,
        pose_estimator_config=MP_CONFIG
    )

    print("-" * 60)
    if success:
        print(f"--- Pickle file created successfully at '{OUTPUT_PICKLE_PATH}' ---")
    else:
        print("--- Pickle file creation failed ---")