import cv2
import os
from ultralytics import YOLO


def run_yolo_inference_on_frame(model, frame):
    """Runs YOLO inference on a single frame."""
    results = model(frame, verbose=False)
    processed_frame = results[0].plot()
    return processed_frame


def process_video(input_path, output_path, model_path):
    """
    Processes a video with a YOLO model.
    Returns True on success, False on failure.
    """
    cap = None
    out = None
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return False
        model = YOLO(model_path)

        if not os.path.exists(input_path):
            print(f"Error: Input video not found at {input_path}")
            return False

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- THIS IS THE FIX ---
        # The 'avc1' codec (H.264) is highly compatible with web browsers.
        # 'mp4v' is not always supported for in-browser playback.
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print("Error: Could not open video writer.")
            return False

        print(f"Processing video: {os.path.basename(input_path)}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = run_yolo_inference_on_frame(model, frame)
            out.write(processed_frame)

            frame_count += 1
            print(f"Processed frame: {frame_count} of {total_frames}", end='\r')

        print(f"\nFinished processing. Output video saved to: {output_path}")
        return True

    except Exception as e:
        print(f"\nAn error occurred during video processing: {e}")
        if out is not None:
            out.release()
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
                print(f"Removed input file: {input_path}")
            except OSError as e:
                print(f"Error removing input file {input_path}: {e}")

