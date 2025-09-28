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
    Processes a video file using a YOLO model.
    Returns True on success, False on failure.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False

    try:
        print("Loading YOLO model...")
        model = YOLO(model_path)
        print("Model loaded successfully.")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {input_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Processing video: {os.path.basename(input_path)}")
        print(f"Properties: {frame_width}x{frame_height} @ {fps:.2f} FPS")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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

    except Exception as e:
        print(f"\nAn error occurred during video processing: {e}")
        return False
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        cv2.destroyAllWindows()

    # Clean up the original uploaded file to save space
    if os.path.exists(input_path):
        os.remove(input_path)
        print(f"Removed input file: {input_path}")

    return True

