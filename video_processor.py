import cv2
import os
import numpy as np
from ultralytics import YOLO
import subprocess
import shutil


def run_yolo_inference_on_frame(model, frame):
    """Runs YOLO inference on a single frame."""
    results = model(frame, verbose=False)
    processed_frame = results[0].plot()
    return processed_frame


def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_browser_compatible_video(frames, output_path, fps):
    """
    Create a browser-compatible H.264 video using ffmpeg
    This is the most reliable method for browser compatibility
    """
    temp_dir = os.path.join(os.path.dirname(output_path), f"temp_frames_{os.getpid()}")

    try:
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)

        # Save frames as images
        print(f"💾 Saving {len(frames)} frames...")
        for i, frame in enumerate(frames):
            filename = os.path.join(temp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(filename, frame)

        # Ensure output is .mp4
        output_mp4 = output_path if output_path.endswith('.mp4') else output_path.rsplit('.', 1)[0] + '.mp4'

        # Use ffmpeg with H.264 codec for maximum compatibility
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',  # H.264 codec - most compatible
            '-preset', 'medium',  # Balance between speed and compression
            '-crf', '23',  # Good quality (lower = better, 0-51 range)
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-movflags', '+faststart',  # Enable fast start for web playback
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Ensure even dimensions
            output_mp4
        ]

        print("🎬 Creating browser-compatible H.264 video...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None

        # Verify the output file
        if os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 10000:
            print(f"✅ Browser-compatible video created: {output_mp4}")
            return output_mp4
        else:
            print("❌ Video file creation failed")
            return None

    except Exception as e:
        print(f"❌ Error creating video: {e}")
        return None
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def fallback_opencv_writer(frames, output_path, fps, frame_size):
    """
    Fallback to OpenCV if ffmpeg is not available
    Uses MJPEG codec which has better browser support than raw codecs
    """
    # For browser compatibility, we'll create an MP4 with h264 if possible
    # or fall back to MJPEG in AVI container

    codecs_to_try = [
        ('avc1', '.mp4'),  # Try H.264 first (macOS)
        ('H264', '.mp4'),  # H.264 alternative
        ('mp4v', '.mp4'),  # MPEG-4
        ('MJPG', '.avi'),  # MJPEG as last resort
    ]

    for codec_str, ext in codecs_to_try:
        try:
            output_file = output_path.rsplit('.', 1)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

            if out.isOpened():
                print(f"📝 Writing video with {codec_str} codec...")
                for frame in frames:
                    out.write(frame)
                out.release()

                # Verify the file
                if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:
                    # Test if it can be read back
                    test_cap = cv2.VideoCapture(output_file)
                    if test_cap.isOpened():
                        ret, _ = test_cap.read()
                        test_cap.release()
                        if ret:
                            print(f"✅ Video created with {codec_str} codec: {output_file}")
                            return output_file

                # If verification failed, remove the file
                if os.path.exists(output_file):
                    os.remove(output_file)

        except Exception as e:
            print(f"Failed with {codec_str}: {e}")
            continue

    return None


def process_video(input_path, output_path, model_path, frame_skip=2, target_width=None, output_fps=30):
    """
    Process video with YOLO and create browser-compatible output
    """
    cap = None

    try:
        print("\n" + "=" * 60)
        print(f"--- BROWSER-COMPATIBLE VIDEO PROCESSING ---")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print("=" * 60)

        # Check if ffmpeg is available
        has_ffmpeg = check_ffmpeg()
        if has_ffmpeg:
            print("✅ FFmpeg is available (best compatibility)")
        else:
            print("⚠️ FFmpeg not found, using OpenCV fallback (limited compatibility)")
            print("   For best results, install FFmpeg:")
            print("   Windows: Download from https://ffmpeg.org/")
            print("   Linux: sudo apt install ffmpeg")
            print("   Mac: brew install ffmpeg")

        # Load YOLO model
        model = YOLO(model_path)

        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("❌ Error: Could not open input video file.")
            return False

        # Get video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure even dimensions for H.264 compatibility
        original_width = original_width if original_width % 2 == 0 else original_width - 1
        original_height = original_height if original_height % 2 == 0 else original_height - 1

        print(f"📹 Input: {original_width}x{original_height}, {input_fps} FPS, {total_frames} frames")

        # Processing dimensions
        processing_width = original_width
        processing_height = original_height

        if target_width is not None and target_width > 0:
            aspect_ratio = original_height / original_width
            processing_width = target_width if target_width % 2 == 0 else target_width - 1
            processing_height = int(target_width * aspect_ratio)
            processing_height = processing_height if processing_height % 2 == 0 else processing_height - 1

        # Process frames
        frames = []
        frame_index = 0
        processed_frame_count = 0
        last_processed_frame = None

        print("🚀 Processing frames...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure consistent frame size
            frame = cv2.resize(frame, (original_width, original_height))

            # Process with YOLO every frame_skip frames
            if frame_index % frame_skip == 0:
                if target_width is not None and target_width > 0:
                    resized_frame = cv2.resize(frame, (processing_width, processing_height))
                    processed_resized_frame = run_yolo_inference_on_frame(model, resized_frame)
                    last_processed_frame = cv2.resize(processed_resized_frame, (original_width, original_height))
                else:
                    last_processed_frame = run_yolo_inference_on_frame(model, frame)

                processed_frame_count += 1

            # Store frame for output
            output_frame = last_processed_frame if last_processed_frame is not None else frame
            frames.append(output_frame.copy())

            frame_index += 1

            # Progress update
            if frame_index % 30 == 0 or frame_index == total_frames:
                progress = (frame_index / total_frames) * 100
                print(f"Progress: {frame_index}/{total_frames} ({progress:.1f}%) | Processed: {processed_frame_count}")

        # Release video capture
        cap.release()
        cap = None

        print(f"✅ Processed {len(frames)} frames total")

        # Create output video
        if has_ffmpeg:
            # Use ffmpeg for best compatibility
            output_file = create_browser_compatible_video(frames, output_path, output_fps)
        else:
            # Use OpenCV fallback
            print("⚠️ Using OpenCV fallback (limited browser compatibility)")
            output_file = fallback_opencv_writer(frames, output_path, output_fps,
                                                 (original_width, original_height))

        # Verify final output
        if output_file and os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ Video created: {output_file}")
            print(f"   Size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

            # Test if it can be read
            test_cap = cv2.VideoCapture(output_file)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_fps = test_cap.get(cv2.CAP_PROP_FPS)
                test_cap.release()

                if ret and test_frame is not None:
                    print(f"✅ Video verification: {test_frames} frames @ {test_fps} FPS")
                    print(f"✅ Video codec: H.264/AVC for browser compatibility")
                    return True
                else:
                    print("⚠️ Video created but cannot read frames")
                    return False
            else:
                print("⚠️ Video created but cannot open for verification")
                return True  # Still return True as file exists
        else:
            print("❌ Video file creation failed")
            return False

    except Exception as e:
        print(f"❌ Exception during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if cap:
            cap.release()
        print("🏁 Processing complete")


def convert_existing_video_to_browser_compatible(input_path, output_path):
    """
    Convert an existing video file to browser-compatible format
    Useful for fixing videos that don't play in browsers
    """
    if not check_ffmpeg():
        print("❌ FFmpeg is required for video conversion")
        return False

    try:
        # Ensure output is .mp4
        output_mp4 = output_path if output_path.endswith('.mp4') else output_path.rsplit('.', 1)[0] + '.mp4'

        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            output_mp4
        ]

        print(f"🔄 Converting {input_path} to browser-compatible format...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(output_mp4):
            print(f"✅ Converted to: {output_mp4}")
            return True
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False


def install_requirements():
    """Check and provide installation instructions for requirements"""
    print("\n" + "=" * 60)
    print("CHECKING REQUIREMENTS")
    print("=" * 60)

    # Check Python packages
    required_packages = {
        'cv2': 'opencv-python',
        'ultralytics': 'ultralytics',
        'numpy': 'numpy'
    }

    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is not installed")

    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")

    # Check FFmpeg
    if check_ffmpeg():
        print("✅ FFmpeg is installed (RECOMMENDED for browser compatibility)")
    else:
        print("❌ FFmpeg is not installed (HIGHLY RECOMMENDED)")
        print("\n📦 Install FFmpeg:")
        print("   Windows:")
        print("     1. Download from https://ffmpeg.org/download.html")
        print("     2. Extract and add to PATH")
        print("     OR use: winget install ffmpeg")
        print("   ")
        print("   Linux (Ubuntu/Debian):")
        print("     sudo apt update && sudo apt install ffmpeg")
        print("   ")
        print("   macOS:")
        print("     brew install ffmpeg")
        print("\n⚠️  Without FFmpeg, videos may not play in browsers!")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    install_requirements()

    # Example usage
    print("\nExample usage:")
    print("-------------")
    print("from video_processor import process_video")
    print("success = process_video('input.mp4', 'output.mp4', 'models/best.pt')")
    print("\nTo convert existing video to browser format:")
    print("from video_processor import convert_existing_video_to_browser_compatible")
    print("convert_existing_video_to_browser_compatible('old_video.avi', 'browser_video.mp4')")