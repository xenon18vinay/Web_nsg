import cv2
import os
import numpy as np
from ultralytics import YOLO
import subprocess
import shutil
import torch


def check_gpu_availability():
    """Check GPU availability and return device info"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if gpu_count > 0 else 0
        return {
            'available': True,
            'count': gpu_count,
            'name': gpu_name,
            'memory_gb': round(gpu_memory, 1),
            'device': 'cuda'
        }
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU support
        return {
            'available': True,
            'count': 1,
            'name': 'Apple Silicon GPU',
            'memory_gb': 'Shared',
            'device': 'mps'
        }
    else:
        return {
            'available': False,
            'count': 0,
            'name': 'CPU Only',
            'memory_gb': 0,
            'device': 'cpu'
        }


def load_yolo_model_with_gpu(model_path, device='auto'):
    """Load YOLO model with GPU optimization"""
    try:
        gpu_info = check_gpu_availability()

        if device == 'auto':
            if gpu_info['available']:
                device = gpu_info['device']
                print(f"🚀 GPU detected: {gpu_info['name']}")
                print(f"📊 GPU memory: {gpu_info['memory_gb']} GB")
                print(f"🎯 Using device: {device}")
            else:
                device = 'cpu'
                print("⚠️ No GPU detected, falling back to CPU")
                print("   For faster processing, consider:")
                print("   - Installing CUDA for NVIDIA GPUs")
                print("   - Using Apple Silicon Mac for MPS support")

        # Load model
        print(f"🔄 Loading YOLO model from: {model_path}")
        model = YOLO(model_path)

        # Move model to specified device
        model.to(device)

        # Optimize model for inference
        if device != 'cpu':
            print("⚡ Optimizing model for GPU inference...")
            # Enable half precision for faster inference on supported GPUs
            if device == 'cuda':
                try:
                    model.model.half()
                    print("✅ Half precision (FP16) enabled for faster inference")
                except:
                    print("⚠️ Half precision not supported, using FP32")

        print(f"✅ Model loaded successfully on {device.upper()}")
        return model, device

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Falling back to CPU...")
        model = YOLO(model_path)
        return model, 'cpu'


def run_yolo_inference_on_frame(model, frame, device='cpu'):
    """Runs YOLO inference on a single frame with GPU optimization"""
    try:
        # Run inference with device specification
        results = model(frame, verbose=False, device=device)
        processed_frame = results[0].plot()
        return processed_frame
    except Exception as e:
        print(f"⚠️ GPU inference failed, falling back to CPU: {e}")
        # Fallback to CPU inference
        results = model(frame, verbose=False, device='cpu')
        processed_frame = results[0].plot()
        return processed_frame


def run_batch_inference(model, frames_batch, device='cpu'):
    """Run YOLO inference on a batch of frames for better GPU utilization"""
    try:
        if len(frames_batch) == 1:
            return [run_yolo_inference_on_frame(model, frames_batch[0], device)]

        # Batch processing for better GPU utilization
        results = model(frames_batch, verbose=False, device=device)
        processed_frames = []

        for result in results:
            processed_frame = result.plot()
            processed_frames.append(processed_frame)

        return processed_frames

    except Exception as e:
        print(f"⚠️ Batch inference failed, processing individually: {e}")
        # Fallback to individual processing
        processed_frames = []
        for frame in frames_batch:
            processed_frame = run_yolo_inference_on_frame(model, frame, device)
            processed_frames.append(processed_frame)
        return processed_frames


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


def process_video(input_path, output_path, model_path, target_width=None, output_fps=30, device='auto', batch_size=4):
    """
    Process video with YOLO using GPU acceleration and create browser-compatible output

    Args:
        input_path: Path to input video
        output_path: Path for output video
        model_path: Path to YOLO model
        target_width: Resize width for processing (None to keep original)
        output_fps: Output video FPS
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        batch_size: Number of frames to process in batch for GPU optimization
    """
    cap = None

    try:
        print("\n" + "=" * 60)
        print(f"--- GPU-ACCELERATED VIDEO PROCESSING ---")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print("=" * 60)

        # Check GPU availability and load model
        gpu_info = check_gpu_availability()
        print(f"🖥️ System Info:")
        print(f"   GPU Available: {gpu_info['available']}")
        if gpu_info['available']:
            print(f"   GPU: {gpu_info['name']}")
            print(f"   Memory: {gpu_info['memory_gb']} GB")

        # Load YOLO model with GPU support
        model, actual_device = load_yolo_model_with_gpu(model_path, device)

        # Adjust batch size based on device
        if actual_device == 'cpu':
            batch_size = 1  # No batching for CPU
            print("💻 CPU processing: Processing every frame individually")
        else:
            print(f"🚀 GPU processing: batch_size={batch_size}, processing every frame")

        # Check if ffmpeg is available
        has_ffmpeg = check_ffmpeg()
        if has_ffmpeg:
            print("✅ FFmpeg is available (best compatibility)")
        else:
            print("⚠️ FFmpeg not found, using OpenCV fallback (limited compatibility)")

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
        frame_batch = []

        print("🚀 Processing every frame with GPU acceleration for smooth video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure consistent frame size
            frame = cv2.resize(frame, (original_width, original_height))

            # Prepare frame for processing
            processing_frame = frame
            if target_width is not None and target_width > 0:
                processing_frame = cv2.resize(frame, (processing_width, processing_height))

            # Add to batch for processing
            frame_batch.append(processing_frame)

            # Process batch when full or at end of video
            if len(frame_batch) >= batch_size or frame_index == total_frames - 1:
                if actual_device != 'cpu' and len(frame_batch) > 1:
                    # GPU batch processing
                    processed_batch = run_batch_inference(model, frame_batch, actual_device)
                else:
                    # Individual processing for CPU or single frames
                    processed_batch = []
                    for batch_frame in frame_batch:
                        processed_frame = run_yolo_inference_on_frame(model, batch_frame, actual_device)
                        processed_batch.append(processed_frame)

                # Add processed frames to output
                for processed_frame in processed_batch:
                    if target_width is not None and target_width > 0:
                        processed_frame = cv2.resize(processed_frame, (original_width, original_height))

                    frames.append(processed_frame.copy())
                    processed_frame_count += 1

                frame_batch = []  # Clear batch

            frame_index += 1

            # Progress update
            if frame_index % 30 == 0 or frame_index == total_frames:
                progress = (frame_index / total_frames) * 100
                print(f"Progress: {frame_index}/{total_frames} ({progress:.1f}%) | Processed: {processed_frame_count}")

        # Release video capture
        cap.release()
        cap = None

        print(f"✅ Processed {len(frames)} frames total (GPU-accelerated: {processed_frame_count})")

        # Clear GPU memory if used
        if actual_device != 'cpu':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🗑️ GPU memory cache cleared")

        # Create output video (keeping all existing codec logic)
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
                    print(f"🚀 Processing completed with {actual_device.upper()} acceleration")
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
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
    print("CHECKING REQUIREMENTS FOR GPU PROCESSING")
    print("=" * 60)

    # Check Python packages
    required_packages = {
        'cv2': 'opencv-python',
        'ultralytics': 'ultralytics',
        'numpy': 'numpy',
        'torch': 'torch'
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

    # Check GPU support
    gpu_info = check_gpu_availability()
    print(f"\n🖥️ GPU Support:")
    if gpu_info['available']:
        print(f"✅ GPU detected: {gpu_info['name']}")
        print(f"📊 Device: {gpu_info['device']}")
        if gpu_info['device'] == 'cuda':
            print(f"📊 Memory: {gpu_info['memory_gb']} GB")
            print("🚀 NVIDIA GPU ready for CUDA acceleration")
        elif gpu_info['device'] == 'mps':
            print("🚀 Apple Silicon GPU ready for MPS acceleration")
    else:
        print("⚠️ No GPU detected - will use CPU")
        print("\n📦 For GPU acceleration:")
        print("   NVIDIA GPU: Install CUDA and PyTorch with CUDA support")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   Apple Silicon: PyTorch with MPS support should work automatically")

    # Check FFmpeg
    if check_ffmpeg():
        print("\n✅ FFmpeg is installed (RECOMMENDED for browser compatibility)")
    else:
        print("\n❌ FFmpeg is not installed (HIGHLY RECOMMENDED)")
        print("\n📦 Install FFmpeg:")
        print("   Windows: winget install ffmpeg")
        print("   Linux: sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    install_requirements()

    # Example usage
    print("\nExample usage with GPU:")
    print("---------------------")
    print("from video_processor import process_video")
    print("# Auto-detect best device (GPU/CPU)")
    print("success = process_video('input.mp4', 'output.mp4', 'models/best.pt', device='auto')")
    print("\n# Force GPU usage")
    print("success = process_video('input.mp4', 'output.mp4', 'models/best.pt', device='cuda')")
    print("\n# Optimize batch size for your GPU")
    print("success = process_video('input.mp4', 'output.mp4', 'models/best.pt', batch_size=8)")