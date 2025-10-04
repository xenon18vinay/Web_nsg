import os
import subprocess
from flask import Flask, request, render_template, url_for, send_file
from werkzeug.utils import secure_filename
from video_processor import process_video, check_ffmpeg, convert_existing_video_to_browser_compatible, \
    check_gpu_availability

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300 MB max upload size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_folders():
    """Create necessary folders if they don't exist"""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def cleanup_old_files():
    """Clean up old processed files (optional, for storage management)"""
    try:
        # Clean files older than 1 hour
        import time
        current_time = time.time()

        for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > 3600:  # 1 hour in seconds
                            try:
                                os.remove(file_path)
                                print(f"🗑️ Cleaned up old file: {file}")
                            except:
                                pass
    except Exception as e:
        print(f"Cleanup error: {e}")


def get_optimal_processing_settings():
    """Get optimal processing settings based on available hardware"""
    gpu_info = check_gpu_availability()

    if gpu_info['available']:
        if gpu_info['device'] == 'cuda':
            # NVIDIA GPU - can handle larger batches
            if gpu_info['memory_gb'] >= 8:
                return {
                    'device': 'cuda',
                    'batch_size': 8,
                    'description': f"🚀 NVIDIA GPU ({gpu_info['memory_gb']}GB) - High Performance"
                }
            elif gpu_info['memory_gb'] >= 4:
                return {
                    'device': 'cuda',
                    'batch_size': 4,
                    'description': f"🚀 NVIDIA GPU ({gpu_info['memory_gb']}GB) - Good Performance"
                }
            else:
                return {
                    'device': 'cuda',
                    'batch_size': 2,
                    'description': f"🚀 NVIDIA GPU ({gpu_info['memory_gb']}GB) - Conservative"
                }
        elif gpu_info['device'] == 'mps':
            # Apple Silicon GPU
            return {
                'device': 'mps',
                'batch_size': 4,
                'description': f"🍎 Apple Silicon GPU - Optimized"
            }

    # CPU fallback
    return {
        'device': 'cpu',
        'batch_size': 1,
        'description': "💻 CPU Processing - Every Frame"
    }


@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        # Validate file upload
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in the request.")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        if not allowed_file(file.filename):
            return render_template('index.html',
                                   error=f"File type not allowed supported formats: {', '.join(ALLOWED_EXTENSIONS)}")

        if file:

            filename = secure_filename(file.filename)
            base_name = os.path.splitext(filename)[0]


            import time
            timestamp = str(int(time.time()))
            unique_base = f"{base_name}_{timestamp}"

            input_path = os.path.join(UPLOAD_FOLDER, f"{unique_base}{os.path.splitext(filename)[1]}")


            output_filename = f"processed_{unique_base}.mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            try:

                file.save(input_path)
                print(f" Saved upload: {input_path}")

                # Check if model exists
                if not os.path.exists(MODEL_PATH):
                    return render_template('index.html',
                                           error=f"Model file not found at {MODEL_PATH}. Please ensure the YOLO model is in place.")

                # Get optimal processing settings for available hardware
                settings = get_optimal_processing_settings()
                print(f"⚙️ Processing mode: {settings['description']}")

                # Process the video with GPU acceleration
                print(f"🎬 GPU Processing: {input_path} -> {output_path}")
                print(
                    f"🔧 Settings: Device={settings['device']}, Batch={settings['batch_size']}, Processing=Every Frame")

                success = process_video(
                    input_path,
                    output_path,
                    MODEL_PATH,
                    output_fps=30,  # Standard web framerate
                    device=settings['device'],
                    batch_size=settings['batch_size']
                )

                if success:
                    # Find the actual output file (might have different extension)
                    actual_output_path = None
                    actual_output_filename = None

                    # Check for the expected output
                    if os.path.exists(output_path):
                        actual_output_path = output_path
                        actual_output_filename = output_filename
                    else:
                        # Look for any processed file with our unique base
                        for file in os.listdir(OUTPUT_FOLDER):
                            if file.startswith(f"processed_{unique_base}"):
                                actual_output_path = os.path.join(OUTPUT_FOLDER, file)
                                actual_output_filename = file
                                break

                    if actual_output_path and os.path.exists(actual_output_path):
                        file_size = os.path.getsize(actual_output_path)

                        # Verify it's a valid video file
                        if file_size > 10000:  # At least 10KB
                            print(f"✅ Output video ready: {actual_output_filename}")
                            print(f"   Size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
                            print(f"   Processed with: {settings['description']}")

                            # Additional browser compatibility check
                            if not actual_output_filename.endswith('.mp4'):
                                print("⚠️ Output is not MP4, attempting conversion for browser compatibility...")
                                mp4_path = actual_output_path.rsplit('.', 1)[0] + '.mp4'
                                if convert_existing_video_to_browser_compatible(actual_output_path, mp4_path):
                                    actual_output_filename = os.path.basename(mp4_path)
                                    print(f"✅ Converted to MP4: {actual_output_filename}")

                            # Clean up the uploaded file to save space
                            try:
                                os.remove(input_path)
                            except:
                                pass

                            return render_template('index.html',
                                                   processed_video_filename=actual_output_filename,
                                                   file_size_mb=round(file_size / 1024 / 1024, 1),
                                                   processing_info=settings['description'])
                        else:
                            return render_template('index.html',
                                                   error="Video processing completed but output file is too small. Processing may have failed.")
                    else:
                        return render_template('index.html',
                                               error="Video processing completed but output file not found.")
                else:
                    # Processing failed
                    error_msg = "GPU-accelerated video processing failed. "

                    # Check for common issues
                    gpu_info = check_gpu_availability()
                    if not gpu_info['available']:
                        error_msg += "No GPU detected - ensure CUDA/PyTorch is properly installed for acceleration. "

                    if not check_ffmpeg():
                        error_msg += "FFmpeg is not installed - this is required for browser-compatible videos. "

                    error_msg += "Please check the console output for details."

                    return render_template('index.html', error=error_msg)

            except Exception as e:

                import traceback
                traceback.print_exc()
                error_details = str(e)
                if "CUDA" in error_details:
                    error_msg = f"GPU processing error: {error_details}. Try CPU processing or check CUDA installation."
                elif "MPS" in error_details:
                    error_msg = f"Apple GPU processing error: {error_details}. Try CPU processing."
                else:
                    error_msg = f"Processing error: {error_details}"

                return render_template('index.html', error=error_msg)
            finally:
                # Optional: Clean up old files periodically
                cleanup_old_files()

    # GET request - show upload form with system info
    gpu_info = check_gpu_availability()
    processing_mode = get_optimal_processing_settings()

    return render_template('index.html',
                           gpu_info=gpu_info,
                           processing_mode=processing_mode)


@app.route('/download/<filename>')
def download_file(filename):
    """Provide direct download of processed video"""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, secure_filename(filename))
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Download error: {str(e)}", 500


@app.route('/health')
def health_check():
    """Health check endpoint with GPU info"""
    gpu_info = check_gpu_availability()
    processing_settings = get_optimal_processing_settings()

    status = {
        'status': 'healthy',
        'ffmpeg': check_ffmpeg(),
        'model_exists': os.path.exists(MODEL_PATH),
        'upload_folder': os.path.exists(UPLOAD_FOLDER),
        'output_folder': os.path.exists(OUTPUT_FOLDER),
        'gpu_available': gpu_info['available'],
        'gpu_device': gpu_info['device'],
        'gpu_name': gpu_info['name'],
        'processing_mode': processing_settings['description']
    }
    return status


@app.route('/system-info')
def system_info():
    gpu_info = check_gpu_availability()
    processing_settings = get_optimal_processing_settings()

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "Not available"
    except:
        torch_version = "Not installed"
        cuda_available = False
        cuda_version = "Not available"

    info = {
        'gpu_info': gpu_info,
        'processing_settings': processing_settings,
        'torch_version': torch_version,
        'cuda_available': cuda_available,
        'cuda_version': cuda_version,
        'ffmpeg_available': check_ffmpeg(),
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    }
    return info


if __name__ == '__main__':
    # Initial setup
    create_folders()

    print("\n" + "=" * 70)
    print("GPU-ACCELERATED YOLO VIDEO PROCESSOR - STARTUP CHECK")
    print("=" * 70)

    # Check for model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ WARNING: Model file not found at {MODEL_PATH}")
        print("   The application will not work until the model is placed in:")
        print(f"   {MODEL_PATH}")
    else:
        print(f"✅ Model found at {MODEL_PATH}")

    # Check GPU capabilities
    gpu_info = check_gpu_availability()
    processing_settings = get_optimal_processing_settings()

    print(f"\n🖥️ Hardware Detection:")
    if gpu_info['available']:
        print(f"✅ GPU Available: {gpu_info['name']}")
        print(f"📊 Device: {gpu_info['device'].upper()}")
        if gpu_info['memory_gb'] != 'Shared':
            print(f"💾 Memory: {gpu_info['memory_gb']} GB")
        print(f"⚙️ Processing Mode: {processing_settings['description']}")
        print(f"🔧 Batch Size: {processing_settings['batch_size']}")
        print("📽️ Processing: Every single frame for smooth video")
        print("📽️ Processing: Every single frame for smooth video")
    else:
        print("⚠️ No GPU detected - using CPU processing")
        print("   For GPU acceleration:")
        print("   - NVIDIA GPU: Install CUDA and PyTorch with CUDA support")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("   - Apple Silicon: Should work automatically with MPS")

    # Check PyTorch and CUDA
    try:
        import torch

        print(f"\n🔥 PyTorch: v{torch.__version__}")
        if torch.cuda.is_available():
            print(f"🚀 CUDA: v{torch.version.cuda} (GPU acceleration ready)")
            print(f"🎯 Available GPUs: {torch.cuda.device_count()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 MPS: Available (Apple Silicon GPU acceleration ready)")
        else:
            print("💻 GPU acceleration: Not available (CPU only)")
    except ImportError:
        print("❌ PyTorch not installed - GPU acceleration unavailable")

    # Check for FFmpeg
    if check_ffmpeg():
        print("\n✅ FFmpeg is installed (videos will be browser-compatible)")
    else:
        print("\n⚠️ FFmpeg is not installed")
        print("   Videos may not play in browsers without FFmpeg!")
        print("   Install with: sudo apt install ffmpeg (Linux)")
        print("                brew install ffmpeg (Mac)")
        print("                winget install ffmpeg (Windows)")

    print("=" * 70)
    print(f"🌐 Starting server at http://localhost:5000")
    print(f"📊 System info: http://localhost:5000/system-info")
    print(f"❤️ Health check: http://localhost:5000/health")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)