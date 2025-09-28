import os
import subprocess
from flask import Flask, request, render_template, url_for, send_file
from werkzeug.utils import secure_filename
from video_processor import process_video, check_ffmpeg, convert_existing_video_to_browser_compatible

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
                                   error=f"File type not allowed. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")

        if file:
            # Secure the filename
            filename = secure_filename(file.filename)
            base_name = os.path.splitext(filename)[0]

            # Generate unique filename to avoid conflicts
            import time
            timestamp = str(int(time.time()))
            unique_base = f"{base_name}_{timestamp}"

            input_path = os.path.join(UPLOAD_FOLDER, f"{unique_base}{os.path.splitext(filename)[1]}")

            # Always output as .mp4 for browser compatibility
            output_filename = f"processed_{unique_base}.mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            try:
                # Save uploaded file
                file.save(input_path)
                print(f"📁 Saved upload: {input_path}")

                # Check if model exists
                if not os.path.exists(MODEL_PATH):
                    return render_template('index.html',
                                           error=f"Model file not found at {MODEL_PATH}. Please ensure the YOLO model is in place.")

                # Process the video
                print(f"🎬 Processing: {input_path} -> {output_path}")
                success = process_video(
                    input_path,
                    output_path,
                    MODEL_PATH,
                    frame_skip=2,  # Process every 2nd frame for speed
                    output_fps=30  # Standard web framerate
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
                                                   file_size_mb=round(file_size / 1024 / 1024, 1))
                        else:
                            return render_template('index.html',
                                                   error="Video processing completed but output file is too small. Processing may have failed.")
                    else:
                        return render_template('index.html',
                                               error="Video processing completed but output file not found.")
                else:
                    # Processing failed
                    error_msg = "Video processing failed. "

                    # Check for common issues
                    if not check_ffmpeg():
                        error_msg += "FFmpeg is not installed - this is required for browser-compatible videos. "

                    error_msg += "Please check the console output for details."

                    return render_template('index.html', error=error_msg)

            except Exception as e:
                print(f"❌ Error during processing: {e}")
                import traceback
                traceback.print_exc()
                return render_template('index.html',
                                       error=f"An error occurred during processing: {str(e)}")
            finally:
                # Optional: Clean up old files periodically
                cleanup_old_files()

    # GET request - show upload form
    return render_template('index.html')


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
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'ffmpeg': check_ffmpeg(),
        'model_exists': os.path.exists(MODEL_PATH),
        'upload_folder': os.path.exists(UPLOAD_FOLDER),
        'output_folder': os.path.exists(OUTPUT_FOLDER)
    }
    return status


if __name__ == '__main__':
    # Initial setup
    create_folders()

    print("\n" + "=" * 60)
    print("YOLO VIDEO PROCESSOR - STARTUP CHECK")
    print("=" * 60)

    # Check for model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ WARNING: Model file not found at {MODEL_PATH}")
        print("   The application will not work until the model is placed in:")
        print(f"   {MODEL_PATH}")
    else:
        print(f"✅ Model found at {MODEL_PATH}")

    # Check for FFmpeg
    if check_ffmpeg():
        print("✅ FFmpeg is installed (videos will be browser-compatible)")
    else:
        print("⚠️ FFmpeg is not installed")
        print("   Videos may not play in browsers without FFmpeg!")
        print("   Install with: sudo apt install ffmpeg (Linux)")
        print("                brew install ffmpeg (Mac)")
        print("                Download from ffmpeg.org (Windows)")

    print("=" * 60)
    print(f"Starting server at http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)