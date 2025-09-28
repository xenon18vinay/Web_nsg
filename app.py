import os
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from video_processor import process_video

# Get the absolute path to the folder where this script is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define all folder paths based on that absolute path.
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')  # <-- Your model name here

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300 MB max upload size


def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_folders():
    """Create necessary folders if they don't exist."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- THIS IS THE FIX ---
# Call create_folders() when the application starts.
# This ensures the directories exist before any request is handled.
create_folders()


# --------------------

@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part in the request.")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No file selected.")

        if not allowed_file(file.filename):
            return render_template('index.html', error="File type not allowed.")

        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)

            output_filename = f"processed_{filename}"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            file.save(input_path)

            success = process_video(input_path, output_path, MODEL_PATH)

            if success:
                return render_template('index.html', processed_video_filename=output_filename)
            else:
                return render_template('index.html',
                                       error="Video processing failed. The file may be corrupt or an unsupported format.")

    return render_template('index.html')


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("-" * 50)
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("The application will not work until the model is placed in the correct directory.")
        print("-" * 50)
    app.run(debug=True)

