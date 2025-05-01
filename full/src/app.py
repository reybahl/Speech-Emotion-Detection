from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from inference import predict, predict_dl
from scripts.feature_extraction import extract_features_as_df

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
MODELS = ['lgb', 'xgb', 'rf', 'mlp', 'lr', 'cnn', 'vit']
CLASS_TO_EMOTION = {
    'ANG': 'Anger',
    'DIS': 'Disgust',
    'FEA': 'Fear',
    'HAP': 'Happiness',
    'NEU': 'Neutral',
    'SAD': 'Sadness'
}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':    
        return jsonify({'error': 'No file selected'}), 400

    model = request.form.get('model', 'lgb')
    
    if model not in MODELS:
        return jsonify({'error': 'Invalid model'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if model == 'cnn':
            prediction = predict_dl(file_path, 'cnn')
        elif model == 'vit':
            prediction = predict_dl(file_path, 'vit')
        else:
            X = extract_features_as_df(file_path)
            prediction = predict(model, X)

        emotion = CLASS_TO_EMOTION[prediction]
        return jsonify({'prediction': emotion}), 200
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
