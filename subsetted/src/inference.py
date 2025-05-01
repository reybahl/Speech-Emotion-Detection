import pickle
import os
from scripts.feature_extraction import extract_features_as_df, save_spectrogram
import numpy as np
import torch
from CNN import CNN
from PIL import Image
import timm

CLASSES = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

def get_model(model_name):
    path = os.path.join('models', f'{model_name}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

lr = get_model('lr') 

# get feature names from logistic regression model
feature_cols = lr.feature_names_in_

# if the file exists, load the validation transform
if os.path.exists('models/val_transform.pth'):
    val_transform = torch.load('models/val_transform.pth', weights_only=False, map_location=torch.device('cpu'))
else:
    val_transform = None

if os.path.exists('models/cnn.pth'):
    cnn = CNN(num_classes=len(CLASSES))
    cnn.load_state_dict(torch.load('models/cnn.pth', weights_only=False, map_location=torch.device('cpu')))
    cnn = cnn.to('cpu')
    cnn.eval()
else:
    cnn = None

if os.path.exists('models/vit.pth'):
    VIT_MODEL_NAME = 'vit_tiny_patch16_224'

    vit = timm.create_model(
        VIT_MODEL_NAME,
        pretrained=True,
        num_classes=len(CLASSES)
    ).to('cpu')
    vit.load_state_dict(torch.load('models/vit.pth', weights_only=False, map_location=torch.device('cpu')))
    vit.eval()
else:
    vit = None

# separate function for neural nets
def predict_dl(audio_path, model_name):
    output_path = os.path.join('uploads', f'{os.path.splitext(os.path.basename(audio_path))[0]}.png')
    save_spectrogram('', '', audio_path, output_path = output_path)
    image = Image.open(output_path).convert('RGB')
    image = val_transform(image).unsqueeze(0).to('cpu')
    with torch.no_grad():
        if model_name == 'vit':
            output = vit(image)
        else:
            output = cnn(image)
        _, predicted = output.max(1)

    return CLASSES[predicted.item()]


def predict(model_name, X):
    X = X[feature_cols]
    try:
        model = get_model(model_name)
    except FileNotFoundError:
        return 'Model not found'
    
    if model_name == 'lgb':
        idx = model.predict(X).argmax(axis=1)[0]
        return CLASSES[idx]
    elif model_name == 'xgb':
        idx = model.predict(X)[0]
        return CLASSES[idx]
    else:
        pred = model.predict(X)
        if type(pred) == np.ndarray:
            return pred[0]
        else:
            return pred

if __name__ == '__main__':
    # sample prediction using LGBM model
    X = extract_features_as_df('data/test/ANG/1011_IEO_ANG_LO.mp3')
    print(predict('lr', X))