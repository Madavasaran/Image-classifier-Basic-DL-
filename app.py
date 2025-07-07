import os
import torch
from torchvision import models, transforms
from torchvision.models import VGG19_Weights
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = VGG19_Weights.DEFAULT
model = models.vgg19(weights=weights).to(device)
model.eval()

# Use transform from the weights
transform = weights.transforms()

# Prediction function
def model_predict(img_path, model):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        _, predicted = output.max(1)

    class_names = weights.meta["categories"]
    return class_names[predicted.item()]

@app.route('/', methods=['GET'])
def index():
    return render_template('Visual_page.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(filepath)

        result = model_predict(filepath, model)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
