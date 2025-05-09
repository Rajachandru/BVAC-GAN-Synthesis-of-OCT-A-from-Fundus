import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import tensorflow as tf
import io

# --- Model Loading (same as your code) ---
from tensorflow.keras import layers

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale",
                                     shape=(input_shape[-1],),
                                     initializer="ones",
                                     trainable=True)
        self.offset = self.add_weight(name="offset",
                                      shape=(input_shape[-1],),
                                      initializer="zeros",
                                      trainable=True)
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config

class ThresholdSEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16, threshold=0.5, **kwargs):
        super(ThresholdSEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.threshold = threshold
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // reduction, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        x = self.global_avg_pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = tf.where(x > self.threshold, x, tf.zeros_like(x))
        input_shape = tf.shape(inputs)
        reshape_shape = [input_shape[0], 1, 1, self.channels]
        x = tf.reshape(x, reshape_shape)
        return inputs * x

    def get_config(self):
        config = super(ThresholdSEBlock, self).get_config()
        config.update({
            "channels": self.channels,
            "reduction": self.reduction,
            "threshold": self.threshold
        })
        return config

# --- Load model ---
custom_objects = {
    'InstanceNormalization': InstanceNormalization,
    'ThresholdSEBlock': ThresholdSEBlock
}
# "C:\Users\B SAKETH REDDY\Downloads\Fundas\"
model_path = 'C:\Personal\Research\IEEE-IES-GenAI-Hackathon2025\coding\Fundus_to_OCTA\generator_g.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

loaded_generator_g = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# --- Flask App ---
app = Flask(__name__)

IMG_HEIGHT = 256
IMG_WIDTH = 256
CHANNELS = 3

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image).astype(np.float32) / 127.5 - 1.0  # normalize to [-1, 1]
    return np.expand_dims(image_array, axis=0)

def postprocess_image(output_tensor):
    output_array = (output_tensor[0] * 0.5 + 0.5) * 255.0  # scale back to [0, 255]
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_array)
    return output_image

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    image_file = request.files['image']
    try:
        input_tensor = preprocess_image(image_file.read())
        prediction = loaded_generator_g.predict(input_tensor)
        output_image = postprocess_image(prediction)

        img_io = io.BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return '''

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AI Image Generator</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg-light: #ffffff;
      --bg-dark: #1e1e1e;
      --text-light: #333;
      --text-dark: #f5f5f5;
      --card-light: #ffffff;
      --card-dark: #2c2c2c;
      --accent-light: #2a5298;
      --accent-dark: #90caf9;
    }

    * {
      box-sizing: border-box;
    }

    body {
      font-family: 'Poppins', sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      background: linear-gradient(135deg, #1e3c72, #2a5298);
      transition: background 0.3s ease;
    }

    body.dark {
      background: linear-gradient(135deg, #121212, #1e1e1e);
    }

    .container {
      background: var(--card-light);
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
      text-align: center;
      width: 90%;
      max-width: 600px;
      color: var(--text-light);
      position: relative;
      transition: background 0.3s, color 0.3s;
    }

    body.dark .container {
      background: var(--card-dark);
      color: var(--text-dark);
    }

    .toggle-container {
      position: absolute;
      top: 15px;
      right: 15px;
    }

    .toggle-switch {
      position: relative;
      display: inline-block;
      width: 50px;
      height: 26px;
    }

    .toggle-switch input {
      opacity: 0;
      width: 0;
      height: 0;
    }

    .slider {
      position: absolute;
      cursor: pointer;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: #ccc;
      transition: .4s;
      border-radius: 34px;
    }

    .slider:before {
      position: absolute;
      content: "";
      height: 20px;
      width: 20px;
      left: 4px;
      bottom: 3px;
      background-color: white;
      transition: .4s;
      border-radius: 50%;
    }

    input:checked + .slider {
      background-color: var(--accent-light);
    }

    input:checked + .slider:before {
      transform: translateX(24px);
    }

    h1 {
      font-weight: 600;
      margin-bottom: 20px;
    }

    .file-input {
      background: #f4f4f4;
      padding: 15px;
      border-radius: 6px;
      border: 2px dashed var(--accent-light);
      display: block;
      width: 100%;
      cursor: pointer;
      text-align: center;
      transition: 0.3s;
    }

    .file-input:hover {
      background: #dcdcdc;
    }

    input[type="file"] {
      display: none;
    }

    .file-name {
      margin-top: 10px;
      font-size: 14px;
      word-break: break-word;
    }

    .btn {
      background: var(--accent-light);
      color: white;
      border: none;
      padding: 12px 18px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      transition: 0.3s;
      display: block;
      width: 100%;
      margin-top: 15px;
    }

    .btn:hover {
      background: #1e3c72;
      transform: translateY(-2px);
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .btn:disabled {
      background: #ccc;
      cursor: not-allowed;
    }

    .output {
      margin-top: 20px;
      text-align: center;
      display: none;
    }

    .images {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 15px;
      margin-top: 15px;
    }

    .images img {
      width: 100%;
      max-width: 270px;
      border-radius: 10px;
      box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
      animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }

    .download-btn {
      display: none;
      background: #28a745;
      color: white;
      border: none;
      padding: 10px 15px;
      border-radius: 6px;
      cursor: pointer;
      margin-top: 10px;
    }

    .progress-bar {
      width: 100%;
      background: #ddd;
      height: 8px;
      margin-top: 10px;
      border-radius: 5px;
      display: none;
    }

    .progress-bar div {
      height: 100%;
      background: var(--accent-light);
      width: 0%;
      border-radius: 5px;
      transition: width 0.4s ease;
    }

    body.dark .file-input {
      background: #444;
      border-color: var(--accent-dark);
    }

    body.dark .btn {
      background: var(--accent-dark);
    }

    body.dark .btn:hover {
      background: #2196f3;
    }

    @media (max-width: 500px) {
      .images {
        flex-direction: column;
        align-items: center;
      }

      .images img {
        width: 90%;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="toggle-container">
      <label class="toggle-switch">
        <input type="checkbox" id="theme-toggle">
        <span class="slider"></span>
      </label>
    </div>

    <h1>AI-Powered Image Generator</h1>
    <form id="upload-form" enctype="multipart/form-data">
      <label class="file-input" for="image-input">Click or Drag & Drop to Upload Image</label>
      <input type="file" name="image" id="image-input" accept="image/*" required/>
      <p class="file-name" id="file-name">No file chosen</p>
      <button type="submit" class="btn" id="generate-btn" disabled>Upload and Generate Image</button>
      <div class="progress-bar" id="progress-bar"><div></div></div>
    </form>
    <div class="output" id="output">
      <p>Generated image will appear below:</p>
      <div class="images">
        <img id="uploaded-image" src="" alt="Uploaded Image"/>
        <img id="generated-image" src="" alt="Generated Image"/>
      </div>
      <button class="download-btn" id="download-btn">Download Image</button>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById("image-input");
    const fileNameDisplay = document.getElementById("file-name");
    const generateBtn = document.getElementById("generate-btn");
    const progressBar = document.getElementById("progress-bar").firstElementChild;
    const generatedImage = document.getElementById("generated-image");
    const uploadedImage = document.getElementById("uploaded-image");
    const downloadBtn = document.getElementById("download-btn");
    const outputSection = document.getElementById("output");
    const themeToggle = document.getElementById("theme-toggle");

    // Theme toggle
    const setTheme = (isDark) => {
      document.body.classList.toggle("dark", isDark);
      localStorage.setItem("theme", isDark ? "dark" : "light");
      themeToggle.checked = isDark;
    }

    themeToggle.addEventListener("change", () => {
      setTheme(themeToggle.checked);
    });

    // Load saved theme
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "dark") {
      setTheme(true);
    }

    fileInput.addEventListener("change", function(event) {
      const file = event.target.files[0];
      if (file) {
        fileNameDisplay.textContent = file.name;
        const reader = new FileReader();
        reader.onload = function(e) {
          uploadedImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
        generateBtn.disabled = false;
      }
    });

    document.getElementById("upload-form").addEventListener("submit", async function(event) {
      event.preventDefault();
      generateBtn.disabled = true;
      outputSection.style.display = "block";
      document.getElementById("progress-bar").style.display = "block";
      progressBar.style.width = "0%";

      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        progressBar.style.width = progress + "%";
        if (progress >= 100) clearInterval(interval);
      }, 200);

      const formData = new FormData();
      formData.append("image", fileInput.files[0]);

      try {
        const response = await fetch("/generate", { method: "POST", body: formData });
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        generatedImage.src = imageUrl;
        downloadBtn.style.display = "inline-block";
      } catch (error) {
        alert("Error generating image. Please try again.");
      } finally {
        generateBtn.disabled = false;
        document.getElementById("progress-bar").style.display = "none";
      }
    });

    downloadBtn.addEventListener("click", () => {
      const link = document.createElement("a");
      link.href = generatedImage.src;
      link.download = "generated_image.png";
      link.click();
    });
  </script>
</body>
</html>
    '''



if __name__ == '__main__':
    app.run(debug=True)
