from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import os
import io
import numpy as np
import onnxruntime
import time
import onnx
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.onnx_model import OnnxModel
import shutil

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
MAX_HEIGHT = 1600

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class ESRGAN:
    def __init__(self, model_path, tile_size=256, prepad=10, scale=2):
        self.model_path = model_path
        self.tile_size = tile_size
        self.prepad = prepad
        self.scale = scale
        self._init_model()

    def _tile_preprocess(self, img):
        input_data = np.array(img).transpose(2, 0, 1)
        img_data = input_data.astype('float32') / 255.0  # Change to float32
        padded_size = self.tile_size + self.prepad*2
        norm_img_data = img_data.reshape(1, 3, padded_size, padded_size)  # Keep as float32
        return norm_img_data

    def _into_tiles(self, img):
        self.width, self.height = img.size
        tile_size = self.tile_size
        prepad = self.prepad
        self.num_width = int(np.ceil(self.width/tile_size))
        self.num_height = int(np.ceil(self.height/tile_size))
        self.pad_width = self.num_width*tile_size
        self.pad_height = self.num_height*tile_size
        pad_img = Image.new("RGB", (self.pad_width, self.pad_height))
        pad_img.paste(img)
        tiles = []
        for i in range(self.num_height):
            for j in range(self.num_width):
                box = [j*tile_size, i*tile_size, (j+1)*tile_size, (i+1)*tile_size]
                box = [box[0]-prepad, box[1]-prepad, box[2]+prepad, box[3]+prepad]
                tiles.append(self._tile_preprocess(pad_img.crop(tuple(box))))
        return tiles

    def _into_whole(self, tiles):
        scaled_tile = self.scale * self.tile_size
        scaled_pad = self.scale * self.prepad
        out_img = Image.new("RGB", (self.pad_width*self.scale, self.pad_height*self.scale))
        paste_cnt = 0
        for i in range(self.num_height):
            for j in range(self.num_width):
                box = (scaled_pad,scaled_pad,scaled_pad+scaled_tile,scaled_pad+scaled_tile)
                tile = tiles[paste_cnt].resize((scaled_pad*2+scaled_tile,scaled_pad*2+scaled_tile))
                paste_pos = (j*scaled_tile, i*scaled_tile)
                out_img.paste(tile.crop(box), paste_pos)
                paste_cnt += 1
        return out_img.crop((0, 0, self.width*self.scale, self.height*self.scale))

    def _init_model(self):
        self.exec_provider = 'CPUExecutionProvider'
        self.session_opti = onnxruntime.SessionOptions()
        self.session_opti.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session_opti.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        self.session_opti.intra_op_num_threads = 8  # Adjust based on your CPU

        self.session = onnxruntime.InferenceSession(
            self.model_path, 
            self.session_opti, 
            providers=[self.exec_provider]
        )
        
        self.model_input = self.session.get_inputs()[0].name
    
        # Print expected input type
        print(f"Expected input type: {self.session.get_inputs()[0].type}")  
        
    
    def get_result(self, img):
        start_time = time.time()
        input_tiles = self._into_tiles(img)
        tiling_time = time.time() - start_time

        start_time = time.time()
        output_tiles = []
        for i, tile in enumerate(input_tiles):
            tile_start = time.time()
            result = self.session.run([], {self.model_input: tile})[0][0]
            result = np.clip(result.transpose(1, 2, 0), 0, 1) * 255.0
            output_tiles.append(Image.fromarray(result.round().astype(np.uint8)))
            print(f"Tile {i+1}/{len(input_tiles)} processed in {time.time() - tile_start:.2f} seconds")
        inference_time = time.time() - start_time

        start_time = time.time()
        final_image = self._into_whole(output_tiles)
        combining_time = time.time() - start_time

        print(f"Tiling time: {tiling_time:.2f} seconds")
        print(f"Inference time: {inference_time:.2f} seconds")
        print(f"Combining time: {combining_time:.2f} seconds")
        print(f"Total time: {tiling_time + inference_time + combining_time:.2f} seconds")

        return final_image

model_path = 'models/2xLiloScale_80K.onnx'
model = ESRGAN(model_path, tile_size=256, scale=2)

# Ensure upload and output directories exist
if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            # Check file size
            file_contents = file.read()
            if len(file_contents) > MAX_FILE_SIZE:
                return render_template('index.html', error='File size exceeds 2MB limit')
            
            # Check image dimensions
            image = Image.open(io.BytesIO(file_contents))
            if image.height > MAX_HEIGHT:
                return render_template('index.html', error='Image height exceeds 1600px limit')
            
            # Save input image
            input_filename = f"input_{int(time.time())}.png"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            image.save(input_path)
            
            # Process image
            output_image = model.get_result(image)
            
            # Save output image
            output_filename = f"output_{int(time.time())}.png"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            output_image.save(output_path)
            
            return render_template('result.html', input_image=input_filename, output_image=output_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)