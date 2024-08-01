from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from PIL import Image
from io import BytesIO
import base64
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import os
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import requests
from keybert import KeyBERT
import re

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

# Model setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def load_model():
    base_model_name = "microsoft/Florence-2-base-ft"
    adapter_names = [
        "NikshepShetty/Florence-2-DOCCI-FT"
    ]

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, attn_implementation="sdpa",
                                                          torch_dtype=dtype, trust_remote_code=True).to(device)

    for adapter_name in adapter_names:
        app.logger.info(f"Merging adapter: {adapter_name}")
        adapter_model = PeftModel.from_pretrained(base_model, adapter_name, trust_remote_code=True)
        base_model = adapter_model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    return base_model, processor

# Load model and processor
model, processor = load_model()
app.logger.info("Model and processor loaded successfully")


# Load KeyBERT model
kw_model = KeyBERT()

def get_keywords(text, num_keywords=3):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=num_keywords)
    hashtags = ["#{}".format(re.sub(r'[^\w]', '', word)) for word, _ in keywords]
    return hashtags

def get_image_from_data(image_data):
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    return Image.open(BytesIO(image_bytes))

def generate_caption(image):
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>",
                                                      image_size=(image.width, image.height))
    caption=""
    try:
        caption = parsed_answer['<MORE_DETAILED_CAPTION>']
    except:
        pass

    return caption

@app.route('/generate_caption', methods=['POST'])
def caption_route():
    app.logger.info("Received request for caption generation")
    data = request.json
    
    if not data or ('image_data' not in data and 'image_url' not in data):
        app.logger.error("No image data or URL provided in request")
        return jsonify({"error": "No image data or URL provided"}), 400
    
    try:
        if 'image_data' in data:
            image = get_image_from_data(data['image_data'])
        else:
            response = requests.get(data['image_url'])
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        
        app.logger.info("Image successfully loaded")
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400
    
    try:
        caption = generate_caption(image)  
        app.logger.info(f"Caption generated: {caption}")
        
        # Generate hashtags
        hashtags = get_keywords(caption)
        app.logger.info(f"Hashtags generated: {hashtags}")
        
        return jsonify({"caption": caption, "hashtags": hashtags})
    except Exception as e:
        app.logger.error(f"Error generating caption: {str(e)}")
        return jsonify({"error": f"Error generating caption: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)