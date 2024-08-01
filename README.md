# CapFlow: Image Captioning and Hashtag Generation Chrome Extension

## Project Overview

CapFlow is a Chrome extension that leverages advanced AI models to generate captions for images on web pages and create relevant hashtags. The project demonstrates proficiency in full-stack development, AI integration, and containerization technologies.


## Technologies Used

- Frontend: Chrome Extension (JavaScript, HTML, CSS)
- Backend: Flask (Python)
- AI Models:
  - Image Captioning: Microsoft Florence-2-base-ft (230 million parameter Vision Language Model)
  - Hashtag Generation: [Model details to be added]
- Containerization: Docker

## Key Features

1. **Chrome Extension**: Allows users to select images on web pages for captioning and hashtag generation.
2. **Flask Backend**: Hosts the AI models and provides an API for the extension.
3. **Advanced AI Models**: 
   - Utilizes Microsoft's Florence-2-base-ft model with custom fine-tuned adapters for image captioning.
   - Employs a separate model for generating relevant hashtags from captions.
4. **Docker Integration**: Demonstrates containerization skills for easy deployment and scalability.

## AI Model Details

### Image Captioning Model

The image captioning function in `app.py` uses Microsoft's Florence-2-base-ft, a 230 million parameter Vision Language Model. This base model is enhanced with three fine-tuned LoRA (Low-Rank Adaptation) adapters, each trained on separate datasets:

1. DOCCI adapter
2. Pixelprose adapter
3. Recap DataComp adapter

### Hashtag Generation Model

The hashtag generation function in `app.py` takes the captions produced by the image captioning model and generates three relevant hashtags. 

[Implementation details to be added]

## Performance Metrics

### Image Captioning Model

Below are the performance metrics for various configurations of the image captioning model:

| Model Configuration | METEOR | BLEU | ROUGE-L | CIDEr | CAPTURE |
|---------------------|--------|------|---------|-------|---------|
| Base Florence Model | 0.2128 | 0.1100 | 0.2753 | 0.0312 | 0.5458 |
| DOCCI Adapter | 0.2671 | 0.1850 | 0.2874 | 0.0863 | 0.5757 |
| Pixelprose Adapter | 0.2501 | 0.1552 | 0.2982 | 0.0388 | 0.5554 |
| Recap DataComp Adapter | 0.2397 | 0.1501 | 0.2942 | 0.0348 | 0.5530 |
| DOCCI + Pixelprose | 0.2532 | 0.1753 | 0.2809 | 0.0589 | 0.5425 |
| DOCCI + DataComp | 0.2616 | 0.1847 | 0.2981 | 0.0623 | 0.5637 |
| All Adapters Combined | 0.2230 | 0.1382 | 0.2723 | 0.0321 | 0.5242 |


### Hashtag Generation Model

[Evaluation metrics to be added]

# CapFlow: Image Captioning and Hashtag Generation Chrome Extension

[Previous sections remain unchanged]

## Setup and Installation

### Option 1: Using Docker

1. Ensure you have Docker and Docker Compose installed on your system.

2. Clone the repository:
   ```
   git clone https://github.com/yourusername/capflow.git
   cd capflow
   ```

3. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

This will start both the caption-api service on port 5000 and set up the chrome-extension service.

### Option 2: Local Setup with Virtual Environment

1. Ensure you have Python 3.10 or later installed.

2. Clone the repository:
   ```
   git clone https://github.com/yourusername/capflow.git
   cd capflow
   ```

3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install the required packages:
   ```
   cd caption-model-api
   pip install -r requirements.txt
   ```

5. Run the Flask application:
   ```
   python app.py
   ```

The API will be available at `http://localhost:5000`.

## Chrome Extension Setup

After setting up the backend (either through Docker or locally):

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable "Developer mode" in the top right corner.
3. Click "Load unpacked" and select the `browser-extension` directory from the cloned repository.

The CapFlow extension should now be installed and ready to use.

## Usage

1. Navigate to a webpage with images.
2. Click on the CapFlow extension icon in your browser.
3. Use the extension interface to select an image on the page.
4. The extension will send the image to the backend for processing.
5. View the generated caption and hashtags in the extension popup.
