# CapFlow: Image Captioning and Hashtag Generation Chrome Extension

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Video Demo](#video-demo)
- [Key Features](#key-features)
- [AI Model Details](#ai-model-details)
  - [Image Captioning Model](#image-captioning-model)
  - [Hashtag Generation Model](#hashtag-generation-model)
- [Performance Metrics](#performance-metrics)
  - [Image Captioning Model](#image-captioning-model-1)
  - [Hashtag Generation Model](#hashtag-generation-model-1)
- [Setup and Installation](#setup-and-installation)
  - [Option 1: Using Docker](#option-1-using-docker)
  - [Option 2: Local Setup with Virtual Environment](#option-2-local-setup-with-virtual-environment)
  - [Chrome Extension Setup](#chrome-extension-setup)
- [Usage](#usage)
- [References](#references)

## Project Overview

CapFlow is a Chrome extension that uses the fine-tuned version of Microsoft's Florence 2 base model to generate descriptive captions for images on web pages and create relevant hashtags. 

## Video Demo

Check out our demo video to see CapFlow in action:

<div style="position: relative; width: 480px; height: 270px;">
    <a href="https://youtu.be/9vuyXFFfgok" target="_blank">
        <img src="https://img.youtube.com/vi/9vuyXFFfgok/maxresdefault.jpg" alt="CapFlow Demo" width="480" height="270" border="10" />
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);">
            <img src="path/to/play_button.png" alt="Play Button" width="68" height="48" />
        </div>
    </a>
</div>

## Technologies Used

- Frontend: Chrome Extension (JavaScript, HTML, CSS)
- Backend: Flask (Python)
- AI Models:
  - Image Captioning: Microsoft Florence-2-base-ft (230 million parameter Vision Language Model)
  - Hashtag Generation: KeyBERT
- Containerization: Docker

## Key Features

1. **Chrome Extension**: Allows users to select images on web pages for captioning and hashtag generation.
2. **Flask Backend**: Hosts the AI models and provides an API for the extension.
3. **Advanced AI Models**: 
   - Utilizes Microsoft's Florence-2-base-ft model with custom fine-tuned adapters for image captioning.
   - Implements KeyBERT for efficient hashtag generation from captions.
4. **Docker Integration**: Demonstrates containerization skills for easy deployment and scalability.

## AI Model Details

### Image Captioning Model

The image captioning function in `app.py` uses Microsoft's Florence-2-base-ft, a 232 million parameter Vision Language Model. It is a vision foundation model designed to handle a wide range of computer vision and vision-language tasks using a unified, prompt-based representation. It can perform tasks such as captioning, object detection, grounding, and segmentation by interpreting text prompts and generating text outputs. Florence-2’s training leveraged FLD-5B, a dataset with 5.4 billion annotations across 126 million images, created through automated image annotation and model refinement(Xiao et al., 2024). For this pWe used Florence-2's detailed image captioning functionality for this project <MORE_DETAILED_CAPTION> special tag.

This base Florence model was the fine-tuned using three seperate datasets using LoRA (Low-Rank Adaptation), to creat three LoRA adapters:

1. DOCCI adapter
2. Pixelprose adapter
3. Recap DataComp adapter

We used LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into different layers of the model. This approach reduces the number of trainable parameters for downstream tasks, which in turn decreases both the time required to fine-tune the model and the computational requirements(Hu et al., 2021). Huggingface provides a lot of documentation on LoRA and other PEFT models that aim to reduce resources required to train various deep learning models while retaining most of the performance (https://huggingface.co/docs/peft/index) 

We checked the performance of these adapters, both individually and all possible combinations of these adapters with the base model, based on which we selected the DOCCI adapter as the sole adapter for this project. These results can be found in the performance metrics section below.

### Hashtag Generation Model

The hashtag generation function in `app.py` uses KeyBERT (Khan et al., 2022), an unsupervised keyword extraction method that leverages BERT embeddings to identify keywords that best represent the underlying text. The process consists of three main steps:

1. Candidate keyword extraction: Uses Scikit-Learn's Count Vectorizer to obtain a list of candidate n-grams, ranking them based on their frequency in the original document.

2. BERT embedding: Both the input text (image caption) and the n-gram candidates are transformed into numeric data using the BERT model.

3. Similarity calculation: KeyBERT identifies n-gram candidates that are similar to the document using cosine similarity. The candidates most similar to the document are more likely to be suitable hashtags expressing the document's content.

The similarity is calculated using the following formula:

$$ Similarity = COS(W \cdot S) $$

Where W is the word's word embedding vector and S is the sentence embedding vector.

This approach differs from traditional frequency-based methods by focusing on the relevance between words in the context of the sentence, utilizing the semantic and contextual information of words and phrases in the extraction process.

## Performance Metrics

### Image Captioning Model

For the evaluation of these model adapters, we used 4 traditional metrics, METEOR, BLEU, ROUGE-L and CIDEr, and one new metric which is specifically designed to evaluate long descriptive captions, CAPTURE.

The CAPTURE (CAPtion evaluation by exTracing and coUpling coRE information) metric is designed to evaluate detailed image captions more reliably than existing methods. It works by extracting visual elements - objects, attributes, and relations - from both the generated and reference captions using a T5-based Factual parser. It then applies a stop words filter to focus on tangible elements. It employs a three-stage matching process: exact matching for identical elements, synonym matching using WordNet for similar meanings, and soft matching using Sentence BERT embeddings for remaining elements. CAPTURE calculates precision and recall for each type of visual element, combines these into F1 scores, and produces a final weighted score. This approach allows CAPTURE to assess caption quality while being robust to variations in writing style. The CAPTURE Metric score is calculated as:

$$ \text{Score} = \frac{5 \cdot F1_{obj} + 5 \cdot F1_{attr} + 2 \cdot F1_{rel}}{5 + 5 + 2} $$

Where:
- $F1_{obj}$ is the F1 score for objects
- $F1_{attr}$ is the F1 score for attributes
- $F1_{rel}$ is the F1 score for relations

The authors of the original CAPTURE paper demonstrate that it achieves higher consistency with human judgments compared to other caption evaluation metrics, making it a valuable tool for assessing and improving LVLM performance in detail image captioning tasks(Dong et al., 2024).

Below are the performance metrics for various configurations of the image captioning model, which were fine-tuned by us, compared to the base model:

| Model Configuration | METEOR | BLEU | ROUGE-L | CIDEr | CAPTURE |
|---------------------|--------|------|---------|-------|---------|
| Base Florence Model | 0.2128 | 0.1100 | 0.2753 | 0.0312 | 0.5458 |
| DOCCI Adapter | 0.2671 | 0.1850 | 0.2874 | 0.0863 | 0.5757 |
| Pixelprose Adapter | 0.2501 | 0.1552 | 0.2982 | 0.0388 | 0.5554 |
| Recap DataComp Adapter | 0.2397 | 0.1501 | 0.2942 | 0.0348 | 0.5530 |
| DOCCI + Pixelprose | 0.2532 | 0.1753 | 0.2809 | 0.0589 | 0.5425 |
| DOCCI + DataComp | 0.2616 | 0.1847 | 0.2981 | 0.0623 | 0.5637 |
| Pixelprose + DataComp | 0.2066 | 0.1110 | 0.2734 | 0.0149 | 0.5237 |
| All Adapters Combined | 0.2230 | 0.1382 | 0.2723 | 0.0321 | 0.5242 |

Based on these results, we observed that the DOCCI adapter increased the performance of the model for the captioning task the most across most traditional metrics and, more importantly, for the CAPTURE metric. Another important observation is how each adapter individually outperforms the base Florence-2 model, but overall combination of the three produces worse results. These results showcase how the quality of DOCCI dataset can vastly improve the performance of descriptive image captioning task.

We compare our DOCCI adapted model to some of the new Vision Language Models(VLMs) that have been released recently.


| Model               | METEOR | BLEU   | ROUGE-L | CIDEr  | CAPTURE | 
|---------------------|--------|--------|---------|--------|---------|
| Paligemma-3b-mix-448 | 0.1242 | 0.0530 | 0.1947  | 0.0157 | 0.4038  |
| Llava-v1.6-mistral-7b-hf | 0.3346 | 0.2754 | 0.3231  | 0.0883 | 0.5674  |
| Phi-3-vision-128k-instruct | 0.3144 | 0.2359 | 0.3230  | 0.0955 | 0.5509  |
| Base Florence-2 model | 0.2128 | 0.1100 | 0.2753  | 0.0312 | 0.5458  |
| Our Model | 0.2671 | 0.1850 | 0.2874  | 0.0863 | 0.5757  |

For its size of only 232 Million parameters, the base Florence-2 model (microsoft/Florence-2-base-ft) performs incredibly well against a much larger model like Llava-mistral with 7 Billion parameter. Even though it gets beaten by both Llava-mistral and phi-3-vision (4.2B paramaters), it gets extremely close. With the help of our fine-tuned DOCCI adapter, we were able to outperform these models. In comparision, the Paligemma (3B parameter) model really falls behind the others, though its important to note that this model is geared towards more research oriented studied and Google recommends fine-tuning it for specific tasks.

As a browser extension, speed is very important. We recorded the time taken for caption generation by each of the previously mentioned models over 100 images. The results are shown below.

![SpeedTest](https://github.com/user-attachments/assets/fe754587-e57d-47ee-b675-ac8cccd1a3c4)

It is not surprising that due to its smaller size, both the base Florence-2 model and our adapter built on top of it show remarkably quick times below one second. The PaliGemma model, being a small model itself, is also very quick, in fact, it is quicker on average than both Florence-2 and our model. However, it is inconsistent as it occasionally reaches the 3-second mark for some samples. The Phi-3-Vision model, with its 4.2B parameters, performs decently in terms of speed but lags in comparison. The Llava-mistral model, being the largest, was significantly off the mark. Additionally, the Llava-mistral model is extremely difficult to run on consumer-grade hardware and would require expensive cloud servers if we were to deploy this extension.

For the speed-to-performance ratio, our model performs best out of all the models we have tested. With only a slight increase in processing time, we achieve a significant improvement in performance compared to the base Florence model. The chart below showcases how it outperforms significantly larger models in detailed image captioning tasks while taking only a fraction of the time. This would likely result in higher user satisfaction due to quicker loading times, while also reducing processing and storage costs on cloud infrastructure.

![SpeedvsCapture](https://github.com/user-attachments/assets/dbc48be0-d69d-444c-a5df-14bb1fe1e865)

| Model               | CAPTURE | Avg Time (in sec)   |
|---------------------|--------|--------|
| Paligemma           | 0.4038 | 1.201 |
| LLAVA-mistral       | 0.5674 | 17.983 |
| Phi-3-vision        | 0.5509 | 7.358 |
| Base Florence model | 0.5458 | 0.523 |
| Our Model      | 0.5757 | 0.688 |

### Hashtag Generation Model

We evaluated the performance of our KeyBERT-based hashtag generation model against other methods, including Regular BERT, YAKE (Yet Another Keyword Extractor), and Gemma-2-2B-it. The evaluation metrics used were Precision, Recall, F1-score, and average processing time. Here are the results:

| Method | Precision | Recall | F1-score | Avg Time (s) |
|--------|-----------|--------|----------|--------------|
| KeyBERT | 0.6132 | 0.6105 | 0.5932 | 0.0201 |
| Regular BERT | 0.3392 | 0.3222 | 0.3199 | 0.0107 |
| YAKE | 0.5126 | 0.5068 | 0.4939 | 0.0064 |
| Gemma-2-2B-it | 0.4808 | 0.4239 | 0.4537 | 2.6293 |

KeyBERT demonstrated the highest precision, recall, and F1-score, while maintaining a relatively fast processing time. This makes it well-suited for real-time applications like our Chrome extension. YAKE proved to be the fastest method but with moderate performance, while Gemma-2-2B-it showed decent performance but with significantly higher processing time, making it less suitable for our use case.

![Precision-Recall Performance of Hashtag Generation Methods](https://github.com/user-attachments/assets/bfd2aaed-7908-4350-ad06-d65cac926ddb)

The precision-recall trade-off, visualized above, illustrates KeyBERT's superior performance. KeyBERT's data point is positioned furthest from the origin and closest to the top-right corner, indicating its optimal balance between precision and recall. YAKE follows as the second-best performer, while Gemma-2-2B-it and Regular BERT show lower performance in both precision and recall.

![Speed-Accuracy Trade-off in Hashtag Generation Methods (Log-Scaled)](https://github.com/user-attachments/assets/1ea83791-793f-474e-b43e-e342c6b16287)

This visualization provides insight into the speed-accuracy trade-off among these methods. The x-axis represents the average processing time on a logarithmic scale, while the y-axis shows the F1-score. This reveals that while KeyBERT achieves the highest F1-score with a relatively low processing time, making it an excellent choice for real-time applications. YAKE offers the fastest processing time with moderate accuracy. Gemma-2-2B-it, despite its more complex architecture, doesn't outperform KeyBERT and is significantly slower, making it less suitable for a browser extension. Regular BERT's position on this graph underscores its suboptimal performance, offering neither a speed nor an accuracy advantage.

Based on these results, we concluded that KeyBERT provides the most effective solution for our hashtag generation task within the context of a browser extension. Its superior accuracy in generating relevant hashtags, combined with its relatively fast processing time, makes it well-suited for real-time applications.


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

### Chrome Extension Setup

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

## Resources
• Third-Party Dataset Sources:
  – DOCCI Dataset: https://huggingface.co/datasets/google/docci
  – PixelproseDataset: https://huggingface.co/datasets/tomg-group-umd/pixelprose
  – Recap DataComp Dataset: https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B
• Our Fine-tuned LoRA Adapters:
  – DOCCIAdapter: https://huggingface.co/NikshepShetty/Florence-2-DOCCI-FT
  – PixelproseAdapter: https://huggingface.co/NikshepShetty/Florence-2-pixelprose
  – Recap DataComp Adapter: https://huggingface.co/NikshepShetty/Florence-2-Recap-DataComp
• Third-Party Evaluation Datasets:
  – DetailCaps-4870 (Caption Evaluation): https://huggingface.co/datasets/foundation-multimodal-models/DetailCaps-4870
  – TechKeywordsTopicsSummary(HashtagEvaluation): https://huggingface.co/datasets/ilsilfverskiold/tech-keywords-topics-summary

## References

1. Dong, H., Li, J., Wu, B., Wang, J., Zhang, Y. and Guo, H., 2024. Benchmarking and Improving Detail Image Caption. arXiv preprint arXiv:2405.19092.
2. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. and Chen, W., 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.
3. Khan, M. Q., Shahid, A., Uddin, M. I., Roman, M., Alharbi, A., Alosaimi, W., Almalki, J., & Alshahrani, S. M. (2022). Impact analysis of keyword extraction using contextual word embedding. PeerJ Computer Science, 8, e967. https://doi.org/10.7717/peerj-cs.967
4. Xiao, B., Wu, H., Xu, W., Dai, X., Hu, H., Lu, Y., Zeng, M., Liu, C. and Yuan, L., 2024. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4818-4829).
