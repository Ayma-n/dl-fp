# Generative CLIP-Conditioned Variational Autoencoders (CVAEs)
Work inspired by the full text-conditional image generation stack introduced in [Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen. Hierarchical Text-Conditional Image Generation with CLIP Latents. 2022.](https://arxiv.org/pdf/2204.06125.pdf)

Final Project for CSCI1470 "Deep Learning" during Spring 2024 at Brown University. <br/>
By Ariana Azarbal (aazarbal), Ayman Benjelloun Touimi (abenjell), and Sofia Tazi (stazi).

## Other Final Deliverables
- [DL Day Poster](https://github.com/Ayma-n/dl-fp/blob/main/poster.pdf)
- [Final Report](https://docs.google.com/document/d/1eWXnU5SLrRJec3WjQCzKKW7B6-9bn8qj7qAlRBV4qeo/edit?usp=sharing)

## How to Navigate this Repository

### Resources
- The datasets we've used, final weights after training, and Google Colab versions of our notebooks can be found on our [Final Handin](https://drive.google.com/drive/folders/18ivvw1xlyJ4zIK8Tr03ZSgloiA6ZjYRi?usp=drive_link) Google Drive folder. **Note that the notebooks assume this folder is called "Deep Learning FP" and is present in your main Google Drive folder.**

### Notebooks
- The bulk of the code we run is in the Google Colab notebooks, which we've included copies of in the `final_notebooks/` directory. The names of the notebooks should outline what dimensions of images we're training or testing the model on, and different properties (dropout vs. no drouput, etc.)

### Python Modules and Scripts
We make use of three main modules or scripts we wrote:
- `clip_wrapper.py` contains a few functions dedicated to interacting with the `clip` Python library. It provides convenient methods to obtain image or text embeddings.
- `preprocess.py` contains our preprocessing code. It contains a few methods to obtain a Tensorflow Dataset of Image/Embedding pairs. We incldued multiple methods, that allow us to get either 64x64 and 128x128 images, and their image or text embeddings. 
- `offline_preprocess.py` is a Python script designed to be run offline to prepare the dataset, even before preprocessing. It is this script that narrows down the set of images in the training dataset to bucolic-themed images. This script was a necessity due to the MS COCO training dataset being too large to unzip in Google Colab. Note that the ZIP files referenced there can be found on the original MS COCO website, [here](https://cocodataset.org/#download).

Note: Two other Python files, `model_64x64.py` and `model_128x128.py` include copies of our model. They are just there for reference, and are not directly used inside the notebooks.

## Introduction
Contrastive Language Image Pretraining (CLIP)  opens up vast possibilities for multimodal learning by producing an aligned representation of text and image data in a shared latent space. Not only has this model demonstrated impressive zero-shot capabilities, but has been incorporated as an effective image encoder in Vision-Language Models (VLMs) which perform tasks such as visual question and answering, object detection, etc .  

In the context of Diffusion model-based generation, the paper "Hierarchical Text-Conditional Image Generation with CLIP Latents," presented the method of conditioning on CLIP embeddings. Instead of employing a computationally intensive diffusion model, our group chose to explore integrating CLIP embeddings with a Variational Autoencoder (VAE). We experiment with various image sizes, architectures, and CLIP encoding modes (image embeddings vs. text embeddings). 

## Related Work
- [The paper introducing unCLIP](https://arxiv.org/pdf/2204.06125.pdf), a powerful CLIP-conditioned generative diffusion model which inspired us to apply the basic principle to VAEs. 
- The [original CLIP paper](https://arxiv.org/pdf/2103.00020.pdf). CLIP aligns image and text embeddings in the same representation space, which we leverage for generation.

## Data
- [MS-COCO](https://cocodataset.org/#home)  filtered to be  <br/>
a) bucolic (relating to pastoral life, i.e. filtering for “cow”, “sheep”, etc.) <br/>
b) naturey (relating to broader natural terms like “grass”, “mountain”, “hillside”, etc.).

## Methodology
We train a VAE (consisting of multiple encoders and a decoder) on MS-COCO with reconstruction and KL-divergence loss. We condition by concatenating either CLIP image encodings or caption encodings to the output of the encoder (which processed the original image). For generation, we concatenate the CLIP text encoding of the prompt to noise and feed this to the decoder. 

## Metrics
Given this task is intrinsically a generative task, it is difficult to find a very clear metric for accuracy. We use our model's loss to track whether the model learns, but assessment of its performance is primarily qualitative.

In the original unCLIP paper, the authors use a similar process and assess the model's performance using human evaluation. Participants are asked to assess Photorealism, Caption Similarity, and Diversity. They primarily compare two versions of their model: one which derives embeddings autoregressively from the CLIP embeddings, and one involving a diffusion model. It seems like the authors were trying to outperform GLIDE, which they did for certain assessments and for certain guidance scales. For reference, this parameter assesses the influence of the guiding mechanism during the diffusion process.

## Ethics
- Why is Deep Learning a good approach to this problem?

It is difficult to imagine how a procedural algorithm could generate an image from a prompt. It would be virtually impossible to capture the immense diversity of possible words and prompts, and also the diversity in variations of output images from the same prompt. This task of identifying features in a given image is itself complex (but algorithms exist for that, like bag of words), but the task of directly linking visual representation and textual representation is even more substantial and difficult to achieve. The deep inferences that a deep model is able to make are crucial, and this is why we think Deep Learning is a good approach to the problem.

- What broader societal issues are relevant to your chosen problem space?

Generative tasks as a whole pose deep societal issues. The ability of a model to generate virtually any image makes generating fake images very easy, which can be used to spread misinformation over social media. This is particularly dangerous when these generated images involve real people, like political figures. To put it simply, it will be extremely difficult from now on to distinguish what is real from what is fake.

Another important think to consider is how data is used to train these models. In many cases, data is scraped from the Internet, often without the consent of the original artists. It is not fair for creators to have their intellectual property used to train a model people might rely on, instead of requesting service from the original creators and compensating them appropriately.
