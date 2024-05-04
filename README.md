# Image Generation with CLIP Conditioners using Conditional Variational Autoencoders (CVAEs)
Work inspired by the full text-conditional image generation stack introduced in [Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen. Hierarchical Text-Conditional Image Generation with CLIP Latents. 2022.](https://arxiv.org/pdf/2204.06125.pdf)

Final Project for CSCI1470 "Deep Learning" during Spring 2024 at Brown University. <br/>
By Ariana Azarbal (aazarbal), Ayman Benjelloun Touimi (abenjell), and Sofia Tazi (stazi).

## Introduction
Our group is interested in exploring the intricacies of text of visual data, and understanding more about the subtleties of language and nuances of text-based image generation. We were particularly intrigued by the CLIP model, and its ability to derive a latent space where both textual and visual embeddings live, effectively bridging the gap between them. Direct relations between text and images can therefore be derived, opening a plethora of possibilities, and allowing us to study the influence of visual embeddings on textual embeddings, and vice-versa. We thought exploring a paper that involves CLIP would be really interesting, and found OpenAI's 2022 paper "Hierarchical Text-Conditional Image Generation with CLIP Latents" very inspirational. The main goal of this project is to implement a model for text-to-image generation by integrating CLIP embeddings with an autoencoder architecture (less expensive to train and simpler than implementing a diffusion model) to produce diverse and high-quality images that are semantically consistent with given text descriptions. We thought the combination of CLIP and image generation is very powerful, and that implementing this approach would allow us to explore different capabilities that come out of CLIP's latent space characteristics.
 <br/>
The problem addressed in the paper is an example of <strong>structured prediction</strong>.

## Related Work
- The [original CLIP paper](https://arxiv.org/pdf/2103.00020.pdf). CLIP is remarkably useful in our case as the model is capable of directly linking image and text embeddings in the same latent space, thus bridging the gap between them. This means that an image embedding will be closer in the embedding space to a text embedding that describes it accurately. This property of CLIP is particularly interesting, as we are trying to directly link a caption/prompt to a new, generated image.
- The [GLIDE model](https://arxiv.org/pdf/2112.10741.pdf). GLIDE is a text-to-image generation system, similar to unCLIP, that utilizes diffusion models guided by NLP input to create photo-realistic images. Other than essentially performing the same task, the authors of the paper tried leveraging CLIP to train GLIDE (what they call "CLIP guidance"), which consists in using CLIP to assess how well a denoised image matches a given text prompt. 
- [Jonathan Ho, Ajay Jain, Pieter Abbeel. Denoising Diffusion Probabilistic Models. 2020](https://arxiv.org/pdf/2006.11239.pdf). A foundational paper introducing the idea of denoising diffusion models for image synthesis. This part is essential in the architecture we're trying to reproduce, given that it takes care of the actual image generation.

## Data
For this project, we will need a substantial dataset of image-caption pairs. Many of them come to mind:
- [MS-COCO](https://cocodataset.org/#home): Mentioned in the paper, it is one of the most commonly used for tasks involving images and their caption. It's important to note that according to section 5.3, unCLIP is not directly trained on MS-COCO's training data. Because of the large necessary computing power, we might need to use only a subset of MS-COCO.
- [WIT: Wikipedia-Based Image Text Dataset](https://github.com/google-research-datasets/wit). A large multimodal, multilingual dataset with 37M+ image-text examples. Here again, due to the immense amount of data, we will probably need to train on a smaller, randomly chosen subset.
- [Cornell Movie Dialog Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). We thought it would be a fun idea to make use of this dataset. However, we recognize the preprocessing might be a lot of work, given that it would require us to extend appropriate movie scenes / pictures to generate the data.
- [SciXGen](https://aclanthology.org/2021.findings-emnlp.128.pdf) : A scientific paper dataset. Albeit mostly used for text generation, focusing on figures and their descriptions instead could help us create like visual interpretations of scientific content. Performance and preprocessing would be a major concern here, however.
[Recipe1M+](http://pic2recipe.csail.mit.edu/) : Cooking recipes with corresponding food images. Training extensively on this dataset can help us generate images from textual recipes (probably the ingredients).

## Methodology
We are looking to implement, in Tensorflow, the autoencoder which will take, as an input, the conditioned CLIP embeddings and generate images through denoising. To get the embeddings, we would use a pre-trained CLIP model. We could either use OpenAI's implementation [here](https://github.com/openai/CLIP), or make use of publicly-available APIs like the one by Hugging Face [here](https://huggingface.co/docs/transformers/model_doc/clip). <br/>
Because both of these models are implemented in PyTorch, we will need to "convert" the embeddings to Tensorflow. Since they are essentially matrices, one way to do it would be to transform PyTorch `Tensor`s into `numpy` arrays, and use Tensorflow's `convert_to_tensor`.  We can then used these tensors in our diffusion model. 
## Model
Our current model relies on an autoencoder structure. We concatenate CLIP embeddings to a representation of an imput image which has already gone through a portion of our encoder, and then put it through a second encoder portion. Our encoder, particularly the first chunk, includes many convolutional layers. Then, we add random noise and put the output through a decoder, which outputs an image. 

## Metrics
Given this task is intrinsically a generative task, it is difficult to find a very clear metric for accuracy. We will be able to use our model's loss to track whether the model learns, but assessment on its performance will primarily be qualitatively assessed (by looking at the generated images). 

In the original paper, the authors use a similar process and assess the model's performance using human evaluation. Participants are asked to assess Photorealism, Caption Similarity, and Diversity. They primarily compare two versions of their model: one which derives embeddings autoregressively from the CLIP embeddings, and one involving a diffusion model. It seems like the authors were trying to outperform GLIDE, which they did for certain assessments and for certain guidance scales. For reference, this parameter assesses the influence of the guiding mechanism during the diffusion process.

#### Base, Target, and Reach Goals
- <strong>Base Goal</strong>: Implementing a functional diffusion model, which could take some kind of embedding (not necessarily CLIP). 
- <strong>Target Goal</strong>: Using CLIP embeddings to influence image generation.
- <strong>Reach Goal </strong>: Being able to influence the image generated with a modification prompt afterwards.

## Ethics
- Why is Deep Learning a good approach to this problem?
  
It is difficult to imagine how a procedural algorithm could generate an image from a prompt. It would be virtually impossible to capture the immense diversity of possible words and prompts, and also the diversity in variations of output images from the same prompt. This task of identifying features in a given image is itself complex (but algorithms exist for that, like bag of words), but the task of directly linking visual representation and textual representation is even more substantial and difficult to achieve. The deep inferences that a deep model is able to make are crucial, and this is why we think Deep Learning is a good approach to the problem.

- What broader societal issues are relevant to your chosen problem space?
  
Generative tasks as a whole pose deep societal issues. The ability of a model to generate virtually any image makes generating fake images very easy, which can be used to spread misinformation over social media. This is particularly dangerous when these generated images involve real people, like political figures. To put it simply, it will be extremely difficult from now on to distinguish what is real from what is fake.

Another important think to consider is how data is used to train these models. In many cases, data is scraped from the Internet, often without the consent of the original artists. It is not fair for creators to have their intellectual property used to train a model people might rely on, instead of requesting service from the original creators and compensating them appropriately.

## Division of Labor
- Ariana: Investigate how to import the CLIP model and integrate it to our pipeline, as well as how to feed the input data. Investigate implementation of loss function for generated images.
- Ayman: Look through the model helping with transforming CLIP embeddings to make them suitable for the specialized image generation task. Figure out shapes/concatenation/feeding of the embeddings.
- Sofia: Investigate implementation of the U-Net architecture / Hierarchical VAE based diffusion models. 
Because the diffusion model is our base goal, we will all virtually be working on implementation of the diffusion part of the project during the first half.
