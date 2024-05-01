import os
import tensorflow as tf
from bert_wrapper import SequenceTokenizer
import clip_wrapper as cw
import glob
from pycocotools.coco import COCO


def load_coco_data(image_directory, captions_file, categories_file):
    # Initialize COCO with annotations
    coco_captions = COCO(captions_file)

    # Initialize coco with categories
    # coco_categories = COCO(categories_file)

    # Get category IDs for cows and sheep
    # cat_ids = coco_categories.getCatIds(catNms=["cat", "sheep"])

    # print("cat ids: ", cat_ids)

    full_paths = glob.glob(os.path.join(image_directory, "*.jpg"))
    names = [p.split("/")[-1].replace(".jpg", "") for p in full_paths]
    ids_to_get = [int(n) for n in names]

    print("ids to get: ", ids_to_get)

    # Get image IDs
    image_ids = coco_captions.getImgIds()

    # print("img ids: ", image_ids)

    # Load images (get filepaths, and associate with captions)
    images = coco_captions.loadImgs(image_ids)
    filepaths_and_captions = []
    for img in images:
      full_fp = os.path.join(image_directory, img["file_name"])
      annotations = [ann['caption'] for ann in coco_captions.loadAnns(coco_captions.getAnnIds(imgIds=img['id'], iscrowd=None))]
      for anno in annotations:
          if "cow" in anno.lower() or "sheep" in anno.lower():
            filepaths_and_captions.append((full_fp, annotations))
            break
    # Create a Tensorflow dataset from the filepaths and annotations
    dataset = tf.data.Dataset.from_generator(
        lambda: filepaths_and_captions,
        output_types=(tf.string, tf.string),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([None]))
    )

    def load_and_preprocess_image(path, captions):
       image = tf.io.read_file(path)
       image = tf.image.decode_jpeg(image, channels=3)
       image = tf.image.resize(image, [224, 224])
       image = image / 255.0
       return image, captions

    # Create a dataset of ONLY images
    dataset = dataset.map(load_and_preprocess_image)

    # Define Python Function to get image embeddings (this will return a numpy array)
    def get_clip_im_embeddings(images):
        # If single image
        if len(images.shape) == 3:
            return cw.batch_get_image_encodings(tf.expand_dims(images, axis=0)) 
        else:
            return cw.batch_get_image_encodings(images)
    
    # py_function to use the tensors in the dataset to get the embeddings
    def tf_py_function_clip_im_embeddings(images, captions):
        clip_im_embeddings = tf.py_function(get_clip_im_embeddings, [images], tf.float32)
        clip_im_embeddings.set_shape((1, 512))
        return images, clip_im_embeddings, captions
    
    dataset = dataset.map(tf_py_function_clip_im_embeddings)

    def get_clip_text_embeddings(captions):
        # If single image
        if len(captions.shape) == 3:
            return cw.get_text_encoding(tf.expand_dims(captions, axis=0)) 
        else:
            return cw.get_text_encoding(captions)
    
    def tf_py_function_clip_text_embeddings(images, clip_im_embeds, captions):
        clip_txt_embeddings = tf.py_function(get_clip_text_embeddings, [captions], tf.float32)
        clip_txt_embeddings.set_shape((1, 512))
        return images, clip_im_embeds, captions, clip_txt_embeddings
    
    dataset = dataset.map(tf_py_function_clip_text_embeddings)
    
    def get_bert_sequence(captions):
        tokenizer = SequenceTokenizer()
        tokens = tokenizer.tokenize(captions)
        return tokens
    
    def tf_py_function_bert_embeddings(images, clip_im_embeds, captions, clip_txt_embeds):
        bert_embeddings = tf.py_function(get_bert_sequence, [captions], tf.float32)
        return images, clip_im_embeds, captions, clip_txt_embeds, bert_embeddings
    
    dataset = dataset.map(tf_py_function_bert_embeddings)

    return dataset

def get_64x64_images(dataset):
    def resize(image, *args):
        return tf.image.resize(image, [64, 64])
    return dataset.map(resize)

def get_64x64_images_and_embeddings(dataset):
    def resize(image, embeddings, *args):
        return tf.image.resize(image, [64, 64]), embeddings
    return dataset.map(resize)

def get_64x64_images_and_text_embeddings(dataset):
    def resize(image, clip_im_embeds, captions, clip_txt_embeds, bert_embeds):
        return tf.image.resize(image, [64, 64]), clip_txt_embeds
    return dataset.map(resize)

def get_64x64_images_im_embeddings_and_tokens_seq(dataset):
    def resize(image, clip_im_embeds, captions, clip_txt_embeds, bert_embeds):
        return tf.image.resize(image, [64, 64]), clip_im_embeds, bert_embeds
    return dataset.map(resize)

def get_64x64_images_im_embeddings_and_tokens_seq(dataset):
    def resize(image, clip_im_embeds, captions, clip_txt_embeds, bert_embeds):
        return tf.image.resize(image, [64, 64]), clip_txt_embeds, bert_embeds
    return dataset.map(resize)
