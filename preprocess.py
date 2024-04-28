import os
import tensorflow as tf
from pycocotools.coco import COCO
import clip_wrapper as cw

def load_coco_data(image_directory, captions_file):
    # Initialize COCO with annotations
    coco = COCO(captions_file)

    # Get image IDs
    image_ids = coco.getImgIds()

    # Load images (get filepaths, and associate with captions)
    images = coco.loadImgs(image_ids)
    filepaths_and_captions = []
    for img in images:
      full_fp = os.path.join(image_directory, img["file_name"])
      annotations = [ann['caption'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=img['id'], iscrowd=None))]
      filepaths_and_captions.append((full_fp, annotations))

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
       return image, captions

    # Create a dataset of ONLY images
    dataset = dataset.map(load_and_preprocess_image)

    # Define Python Function to get image embeddings (this will return a numpy array)
    def get_clip_embeddings(images):
        # If single image
        if len(images.shape) == 3:
            return cw.batch_get_image_encodings(tf.expand_dims(images, axis=0)) 
        else:
            return cw.batch_get_image_encodings(images)
    
    # py_function to use the tensors in the dataset to get the embeddings
    def tf_py_function_clip_embeddings(images, captions):
        clip_embeddings = tf.py_function(get_clip_embeddings, [images], tf.float32)
        # clip_embeddings.set_shape((None, 512))
        return images, clip_embeddings
    
    dataset = dataset.map(tf_py_function_clip_embeddings)

    return dataset

def get_64x64_images(dataset):
    def resize(image, _):
        return tf.image.resize(image, [64, 64])
    return dataset.map(resize)

def get_64x64_images_and_embeddings(dataset):
    def resize(image, embeddings):
        return tf.image.resize(image, [64, 64]), embeddings
    return dataset.map(resize)
