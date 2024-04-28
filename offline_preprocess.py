import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import preprocess as pp
from pycocotools.coco import COCO
import shutil

def load_coco_data(image_directory, captions_file):
    # Tokens to be matched
    toks = ["cow", "sheep", "cows", "sheeps"]

    print("huh?")
    
    # Initialize COCO with annotations
    coco_captions = COCO(captions_file)

    # Get image IDs
    image_ids = coco_captions.getImgIds()

    images = coco_captions.loadImgs(image_ids)
    to_be_moved = []
    print("Loading", end="")
    for img in images:
      print(".", end="")
      full_fp = os.path.join(image_directory, img["file_name"])
      annotations = [ann['caption'] for ann in coco_captions.loadAnns(coco_captions.getAnnIds(imgIds=img['id'], iscrowd=None))]
      for anno in annotations:
          if "cow" in anno.lower() or "sheep" in anno.lower():
            to_be_moved.append(full_fp)
            break
    
    print("")
    dest = os.path.join(image_directory, "train_offline_preprocess")
    for tbm in to_be_moved:
       shutil.copyfile(tbm, os.path.join(dest, tbm.split("/")[-1]))
    print("Done!")

img_dir = "/Volumes/Ayman Portable SSD/Documents/Coding Stuff/CS-Brown/cs1470/final-project/datasets/train2017"
anno_dir = "/Volumes/Ayman Portable SSD/Documents/Coding Stuff/CS-Brown/cs1470/final-project/datasets/annotations/captions_train2017.json"
load_coco_data(img_dir, anno_dir)