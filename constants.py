
import os
ROOT_DIR=os.path.dirname(os.path.realpath(__file__))
DATA_DIR=os.path.join(ROOT_DIR,"data","polyvore_outfits") ##Directory of dataset
TENSORBOARD_DIR=os.path.join(ROOT_DIR,"logs/tb")
IMAGES_DATA_DIR=os.path.join(DATA_DIR,"images")
DISJOINT_DATA_DIR=os.path.join(DATA_DIR,"disjoint")
JOINT_DATA_DIR=os.path.join(DATA_DIR,"joint")

MAX_OUTFIT_SIZE=5
