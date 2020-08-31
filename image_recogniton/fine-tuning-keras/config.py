# import the necessary packages
import os
import datetime

# SageMaker home directory
HOME = '/home/ec2-user/SageMaker/evd-image-classifier'

#Sagemaker (model) output path
OUTPUT_PATH = '/home/ec2-user/SageMaker/evd-image-classifier/output' #Isidro

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "ungrouped_images"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "dataset"

# define the names of the training, testing, and validation
# directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

# initialize the list of class label names
CLASSES = ["almost_ready", "not_ready", "ready"]

# set the batch size when fine-tuning
BATCH_SIZE = 32

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"

# set the path to the serialized model after training
current_date = datetime.datetime.now() #ISIDRO
current_date = current_date.strftime("%m_%d_%Y_%H_%M_%S") #ISIDRO
MODEL_PATH = os.path.sep.join(["output", "image_classification_"+current_date+".model"])

# define the path to the output training history plots
UNFROZEN_PLOT_PATH = os.path.sep.join(["output", "unfrozen"])
WARMUP_PLOT_PATH = os.path.sep.join(["output", "warmup"])

# select number of epochs to run in first and second pass
WARMUP_EPOCHS = 2          #50 is standard
UNFROZEN_EPOCHS = 2         #20 is standard

#path to s3 bucket
BUCKET = 'us-east-dev-evd-training-data'
#path to classifier directory in S3
LOCATION_MODELS = "MODELS/classifier" 