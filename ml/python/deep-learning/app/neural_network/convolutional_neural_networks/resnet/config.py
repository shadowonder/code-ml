# some training parameters
system_splitter = "\\"  # windows, 更改下本地操作

EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 5
image_height = 224
image_width = 224
channels = 3
# absolute path
folder = "E:\\Workspace\\ml\\code-ml\\ml\\python\\deep-learning\\app\\neural_network\\convolutional_neural_networks\\resnet\\"
save_model_dir = folder + "saved_model" + system_splitter + "model"
dataset_dir = folder + "dataset" + system_splitter
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"

# choose a network
# model = "resnet18"
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"
