import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import faiss  # Facebook AI Similarity Search
import cv2

# Load feature list and filenamesm
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a Sequential model with GlobalMaxPooling2D
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Preprocess the image
img = image.load_img('sample/shirt.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)

# Extract the features of the image using ResNet50
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Faiss index setup
dimension = feature_list.shape[1]  # 2048 for ResNet50 features
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance)
index.add(feature_list)  # Adding the dataset features to the Faiss index

# Search for the 6 nearest neighbors (including the query itself)
distances, indices = index.search(np.array([normalized_result]), k=6)

# Printing the indices of the 5 most similar items (excluding the query image itself)
print(indices)

# Display the results
for file in indices[0][1:6]:  # Skiping the first one (which is the query image itself)
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)
