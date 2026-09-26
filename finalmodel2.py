import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib  # Import joblib for saving and loading models

def load_images(dataset_folder):
    data = []
    classification = []
    class_labels = ['1', '2', '3', '4']
    for label in class_labels:  # iterate over folder names
        label_folder = os.path.join(dataset_folder, label)
        if not os.path.isdir(label_folder):
            continue  # skip if not a folder
        for file in os.listdir(label_folder):
            if file.startswith('.'):
                continue  # skip hidden files like .DS_Store
            image_path = os.path.join(label_folder, file)
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Skipping file {image_path}: {e}")
                continue
            img_as_arr = np.array(image, dtype=np.float32) / 255.0
            data.append(img_as_arr.flatten())
            classification.append(label)
    return data, classification


train_data, train_labels = load_images("drums_train")
test_data, test_labels = load_images("drums_test")

# StandardScaler
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data  = scaler.transform(test_data)

# PCA
pca = PCA(n_components=200)
train_data = pca.fit_transform(train_data)
test_data  = pca.transform(test_data)

# Train SVC
model_path = "svc_model_v1.joblib"
if not os.path.exists(model_path):
    svc_model = SVC(kernel='poly', degree=2, coef0=1.0, probability=False)
    svc_model.fit(train_data, train_labels)
    # Save the model
    joblib.dump(svc_model, model_path)
else:
    # Load the model
    svc_model = joblib.load(model_path)

# Test
y_hat = svc_model.predict(test_data)
wrong = np.sum(y_hat != test_labels)
count = len(test_labels)
print('count:', count)
print('wrong:', wrong)
print(f"Percent Error: {100 * wrong / count}%")