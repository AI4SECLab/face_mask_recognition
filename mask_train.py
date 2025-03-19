import os
import cv2
from mask_detection import *
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mask_detection import *
from sklearn.neural_network import MLPClassifier
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
from tqdm import tqdm
import timm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Hàm lấy danh sách ảnh từ thư mục
def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]


def face_detection(image, sess, node_dict, conf_thresh=0.5):
    #----image processing
    tf_input = node_dict['input']
    model_shape = tf_input.shape
    img_resized = cv2.resize(image, (int(model_shape[2]), int(model_shape[1])))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized = img_resized.astype('float32')
    img_resized /= 255

    #----mask detection
    detection_bboxes = node_dict['detection_bboxes']
    detection_scores = node_dict['detection_scores']
    y_bboxes_output, y_cls_output = sess.run([detection_bboxes, detection_scores],
                                            feed_dict={tf_input: np.expand_dims(img_resized, axis=0)})
    #remove the batch dimension, for batch is always 1 for inference.
    #====anchors config
    feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
    anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
    anchor_ratios = [[1, 0.62, 0.42]] * 5
    anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
    # for inference , the batch size is 1, the model output shape is [1, N, 4],
    # so we expand dim for anchors to [1, anchor_num, 4]
    anchors_exp = np.expand_dims(anchors, axis=0)
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    # keep_idx is the alive bounding box after nms.
    iou_thresh = 0.4
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                bbox_max_scores,
                                                conf_thresh=conf_thresh,
                                                iou_thresh=iou_thresh,
                                                )
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        h, w, _ = img_resized.shape
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * w))
        ymin = max(0, int(bbox[1] * h))
        xmax = min(int(bbox[2] * w), w)
        ymax = min(int(bbox[3] * h), h)
        # Cắt phần khuôn mặt từ ảnh gốc
        face_crop = img_resized[ymin:ymax, xmin:xmax]
        return face_crop

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
import warnings
import torch
from torchvision import models, transforms
from sklearn.tree import DecisionTreeClassifier

def get_image_tensor(images):
    # Define preprocessing
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Apply preprocess
    image_tensor = torch.stack([preprocess(image) for image in images])

    return image_tensor

x = 1

# from PIL import Image
def extract_mask_features(image, sess, node_dict):
    """Extract features from mask detection model's intermediate layer"""
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Tăng kích thước để lấy đặc trưng tốt hơn
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():  # Disable gradient computation
        embedding = facenet_model(image_tensor).squeeze().detach().cpu().numpy()  # Detach and convert to NumPy
    return embedding  # Return first batch

from facenet_pytorch import InceptionResnetV1
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

def train(training_dir, pb_path, node_dict):
    sess, node_dict = model_restore_from_pb(pb_path, node_dict)
    # Extract features for each person using mask detection model
    print("Extracting features from mask detection model...")
    X_features = []
    y_labels = []
    for person_name in tqdm(os.listdir(training_dir)):
        person_dir = os.path.join(training_dir, person_name)
        if os.path.isdir(person_dir):
            for img_path in image_files_in_folder(person_dir):
                # try:
                    image = cv2.imread(img_path)
                    face = face_detection(image, sess, node_dict)
                    if face is not None:
                        # Extract features from mask detection model
                        features = extract_mask_features(face, sess, node_dict)
                        # plt.imshow(face)
                        # plt.show()
                        # return
                        X_features.append(features)
                        y_labels.append(person_name)
                # except Exception as e:
                #     print(f"Error processing {img_path}: {e}")
                #     continue
    
    X_features = np.array(X_features)
    X_features = X_features.reshape(X_features.shape[0], -1)
    y_labels = np.array(y_labels)
    
    # Training classifier replaced with similarity measurement
    print("Computing centroids for each person...")
    person_embeddings = {}
    for person in np.unique(y_labels):
        idx = np.where(y_labels == person)[0]
        centroid = np.mean(X_features[idx], axis=0)
        person_embeddings[person] = centroid
    # Set threshold for Euclidean distance (adjust as needed)
    threshold = 0.7
    
    with open('face_classifier.pkl', 'wb') as f:
        pickle.dump({
            'person_embeddings': person_embeddings,
            'threshold': threshold,
            'feature_shape': X_features.shape[1]
        }, f)
    
    print("Training completed!")
    return person_embeddings, threshold

if __name__ == '__main__':
    node_dict = {'input':'data_1:0',
                'detection_bboxes':'loc_branch_concat_1/concat:0',
                'detection_scores':'cls_branch_concat_1/concat:0'}
    # train("./training_dataset", "./face_mask_detection.pb", node_dict)
    train("./dataset", "./face_mask_detection.pb", node_dict)