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
        extend = 50
        face_crop = img_resized[ymin:ymax, xmin:xmax]
        return face_crop
    
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
import warnings
import torch
from torchvision import models, transforms
from sklearn.tree import DecisionTreeClassifier

def get_resnet():
    # Download model
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    resnet.cpu()

    for param in resnet.parameters():
        param.requires_grad = False
    return resnet

def extract_features(X, resnet):
    result = np.empty((len(X), 2048))
    for i, data in enumerate(X):
        output = resnet(data.unsqueeze(0).cpu())
        output = torch.flatten(output, 1)
        result[i] = output[0].cpu().numpy()
    return result

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

def extract_feature_from_image(image, resnet):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_tensor = torch.stack([preprocess(image)])
    result = extract_features(image_tensor, resnet)
    return result
import math
class ArcMarginModel(nn.Module):
    def __init__(self, num_classes, emb_size=512, s=30.0, m=0.50):
        super(ArcMarginModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        # Modify last layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(emb_size, emb_size)
        )
        self.margin = nn.Parameter(torch.FloatTensor([m]))
        self.scale = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        x = self.backbone(x)
        x = F.normalize(x)
        w = F.normalize(self.weight)
        
        if label is None:  # For inference
            return x
            
        cosine = F.linear(x, w)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output

def create_pairs(X, y):
    pairs = []
    labels = []
    
    n_classes = len(np.unique(y))
    class_indices = [np.where(y == i)[0] for i in range(n_classes)]
    
    for idx1 in range(len(X)):
        current_class = y[idx1]
        # Positive pair
        idx2 = random.choice(class_indices[current_class])
        while idx2 == idx1:
            # print(idx1, idx2, len(class_indices[current_class]) + 1)
            idx2 = (idx2 + 1) % (len(class_indices[current_class]) + 1)
            # idx2 = random.choice(class_indices[current_class])
        pairs.append([idx1, idx2])  # Store indices instead of actual images
        labels.append(0.0)
        
        # Negative pair
        neg_class = random.randint(0, n_classes-1)
        while neg_class == current_class:
            neg_class = random.randint(0, n_classes-1)
            # print(neg_class)
        idx2 = random.choice(class_indices[neg_class])
        pairs.append([idx1, idx2])
        labels.append(1.0)
    
    return np.array(pairs), np.array(labels)

def train(training_dir, pb_path, node_dict):
    sess, node_dict = model_restore_from_pb(pb_path, node_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load and process images
    print("Loading and processing images:")
    X, y = [], []
    person_folders = [f for f in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, f))]
    for person_name in tqdm(person_folders, desc="Processing persons"):
        person_folder = os.path.join(training_dir, person_name)
        for image_path in image_files_in_folder(person_folder):
            image = cv2.imread(image_path)
            face = face_detection(image, sess, node_dict)
            if face is not None:
                face = cv2.resize(face, (128, 128))
                X.append(face)
                y.append(person_name)
                break  # Only take first valid image

    X = np.array(X)
    y = np.array(y)
    
    # Data preprocessing
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Convert labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    np.save('./classes.npy', encoder.classes_)

    # Create and train model
    num_classes = len(np.unique(y))
    model = ArcMarginModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    n_epochs = 20
    batch_size = 32
    best_loss = float('inf')

    print("\nStarting training:")
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        model.train()
        total_loss = 0
        batch_indices = list(range(0, len(X), batch_size))
        random.shuffle(batch_indices)
        
        batch_pbar = tqdm(batch_indices, desc=f"Batch", leave=False)
        for i in batch_pbar:
            batch_X = X[i:i+batch_size]
            batch_y = y_encoded[i:i+batch_size]
            
            # Apply transformations to images
            inputs = torch.stack([train_transform(x) for x in batch_X]).to(device)
            labels = torch.LongTensor(batch_y).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            batch_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = total_loss/len(batch_indices)
        tqdm.write(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './arcface_model.pth')

    # Save reference features
    if not os.path.exists('./reference_features'):
        os.makedirs('./reference_features')
        
    model.eval()
    with torch.no_grad():
        for person_name in encoder.classes_:
            person_idx = np.where(y == person_name)[0][0]
            person_tensor = transforms.ToTensor()(X[person_idx]).unsqueeze(0).to(device)
            features = model(person_tensor)
            np.save(f'./reference_features/{person_name}.npy', features.cpu().numpy())

    return model

def extract_feature_from_image(image, model):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        feature = model.forward_one(image_tensor)
    return feature.cpu().numpy()

if __name__ == '__main__':
    node_dict = {'input':'data_1:0',
                'detection_bboxes':'loc_branch_concat_1/concat:0',
                'detection_scores':'cls_branch_concat_1/concat:0'}
    train("./training_dataset", "./face_mask_detection.pb", node_dict)