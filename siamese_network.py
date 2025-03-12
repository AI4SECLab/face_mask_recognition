import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 
import timm
from typing import Dict, List
import numpy as np

class DistanceLayer(nn.Module):
    def forward(self, anchor_embedding, positive_embedding, negative_embedding):
        # Simplified distance calculation like TensorFlow version
        ap_distance = torch.sum(torch.square(anchor_embedding - positive_embedding), dim=-1)
        an_distance = torch.sum(torch.square(anchor_embedding - negative_embedding), dim=-1)
        return ap_distance, an_distance

class SiameseEncoder(nn.Module):
    def __init__(self, input_shape=(224, 224, 3)):
        super().__init__()
        # Use Xception as base model
        self.base_model = timm.create_model('xception', pretrained=True, num_classes=0)
        
        # Freeze early layers (matching TensorFlow version)
        for i, (name, param) in enumerate(self.base_model.named_parameters()):
            if i < len(list(self.base_model.parameters())) - 27:
                param.requires_grad = False
        
        # Match TensorFlow layer structure
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)  # L2 normalization
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

class SiameseNetwork(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.encoder = SiameseEncoder()
        self.distance_layer = DistanceLayer()
        self.margin = margin
        
        # Tracking metrics like TensorFlow
        self.loss_tracker = []
        self.current_loss = 0.0

    def forward(self, anchor, positive, negative):
        # Get embeddings using single encoder (shared weights)
        anchor_embedding = self.encoder(anchor)
        positive_embedding = self.encoder(positive)
        negative_embedding = self.encoder(negative)
        
        # Get distances using distance layer
        return self.distance_layer(
            anchor_embedding,
            positive_embedding, 
            negative_embedding
        )

    def train_step(self, batch, optimizer):
        """Training step with loss computation"""
        anchor, positive, negative = batch
        optimizer.zero_grad()
        
        ap_dist, an_dist = self(anchor, positive, negative)
        loss = triplet_loss(ap_dist, an_dist, self.margin)
        
        loss.backward()
        optimizer.step()
        
        # Update metrics
        self.current_loss = loss.item()
        self.loss_tracker.append(self.current_loss)
        
        metrics = calculate_accuracy(ap_dist, an_dist)
        return self.current_loss, metrics

    def get_embeddings(self, x):
        return self.encoder(x)

    def evaluate(self, test_loader):
        """Evaluate model on test data"""
        self.eval()
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch in test_loader:
                anchor, positive, negative = batch
                ap_dist, an_dist = self(anchor, positive, negative)
                loss = triplet_loss(ap_dist, an_dist, self.margin)
                metrics = calculate_accuracy(ap_dist, an_dist)
                
                total_loss += loss.item()
                all_metrics.append(metrics)
        
        # Calculate average metrics
        avg_metrics = np.mean(all_metrics, axis=0)
        avg_loss = total_loss / len(test_loader)
        
        self.train()
        return avg_loss, avg_metrics

def triplet_loss(ap_dist, an_dist, margin=1.0):
    """
    Compute triplet loss
    ap_dist: Anchor-positive distances
    an_dist: Anchor-negative distances
    margin: Margin for triplet loss
    """
    loss = torch.clamp(ap_dist - an_dist + margin, min=0.0)
    return torch.mean(loss)

def calculate_accuracy(ap_dist, an_dist):
    """
    Calculate accuracy based on distances
    Returns accuracy, mean distances and standard deviations
    """
    with torch.no_grad():
        # Compute accuracy
        accuracy = torch.mean((ap_dist < an_dist).float())
        
        # Get statistics
        ap_mean = torch.mean(ap_dist)
        an_mean = torch.mean(an_dist)
        ap_std = torch.std(ap_dist)
        an_std = torch.std(an_dist)
        
        return accuracy.item(), ap_mean.item(), an_mean.item(), ap_std.item(), an_std.item()
