import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from siamese_network import SiameseNetwork, triplet_loss, calculate_accuracy
from torchvision import transforms
import random
from tqdm import tqdm
from mask_train import face_detection, image_files_in_folder
import os

class TripletDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.person_to_images = {}
        self.people = []
        
        # Load all images
        for person in os.listdir(data_dir):
            person_dir = os.path.join(data_dir, person)
            if os.path.isdir(person_dir):
                self.person_to_images[person] = []
                self.people.append(person)
                for img_path in image_files_in_folder(person_dir):
                    self.person_to_images[person].append(img_path)
    
    def __len__(self):
        return sum(len(imgs) for imgs in self.person_to_images.values())
    
    def __getitem__(self, idx):
        # Select anchor person
        anchor_person = random.choice(self.people)
        # Select different person for negative
        negative_person = random.choice([p for p in self.people if p != anchor_person])
        
        # Get images
        anchor_img = random.choice(self.person_to_images[anchor_person])
        positive_img = random.choice([img for img in self.person_to_images[anchor_person] if img != anchor_img])
        negative_img = random.choice(self.person_to_images[negative_person])
        
        # Load and transform images
        anchor = self.load_image(anchor_img)
        positive = self.load_image(positive_img)
        negative = self.load_image(negative_img)
        
        return anchor, positive, negative
    
    def load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image

def train_siamese(data_dir, num_epochs=50):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and track best accuracy
    model = SiameseNetwork().to(device)
    best_accuracy = 0.0
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TripletDataset(data_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Split into train/val
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-01)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_metrics = []
        progress_bar = tqdm(train_loader)
        
        for batch in progress_bar:
            # Unpack batch and move to device
            anchor, positive, negative = [x.to(device) for x in batch]
            
            # Training step
            loss, metrics = model.train_step((anchor, positive, negative), optimizer)
            
            # Update tracking
            total_loss += loss
            all_metrics.append(metrics)
            
            # Update progress bar
            accuracy, ap_mean, an_mean, ap_std, an_std = metrics
            progress_bar.set_description(
                f"Epoch {epoch+1} Loss: {loss:.4f} "
                f"Acc: {accuracy:.4f} "
                f"AP: {ap_mean:.4f} AN: {an_mean:.4f}"
            )
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_metrics = np.mean(all_metrics, axis=0)
        
        # Move model to eval mode and evaluate
        model.eval()
        val_loss = 0
        val_metrics_list = []
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_anchor, val_positive, val_negative = [x.to(device) for x in val_batch]
                ap_dist, an_dist = model(val_anchor, val_positive, val_negative)
                val_batch_loss = triplet_loss(ap_dist, an_dist, model.margin)
                val_batch_metrics = calculate_accuracy(ap_dist, an_dist)
                
                val_loss += val_batch_loss.item()
                val_metrics_list.append(val_batch_metrics)
        
        val_loss = val_loss / len(val_loader)
        val_metrics = np.mean(val_metrics_list, axis=0)
        val_accuracy = val_metrics[0]

        # Print metrics and save model
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_metrics[0]:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'siamese_model_best.pth')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_accuracy': best_accuracy
            }, f'siamese_checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'siamese_model_final.pth')
    return model

if __name__ == "__main__":
    train_siamese("./training_dataset", num_epochs=256)
