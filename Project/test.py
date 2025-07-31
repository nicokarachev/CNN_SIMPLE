import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Prepare dataset
local_path = f"../Data/Dataset"
image_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(root=local_path, transform=image_transform)

train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)




#CNN Model with feature map hooks
class MyCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)     # [B, 3, 128, 128] -> [B, 16, 128, 128]
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)    # -> [B, 32, 128, 128]
        self.pool1 = nn.MaxPool2d(2, 2)                 # -> [B, 32, 64, 64]
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)    # -> [B, 64, 64, 64]
        self.pool2 = nn.MaxPool2d(2, 2)                 # -> [B, 64, 32, 32]
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.feature_maps = []  # Save for visualization
        x = nn.ReLU()(self.conv1(x)); self.feature_maps.append(x)
        x = nn.ReLU()(self.conv2(x)); self.feature_maps.append(x)
        x = self.pool1(x)
        x = nn.ReLU()(self.conv3(x)); self.feature_maps.append(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = MyCNNModel().to(device)

# ✅ Loss + Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")



def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            correct += (preds.squeeze() == labels.float()).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')




def visualize_feature_maps(model, image_tensor, save_dir="./img"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        model(image_tensor.unsqueeze(0).to(device))  # Add batch dimension

        for layer_idx, fmap in enumerate(model.feature_maps):
            fmap_cpu = fmap[0].cpu()  # Remove batch dimension
            layer_dir = os.path.join(save_dir, f"layer_{layer_idx+1}")
            os.makedirs(layer_dir, exist_ok=True)

            for ch_idx in range(fmap_cpu.shape[0]):
                plt.imshow(fmap_cpu[ch_idx], cmap='viridis')
                plt.axis('off')

                save_path = os.path.join(layer_dir, f"feature_{ch_idx}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()


# ✅ Run Training + Validation
# train(model, train_loader, criterion, optimizer, 5)
# validate(model, val_loader)
# Save entire model
torch.save(model, "my_cnn_model_entire.pth")

# Load entire model
model = torch.load("my_cnn_model_entire.pth", weights_only=False)
model.eval()

# ✅ Visualize Feature Maps on a Sample
sample_img, _ = dataset[0]
print(len(sample_img))
visualize_feature_maps(model, sample_img)