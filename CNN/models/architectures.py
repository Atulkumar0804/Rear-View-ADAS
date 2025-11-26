"""
CNN Model Architectures for Vehicle Detection & Distance Estimation

Implements:
1. Custom MobileNet-inspired lightweight model
2. SqueezeNet-inspired efficient model  
3. ResNet-inspired deep model
4. Transfer learning versions with fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ================================
# 1. MobileNet-Inspired Architecture
# ================================

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (MobileNet building block)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu6(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu6(x)
        return x

class MobileNetInspired(nn.Module):
    """Lightweight MobileNet-inspired model for vehicle classification"""
    def __init__(self, num_classes=15):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Depthwise separable convolutions
        self.dw_layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
            # 5 more layers with 512 channels
            *[DepthwiseSeparableConv(512, 512, stride=1) for _ in range(5)],
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1),
        )
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),  # Increased dropout for large dataset
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ================================
# 2. SqueezeNet-Inspired Architecture
# ================================

class FireModule(nn.Module):
    """Fire module from SqueezeNet"""
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        
        # Squeeze layer
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Expand layers
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNetInspired(nn.Module):
    """Efficient SqueezeNet-inspired model"""
    def __init__(self, num_classes=15):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            FireModule(512, 64, 256, 256),
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

# ================================
# 3. ResNet-Inspired Architecture
# ================================

class ResidualBlock(nn.Module):
    """Residual block from ResNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ResNetInspired(nn.Module):
    """Deep ResNet-inspired model"""
    def __init__(self, num_classes=15, num_blocks=[2, 2, 2, 2]):
        super().__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc = nn.Linear(512, num_classes)  # ResidualBlock doesn't have expansion
        
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout before final layer
        x = self.fc(x)
        return x

# ================================
# 4. Transfer Learning Models
# ================================

class TransferLearningModel(nn.Module):
    """Transfer learning wrapper with fine-tuning support"""
    def __init__(self, model_name='mobilenet_v2', num_classes=6, freeze_backbone=True):
        super().__init__()
        
        self.model_name = model_name
        
        if model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
            
        elif model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier
            if model_name in ['mobilenet_v2', 'efficientnet_b0']:
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
            else:
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self, num_layers=-1):
        """Unfreeze backbone layers for fine-tuning"""
        if num_layers == -1:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last num_layers
            layers = list(self.backbone.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

# ================================
# 5. LSTM-based Distance Estimator
# ================================

class LSTMDistanceEstimator(nn.Module):
    """LSTM model for estimating distance changes from image sequences"""
    def __init__(self, num_classes=3):  # approaching, stationary, receding
        super().__init__()
        
        # CNN feature extractor (shared for all frames)
        self.cnn = models.mobilenet_v2(pretrained=True).features
        
        # Freeze CNN
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, 
                           num_layers=2, batch_first=True, dropout=0.3)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()
        
        # Extract features for each frame
        features = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]
            feat = self.cnn(frame)
            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
            features.append(feat)
        
        # Stack features
        features = torch.stack(features, dim=1)  # (batch, seq_len, 1280)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Classify
        output = self.classifier(last_hidden)
        
        return output

# ================================
# Model Factory
# ================================

def create_model(model_type, num_classes=15, **kwargs):
    """Factory function to create models"""
    
    # Filter out 'pretrained' from kwargs for TransferLearningModel
    # (it always uses pretrained=True internally)
    transfer_kwargs = {k: v for k, v in kwargs.items() if k != 'pretrained'}
    
    models_dict = {
        'mobilenet_inspired': MobileNetInspired,
        'squeezenet_inspired': SqueezeNetInspired,
        'resnet_inspired': ResNetInspired,
        'transfer_mobilenet': lambda nc: TransferLearningModel('mobilenet_v2', nc, **transfer_kwargs),
        'transfer_resnet18': lambda nc: TransferLearningModel('resnet18', nc, **transfer_kwargs),
        'transfer_resnet50': lambda nc: TransferLearningModel('resnet50', nc, **transfer_kwargs),
        'transfer_efficientnet': lambda nc: TransferLearningModel('efficientnet_b0', nc, **transfer_kwargs),
        'lstm_distance': LSTMDistanceEstimator,
    }
    
    if model_type not in models_dict:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models_dict[model_type](num_classes)

if __name__ == "__main__":
    # Test all models
    print("Testing model architectures...")
    
    batch_size = 4
    img = torch.randn(batch_size, 3, 224, 224)
    
    models_to_test = [
        'mobilenet_inspired',
        'squeezenet_inspired',
        'resnet_inspired',
        'transfer_mobilenet',
    ]
    
    for model_name in models_to_test:
        print(f"\n{model_name}:")
        model = create_model(model_name, num_classes=6)
        output = model(img)
        print(f"  Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test LSTM model
    print(f"\nlstm_distance:")
    seq = torch.randn(batch_size, 5, 3, 224, 224)
    model = create_model('lstm_distance', num_classes=3)
    output = model(seq)
    print(f"  Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
