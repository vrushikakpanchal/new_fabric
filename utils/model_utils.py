import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CombinedClassifier(nn.Module):
    def __init__(self, num_defect_classes, num_fabric_classes):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        feat_dim = base.fc.in_features

        self.defect_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_defect_classes)
        )
        self.fabric_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_fabric_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.defect_head(x), self.fabric_head(x)
