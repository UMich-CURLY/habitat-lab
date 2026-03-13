import torch
import torch.nn as nn


def make_resnet_backbone(num_classes=2, pretrained=False, in_channels=3):
    """
    Create a small ResNet-based classifier for binary gating.
    Falls back to a tiny convnet if torchvision isn't available.

    Returns: nn.Module that maps (B, C, H, W) -> logits (B, num_classes)
    """
    try:
        from torchvision.models import resnet18
        model = resnet18(pretrained=pretrained)
        # adjust input conv if in_channels != 3
        if in_channels != 3:
            # replace first conv
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.conv1 = conv1
        # replace final fc
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
        return model
    except Exception:
        # fallback tiny conv net
        class TinyConvNet(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.fc = nn.Linear(128, num_classes)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return TinyConvNet(num_classes)


if __name__ == "__main__":
    # quick smoke test
    m = make_resnet_backbone(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(y.shape)
