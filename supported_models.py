from torch import nn, optim
from torchvision.models import (
    VGG13_Weights, DenseNet121_Weights, DenseNet161_Weights, ResNet18_Weights,
    ResNet50_Weights, Inception_V3_Weights, EfficientNet_B0_Weights, MobileNet_V2_Weights
)

model_info = {
    'vgg13': {
        'weights': VGG13_Weights.DEFAULT,
        'in_features': lambda model: model.classifier[0].in_features,
        'criterion': nn.NLLLoss,
        'optimizer': lambda model, lr: optim.Adam(model.classifier.parameters(), lr=lr)
    },
    'densenet121': {
        'weights': DenseNet121_Weights.DEFAULT,
        'in_features': lambda model: model.classifier.in_features,
        'criterion': nn.NLLLoss,
        'optimizer': lambda model, lr: optim.Adam(model.classifier.parameters(), lr=lr)
    },
    'densenet161': {
        'weights': DenseNet161_Weights.DEFAULT,
        'in_features': lambda model: model.classifier.in_features,
        'criterion': nn.NLLLoss,
        'optimizer': lambda model, lr: optim.Adam(model.classifier.parameters(), lr=lr)
    },
    'resnet18': {
        'weights': ResNet18_Weights.DEFAULT,
        'in_features': lambda model: model.fc.in_features,
        'criterion': nn.CrossEntropyLoss,
        'optimizer': lambda model, lr: optim.Adam(model.fc.parameters(), lr=lr)
    },
    'resnet50': {
        'weights': ResNet50_Weights.DEFAULT,
        'in_features': lambda model: model.fc.in_features,
        'criterion': nn.CrossEntropyLoss,
        'optimizer': lambda model, lr: optim.Adam(model.fc.parameters(), lr=lr)
    },
    'inception_v3': {
        'weights': Inception_V3_Weights.DEFAULT,
        'in_features': lambda model: model.fc.in_features,
        'criterion': nn.CrossEntropyLoss,
        'optimizer': lambda model, lr: optim.Adam(model.fc.parameters(), lr=lr)
    },
    'efficientnet_b0': {
        'weights': EfficientNet_B0_Weights.DEFAULT,
        'in_features': lambda model: model.classifier[1].in_features,
        'criterion': nn.CrossEntropyLoss,
        'optimizer': lambda model, lr: optim.Adam(model.classifier.parameters(), lr=lr)
    },
    'mobilenet_v2': {
        'weights': MobileNet_V2_Weights.DEFAULT,
        'in_features': lambda model: model.classifier[1].in_features,
        'criterion': nn.CrossEntropyLoss,
        'optimizer': lambda model, lr: optim.Adam(model.classifier.parameters(), lr=lr)
    },
}