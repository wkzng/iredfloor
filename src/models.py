import torch.nn.functional as F
from torch import nn



class WideMLP(nn.Module):
    def __init__(self, input_size:int=28*28, hidden_size:int=128, output_size:int=10, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Flatten(),
          nn.Linear(input_size, hidden_size),
          nn.ReLU(),
          nn.Linear(hidden_size, output_size)
      )
    def forward(self, x):
        return self.layers(x)


class DeepMLP(nn.Module):
    def __init__(self, input_size:int=28*28, depth:int=1, hidden_size:int=128, output_size:int=10, **kwargs):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(input_size, hidden_size), nn.ReLU(),]
        for _ in range(depth):
            layer = [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ]
            layers.extend(layer)
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""
    def __init__(self, input_channels=1, num_classes=10, **kwargs):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)              # 28x28 -> 28x28

        # After 2 max pools: 28x28 -> 14x14 -> 7x7
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization (optional)
        self.dropout = nn.Dropout(0.0)  # Start with no dropout for clean comparison

    def forward(self, x):
        # Conv layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64*7*7)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class DeepCNN(nn.Module):
    """Deeper CNN for more complex dynamics"""
    def __init__(self, input_channels=1, num_classes=10, **kwargs):
        super().__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the size after convolutions: 28 -> 14 -> 7 -> 3 (with 3 pools)
        # Actually: 28 -> 14 -> 7, then 7x7 with 128 channels
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # 28x28 -> 14x14

        # Second conv block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # 14x14 -> 7x7

        # Third conv block
        x = F.relu(self.conv5(x))
        # No pooling here to keep spatial dimensions

        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ModelCreator:
    def __init__(self, model_type:str="deepMLP", model_args:dict={}):
      self.model_type = model_type
      self.model_args = model_args
      self.initial_weights = dict(self.get_model().state_dict())

    def get_model(self) -> nn.Module:
      if self.model_type == "wideMLP":
          return WideMLP(**self.model_args)
      elif self.model_type == "deepMLP":
          return DeepMLP(**self.model_args)
      elif self.model_type == "simpleCNN":
          return SimpleCNN(**self.model_args)
      elif self.model_type == "deepCNN":
          return DeepCNN(**self.model_args)
      else:
          raise ValueError(f"Unknown model type: {self.model_type}")

    def create(self) -> nn.Module:
      model = self.get_model()
      model.load_state_dict(dict(self.initial_weights))
      return model


# model_args = {"depth":3}
# model_creator = ModelCreator("deepMLP", model_args=model_args)
# model_creator.create()