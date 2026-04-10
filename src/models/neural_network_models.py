import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from .base_model import BaseModel

class NNModule(nn.Module):
    """
    Standard PyTorch Module structure from Lab Inspiration.
    Built dynamically based on the layers list.
    """
    def __init__(self, input_size, hidden_layers, activation_fn=nn.ReLU):
        super(NNModule, self).__init__()
        layers = []
        prev_size = input_size
        
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(activation_fn())
            prev_size = h
            
        # Final Classification Layer (Output=1 for Binary Stutter/Fluent)
        layers.append(nn.Linear(prev_size, 1))
        # No Sigmoid here because we'll use BCEWithLogitsLoss during training
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class NeuralNetworkModel(BaseModel):
    """
    Standardized Wrapper for PyTorch Neural Networks.
    Provides a consistent interface for the Stuttering Detection Pipeline.
    """
    def __init__(self, model_name="NN_Default", hidden_layers=[64], 
                 lr=0.01, activation_fn=nn.ReLU, optimizer_class=optim.SGD, **kwargs):
        super().__init__(model_name)
        # Separate training settings from optimizer hyperparameters
        self.epochs = kwargs.pop('epochs', 200) # Default to 200
        
        # input_size is now detected or passed
        self.input_size = kwargs.pop('input_size', 768) 
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.activation_fn = activation_fn
        self.optimizer_class = optimizer_class
        
        # Initialize the actual PyTorch model
        self.model = NNModule(self.input_size, self.hidden_layers, self.activation_fn)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Now kwargs only contains optimizer-specific arguments like momentum
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr, **kwargs)

    def train(self, X_train, y_train, epochs=None):
        num_epochs = epochs if epochs is not None else self.epochs
        print(f"[{self.model_name}] Training PyTorch Network (Layers: {self.hidden_layers}) for {num_epochs} epochs...")
        
        X = torch.FloatTensor(X_train)
        y = torch.FloatTensor(y_train).view(-1, 1)

        # Dynamic Input Size Re-Check
        if X.shape[1] != self.input_size:
            print(f"[{self.model_name}] Re-initializing network for new input dimension: {X.shape[1]}")
            self.input_size = X.shape[1]
            self.model = NNModule(self.input_size, self.hidden_layers, self.activation_fn)
            self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            logits = self.model(X_tensor)
            # Apply Sigmoid and threshold at 0.5
            probs = torch.sigmoid(logits)
            return (probs > 0.5).numpy().astype(int).flatten()

    def save(self, file_path):
        save_dict = {
            'model_state': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers
        }
        torch.save(save_dict, file_path)
        print(f"[{self.model_name}] Model weights saved to {file_path}")

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"[{self.model_name}] Model weights loaded from {file_path}")

# Convenience Subclasses
class ShallowNeuralNetwork(NeuralNetworkModel):
    def __init__(self, model_name="Shallow_NN", hidden_layer_size=100, **kwargs):
        super().__init__(model_name, hidden_layers=[hidden_layer_size], **kwargs)

class DeepNeuralNetwork(NeuralNetworkModel):
    def __init__(self, model_name="Deep_NN", hidden_layer_sizes=(100, 50), **kwargs):
        super().__init__(model_name, hidden_layers=hidden_layer_sizes, **kwargs)
