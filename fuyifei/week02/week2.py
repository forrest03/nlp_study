import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def generate_data(num_samples, num_features):
    X = np.random.rand(num_samples, num_features)
    y = np.argmax(X, axis=1)  # 标签是最大值所在的索引
    return X, y

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

def train_model(model, criterion, optimizer, X_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(y_train).long()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def main():
    num_samples = 1000
    num_features = 10
    num_classes = num_features
    num_epochs = 100
    learning_rate = 0.001

    X_train, y_train = generate_data(num_samples, num_features)

    model = SimpleNN(input_size=num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, criterion, optimizer, X_train, y_train, num_epochs)

if __name__ == "__main__":    main()
