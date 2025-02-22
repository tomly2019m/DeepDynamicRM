import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the processed dataset
processed_data = pd.read_csv('processed_data.csv')

# Split features and labels
X = processed_data.drop(columns=['label']).values
y = processed_data['label'].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to NumPy arrays before train_test_split
X = X.astype(np.float32)
y = y.astype(np.int64)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert back to PyTorch tensors and move to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the self-attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        queries = self.query(x)  # (batch_size, seq_len, output_dim)
        keys = self.key(x)  # (batch_size, seq_len, output_dim)
        values = self.value(x)  # (batch_size, seq_len, output_dim)

        # Calculate attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (
                keys.shape[-1] ** 0.5)  # (batch_size, seq_len, seq_len)
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_len, seq_len)

        # Apply attention weights
        output = torch.matmul(attention_weights, values)  # (batch_size, seq_len, output_dim)
        return output


# Define the neural network model with self-attention
class AttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim):
        super(AttentionMLP, self).__init__()
        self.input_dim = input_dim
        self.attention = SelfAttention(input_dim, attention_dim)
        self.fc1 = nn.Linear(attention_dim, hidden_dim)  # Correct the input size
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 2)  # Output size = 2 (binary classification)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Reshape input for attention: (batch_size, seq_len, input_dim)
        batch_size, feature_dim = x.size()
        seq_len = 1  # Since features are flattened into a sequence
        x = x.view(batch_size, seq_len, feature_dim)

        # Apply self-attention
        attention_out = self.attention(x)  # (batch_size, seq_len, attention_dim)

        # Flatten attention output
        flattened = attention_out.view(batch_size, -1)

        # Pass through fully connected layers
        x = self.relu(self.fc1(flattened))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
hidden_dim = 128
attention_dim = 64
model = AttentionMLP(input_dim, hidden_dim, attention_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training the model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            X_batch, y_batch = batch
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")


# Evaluating the model
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_batch = batch
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


if __name__ == "__main__":
    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs=200)

    # 在训练完成后保存模型权重
    torch.save(model.state_dict(), "attention_mlp_weights.pth")
    print("Model weights saved to attention_mlp_weights.pth")

    # 导出模型为 ONNX 格式
    dummy_input = torch.randn(1, X_train.shape[1], device=device)  # 创建一个示例输入张量
    print(dummy_input.shape)
    onnx_filename = "attention_mlp.onnx"

    torch.onnx.export(
        model,  # 模型
        dummy_input,  # 示例输入
        onnx_filename,  # 保存的 ONNX 文件名
        input_names=["input"],  # 输入节点名称
        output_names=["output"],  # 输出节点名称
        dynamic_axes={  # 动态维度支持
            "input": {0: "batch_size"},  # 输入的 batch_size 为动态
            "output": {0: "batch_size"}  # 输出的 batch_size 为动态
        },
        opset_version=11  # ONNX opset 版本
    )
    print(f"Model exported to ONNX format as {onnx_filename}")

    # Evaluate the model
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y_true, y_pred = evaluate_model(model, test_loader)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Plot confusion matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
