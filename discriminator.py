import torch
import torch.nn as nn

class LSTMDiscriminator(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(LSTMDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.rnn(x)
        # Return logits from the last time step
        return self.fc(out[:, -1, :])

class CNN1DDiscriminator(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, kernel_size=3):
        """
        Processes trajectory data using 1D convolutions.
        Input size corresponds to the number of features (e.g., 3 for x, y, z).
        """
        super(CNN1DDiscriminator, self).__init__()
        
        # We slide over the 'sequence_length' dimension.
        # The kernel 'sees' all 3 channels at once.
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1) # Collapses the sequence length to 1 for classification
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x starts as (Batch, Seq_Len, Features) based on your trajectory collection
        # We permute to (Batch, Features, Seq_Len) for the Conv1d layers
        x = x.permute(0, 2, 1) 
        
        out = self.conv_block(x)
        out = out.view(out.size(0), -1) # Flatten the hidden_size dimension
        return self.fc(out)

class MLPDiscriminator(nn.Module):
    def __init__(self, input_size=3, seq_length=100, hidden_size=128):
        super(MLPDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * seq_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

def run_experiment():
    # 1. Hyperparameters
    input_size = 3      # Assuming [x, y, z] trajectories
    seq_length = 50     # Number of timesteps per trajectory
    hidden_size = 64
    num_layers = 2
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Synthetic Data Generation (Replace this with your MuJoCo trajectories)
    # Format: (Samples, TimeSteps, Features)
    X_train = torch.randn(1000, seq_length, input_size)
    y_train = torch.randint(0, 2, (1000, 1)).float()
    
    X_test = torch.randn(200, seq_length, input_size)
    y_test = torch.randint(0, 2, (200, 1)).float()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True
    )

    # 3. Model, Loss, Optimizer
    model = LSTMDiscriminator(input_size, hidden_size, num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    print(f"Starting training on {device}...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/len(train_loader):.4f}')

    # 5. Testing
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(device))
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test.to(device)).sum().item() / y_test.size(0)
        print(f'\nFinal Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    run_experiment()
