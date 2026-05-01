import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMDiscriminator(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        """
        RNN-based discriminator for binary classification of sequences.
        Args:
            input_size: Number of features per timestep (e.g., 3 for x,y,z).
            hidden_size: Number of features in the RNN hidden state.
            num_layers: Number of recurrent layers.
        """
        super(LSTMDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Using LSTM for better gradient flow over long trajectories
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Final classification layers
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out

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
