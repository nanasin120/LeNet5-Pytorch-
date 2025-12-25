import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import LeNet5
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.001
epochs = 10
writer = SummaryWriter('runs/lenet5_experiment_1')

def train():
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = LeNet5().to(device=device)

    dummy_input = torch.randn(1, 1, 32, 32).to(device)
    writer.add_graph(model, dummy_input)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        print(f"\nAverage Loss: {train_loss/len(train_loader):.4f}, Accuracy: {correct}/{len(test_dataset)} ({100. * correct / len(test_dataset):.2f}%)\n")
        
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Loss/test', test_loss/len(test_loader), epoch)
        writer.add_scalar('Accuracy/test', 100. * correct / len(test_dataset), epoch)

    torch.save(model.state_dict(), "lenet5_mnist.pth")
    writer.close()

    print("Model saved to lenet5_mnist.pth")

if __name__ == "__main__":
    train()
    
# tensorboard --logdir=runs