# (1) Import necessary libraries
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets
import optuna

# (2) Loading the MNIST dataset

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
full_trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)

# Split the training data into a training set and a validation set
# Let's use 80% for training and 20% for validation
train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# (3) Defining the artificial neural network (ANN) structure
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

epochs = 50 # set to train for 50 epochs

# Function to find optimal hyperparameters
def objective(trial):
    # Suggest values for the hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    hidden_size = trial.suggest_int('hidden_size', 10, 100)

    # Create the network
    model = Net(hidden_size=hidden_size)

    # Create data loaders, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training and validation loop
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss}")

        # validation phase
        model.eval()
        val_running_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for val_images, val_labels in valloader:
                val_output = model(val_images)
                val_loss = criterion(val_output, val_labels)
                val_running_loss += val_loss.item()
                _, preds = torch.max(val_output.data, 1)
                val_accuracy += (preds == val_labels).sum().item()

        val_loss = val_running_loss/len(valloader.dataset)
        val_accuracy = val_accuracy/len(valloader.dataset)

        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}')

        # model back to train mode
        model.train()
    
    trial.set_user_attr('trained_model', model)

    return val_loss  # this assumes that you want to minimize validation loss

# (5) Conduct the optimization
study = optuna.create_study(direction="minimize")  # minimize error, maximize accuracy
study.optimize(objective, n_trials=5)

best_trial = study.best_trial  # Get the best trials

# You can access the best parameters from the best trial
best_params = best_trial.params

# Create the network with best parameters
best_model = best_trial.user_attrs['trained_model']

# Create data loaders, loss function, and optimizer with best parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['lr'])

# Training loop with the best parameters
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = best_model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss}")

# (6) Test the model
best_model.eval()  # put the model in evaluation mode

correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = best_model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))


