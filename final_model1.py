import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from PIL import Image

train_image_dir = "drums_train"
test_image_dir = "drums_test"

class_size = 4
batch_size = 32
MAX_ITER = 100
LR = 0.00002
gamma = 0.975

#color image
img_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

#grayscale image
#img_transform = v2.Compose([v2.ToImage(),v2.ConvertImageDtype(torch.float32),v2.Grayscale(num_output_channels=1)])

train_dataset = torchvision.datasets.ImageFolder(
    train_image_dir, transform=img_transform
)

labels = train_dataset.classes
label_count = len(labels)
print(labels)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

if False:
    data_iterator = iter(train_loader)
    X, Y = next(data_iterator)

    image_grid = torchvision.utils.make_grid(X, padding=5, nrow=8)
    plt.imshow(np.transpose(image_grid.numpy(), (1, 2, 0)))
    plt.axis("off")
    plt.show()
    exit()


channels = 16
ksize = 3
K3 = (channels * 2) * 56 * 56  # Flattened size after second pooling layer

model1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=ksize),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=ksize),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(K3, 128),
    nn.ReLU(),
    nn.Linear(128, class_size),
)


model = model1
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
loss_func = nn.MSELoss()
loss_values = []
for epoch in range(MAX_ITER):
    for batch in train_loader:
        x_data, y_data = batch
        y_data = nn.functional.one_hot(y_data, num_classes=label_count).type(
            torch.float32
        )
        y_hat = model(x_data)
        loss = loss_func(y_hat, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch, loss.item())
    loss_values.append(loss.detach().numpy())
    scheduler.step()

test_dataset = torchvision.datasets.ImageFolder(test_image_dir, transform=img_transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

count = 0
wrong = 0

data_iterator = iter(test_loader)
for X, Y in data_iterator:
    y_hat = model(X)
    y_hat = torch.argmax(y_hat, axis=1)
    Z = (Y - y_hat).detach().numpy()
    print(Z)
    count += len(Z)
    wrong += np.count_nonzero(Z)


print("Count:", count)
print("Wrong:", wrong)
print("Percent Error:", 100 * wrong / count,'%')

epochs = list(range(0, MAX_ITER))
plt.plot(epochs, loss_values, color='blue', marker='o', label='Training Loss')
plt.title('CNN Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()
# Save the trained model
model_save_path = "/Users/tylerdurand/Downloads/School Stuff/math3375/final_model1_batch32.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")