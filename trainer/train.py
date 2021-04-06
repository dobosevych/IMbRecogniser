import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from trainer.dataset import IMbDataset
from trainer.models import CRNN

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 5
trainset = IMbDataset("data.csv", "../data/", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

model = CRNN()
criterion = nn.CTCLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        input_lengths = torch.full(size=(batch_size,), fill_value=124, dtype=torch.long)
        target_lengths = torch.full(size=(batch_size,), fill_value=65, dtype=torch.long)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

print('Finished Training')

