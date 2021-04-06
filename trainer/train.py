import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from trainer.dataset import IMbDataset
from trainer.models import CRNN
import os

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 128
trainset = IMbDataset("data.csv", "data/", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

device = "cuda:0"
#device = "cpu"

model = CRNN().to(device)
criterion = nn.CTCLoss(reduction='mean').cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader)):
        inputs, labels = data
        input_lengths = torch.full(size=(inputs.shape[0],), fill_value=189, dtype=torch.int32)
        target_lengths = torch.full(size=(inputs.shape[0],), fill_value=65, dtype=torch.int32)
        labels = torch.flatten(labels)
        labels = labels.type(torch.long)
        optimizer.zero_grad()
        inputs, labels, input_lengths, target_lengths = inputs.to(device), labels.to(device), input_lengths.to(device), target_lengths.to(device)
        outputs = model(inputs)
        print(outputs)
        #print(inputs)
        #print(labels)
        #print(input_lengths)
        #print(target_lengths)

        loss = criterion(outputs, labels, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

    model_path = "models/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_path)

print('Finished Training')

