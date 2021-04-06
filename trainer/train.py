import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from trainer.dataset import IMbDataset
from trainer.models import CRNN, BarcodeRecognizer
import os

transform = transforms.Compose(
    [transforms.ToTensor()])
batch_size = 128
trainset = IMbDataset("data.csv", "data/", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

device = "cuda:0"

model = BarcodeRecognizer(chars=5).to(device)
criterion = nn.CTCLoss(reduction='mean').cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):
    running_loss = 0.0
    pbar = tqdm(trainloader)
    for i, data in enumerate(pbar):
        inputs, labels = data
        #print(labels)
        input_lengths = torch.full(size=(inputs.shape[0],), fill_value=146, dtype=torch.int32)
        target_lengths = torch.full(size=(inputs.shape[0],), fill_value=65, dtype=torch.int32)
        labels = torch.flatten(labels)
        labels = labels
        optimizer.zero_grad()

        inputs, labels, input_lengths, target_lengths = inputs.to(device), labels.to(device), input_lengths.to(device), target_lengths.to(device)
        outputs = model(inputs)
        #print(outputs.shape)
        #print(outputs.shape)
        #print(labels.shape)
        #print(input_lengths.shape)
        #print(target_lengths.shape)

        loss = criterion(outputs, labels, input_lengths, target_lengths)

        loss.backward()
        optimizer.step()
        pbar.set_description("Loss {}".format(loss.item()))
        running_loss += loss.item()

    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))

    model_path = "models/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_path)

print('Finished Training')

