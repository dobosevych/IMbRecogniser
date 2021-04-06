import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import os
from trainer.dataset import IMbDataset
from trainer.models import BarcodeRecognizer
import random

model = BarcodeRecognizer(chars=5)
model.load_state_dict(torch.load("models/model_epoch_50.pth", map_location=torch.device('cpu')))
model.eval()
batch_size = 1
transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = IMbDataset("data.csv", "samples/", transform=transform)

data = trainset[random.randint(0, len(trainset) - 1)]
inputs, labels = data
inputs = inputs.unsqueeze(0)
outputs = model(inputs)
arg_maxes = torch.argmax(outputs, dim=2)
arg_maxes = arg_maxes.squeeze().tolist()
decode = []
for j, index in enumerate(arg_maxes):
    if index != 0:
        if j != 0 and index == arg_maxes[j - 1]:
            continue
        decode.append(str(index))
mapping = {"1": "A", "2": "D", "3": "F", "4": "T"}
predicted = "".join(map(lambda x: mapping[x], map(str, decode)))
correct = "".join(map(lambda x: mapping[x], decode))

print("predicted ", predicted)
print("correct   ", correct)
print("difference", "".join(["0" if c == correct[i] else "1" for i, c in enumerate(predicted)]))