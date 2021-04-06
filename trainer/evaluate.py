import torch
from torchvision.transforms import transforms
from tqdm import tqdm
import os
from trainer.dataset import IMbDataset
from trainer.models import CRNN

model = CRNN()
print(os.listdir("models/"))
model.load_state_dict(torch.load("models/model_epoch_32.pth", map_location=torch.device('cpu')))
model.eval()
batch_size = 1
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = IMbDataset("data.csv", "data/", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

for i, data in enumerate(tqdm(trainloader)):
    #if i < 3:
    #    continue
    inputs, labels = data
    outputs = model(inputs)
    print(labels)
    #print(outputs)
    arg_maxes = torch.argmax(outputs, dim=2)
    print(arg_maxes.shape)
    print(arg_maxes)
    break

# decodes = []
# targets = []
#     for i, args in enumerate(arg_maxes):
#         decode = []
#         targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
#         for j, index in enumerate(args):
#             if index != blank_label:
#                 if collapse_repeated and j != 0 and index == args[j -1]:
#                     continue
#                 decode.append(index.item())
#         decodes.append(text_transform.int_to_text(decode))
#     return decodes, targets