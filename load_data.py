import torch
from torchvision import datasets, transforms

def load_CelebAHQ256(batch_size=4, path='data/CELEBAHQ'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x / 255.0 - 1)
    ])
    celebahq256_dataset = datasets.ImageFolder(root=path, transform=transform)

    trainloader = torch.utils.data.DataLoader(celebahq256_dataset, 
                                              batch_size=batch_size,
                                              num_workers=4)
    return trainloader                                          