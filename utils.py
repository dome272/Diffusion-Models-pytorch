import torchvision
from torch.utils.data import DataLoader


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataloader

