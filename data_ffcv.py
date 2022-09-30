import torch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

def create_cifar_ffcv():
    datasets = {
        'train': torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.train_folder)),
        'test': torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.val_folder))
    }


    for (name, ds) in datasets.items():
        writer = DatasetWriter(f'/datasets/cifar_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)


def get_dataloaders(config):
    loaders = {}
    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
        ])

        # Create loaders
        loaders[name] = Loader(f'/datasets/cifar_{name}.beton',
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                order=OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline,
                                           'label': label_pipeline})
    return loaders["train"], loaders["test"]