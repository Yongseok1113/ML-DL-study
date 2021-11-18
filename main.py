import argparse
import os
import csv
import copy
import tqdm

import torch
from torch import distributed
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from config.module import EMA, CrossEntropyLoss
from config.optimizer import RMSprop
from config.scheduler import StepLR
from config.model.EfficientNet import EfficientNet
from config.utils import util


dataset_path = os.path.join('your', 'data', 'path')


def batch(images, target, model, criterion=None):
    images = images.cuda()
    target = target.cuda()
    if criterion:
        with torch.cuda.amp.autocast():
            loss = criterion(model(images), target)
        return loss
    else:
        return util.accuracy(model(images), target, top_k=(1, 5))


def train(args):
    epochs = 350
    batch_size = 256
    util.set_seeds(args.rank)
    model = EfficientNet(args).cuda()
    lr = batch_size * torch.cuda.device_count() * 0.256 / 4096
    optimizer = RMSprop(util.add_weight_decay(model), lr, 0.9, 1e-3, momentum=0.9)
    ema = EMA(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)

    criterion = CrossEntropyLoss().cuda()
    scheduler = StepLR(optimizer)
    amp_scale = torch.cuda.amp.GradScaler()

    if args.tf:
        last_name = 'last_tf'
        best_name = 'best_tf'
        step_name = 'step_tf'
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        last_name = 'last_pt'
        best_name = 'best_pt'
        step_name = 'step_pt'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(root=os.path.join(dataset_path, 'train'),
                                   transform=transforms.Compose([util.RandomResize(),
                                                                 transforms.ColorJitter(0.4, 0.4, 0.4),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 util.RandomAugment(),
                                                                 transforms.ToTensor(),
                                                                 normalize]))

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    loader = data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    with open(f'weights/{step_name}.csv', 'w') as f:
        if args.local_rank == 0:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'acc@1', 'acc@5'])
            writer.writeheader()
        best_acc1 = 0

        for epoch in range(0, epochs):
            if args.distributed:
                sampler.set_epoch(epoch)
            if args.local_rank == 0:
                print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
                bar = tqdm.tqdm(loader, total=len(loader))
            else:
                bar = loader

            model.train()

            for images, targen in bar:
                loss = batch(images, target, model, criterion)
                optimizer.zero_grad()
                amp_scale.scale(loss).backward()
                amp_scale.step(optimizer)
                amp_scale.update()

                ema.update(model)
                torch.cuda.synchronize()
                if args.local_rank == 0:
                    bar.set_description(('%10s' + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), loss))

            scheduler.step(epoch + 1)
            if args.local_rank == 0:
                acc1, acc5 = test(args, ema.model.eval())
                writer.writerow({'acc@1': str(f'{acc1:.3f}'),
                                 'acc%5': str(f'{acc5:.3f}'),
                                 'epoch': str(epoch + 1).zfill(3)})
                state = {'model': copy.deepcopy(ema.model).half()}
                torch.save(state, f'weights/{last_name}.pt')
                if acc1 > best_acc1:
                    torch.save(state, f'weights/{best_name}.pt')
                del state
                best_acc1 = max(acc1, best_acc1)

    if args.distributed:
        torch.distributed.destroy_process_group()

    torch.cuda.empty_cache()


def test(args, model=None):
    if model is None:
        if args.tf:
            model = torch.load('weights/best_tf.pt', map_location='cuda')['model'].float().eval()
        else:
            model = torch.load('weights/best_pt.pt', map_location='cuda')['model'].float().eval()

    if args.tf:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'),
                                   transforms.Compose([transforms.Resize(416),
                                                       transforms.CenterCrop(384),
                                                       transforms.ToTensor(),
                                                       normalize]))

    loader = data.DataLoader(dataset, 48, num_workers=os.cpu_count(), pin_memory=True)
    top1 = util.AverageMeter()
    top5 = util.AverageMeter()
    with torch.no_grad():
        for images, target in tqdm.tqdm(loader, ('%10s' * 2) % ('acc@1', 'acc@5')):
            acc1, acc5 = batch(images, target, model)
            torch.cuda.synchronize()
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
        acc1, acc5 = top1.avg, top5.avg
        print('%10.3g' * 2 % (acc1, acc5))
    if model is None:
        torch.cuda.empty_cache()
    else:
        return acc1, acc5


def print_parameters(args):
    model = EfficientNet(args).eval()
    _ = model(torch.zeros(1, 3, 224, 224))
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {int(params)}')


def benchmark(args):
    shape = (1, 3, 384, 384)
    util.torch2onnx(EfficientNet(args).export().eval(), shape)
    util.onnx2caffe()
    util.print_benchmark(shape)


def main():
    # python -m torch.distributed.launch --nproc_per_node=3 main.py --train
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tf", action="store_true")

    args = parser.parse_args()
    args.distributed = False
    args.rank = 0
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.rank = torch.distributed.get_rank()
    if args.local_rank == 0:
        print_parameters(args)
    if args.benchmark:
        benchmark(args)
    if args.train:
        train(args)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
