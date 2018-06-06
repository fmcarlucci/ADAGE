import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from dataset.data_loader import GetLoader, get_dataset, dataset_list
from torchvision import datasets

cache = {}


def get_dataloader(dataset_name, image_size, limit, batch_size, tune_stats=False):
    dataloader = cache.get(dataset_name, None)
    if tune_stats:
        mode = "test-tuned"
    else:
        mode = "test"
    if dataloader is None:
        dataloader = torch.utils.data.DataLoader(
            dataset=get_dataset(dataset_name, image_size, mode=mode, limit=limit),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        cache[dataset_name] = dataloader
    return dataloader


def test(dataset_name, epoch, model, image_size, domain, batch_size=1024, limit=None, tune_stats=False):
    assert dataset_name in dataset_list
    model.eval()
    cuda = True
    cudnn.benchmark = True
    lambda_val = 0

    n_total = 0.0
    n_correct = 0.0

    model.train(False)
    dataloader = get_dataloader(dataset_name, image_size, limit, batch_size, tune_stats=tune_stats)
    for i, (t_img, t_label) in enumerate(dataloader):
        batch_size = len(t_label)
        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
        with torch.no_grad():
            class_output, _, _ = model(input_data=t_img, lambda_val=lambda_val, domain=domain)
            pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.view_as(pred)).cpu().sum()
        n_total += batch_size

    accu = n_correct / n_total

    print('epoch: %d, accuracy of the %s dataset (%d batches): %f' % (epoch, dataset_name, len(dataloader), accu))
    return accu
