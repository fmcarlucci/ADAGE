import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
from dataset.data_loader import GetLoader, get_dataset, dataset_list
from torchvision import datasets


def test(dataset_name, epoch, model, image_size, batch_size=1024):
    assert dataset_name in dataset_list
    model.eval()
    cuda = True
    cudnn.benchmark = True
    lambda_val = 0

    dataloader = torch.utils.data.DataLoader(
        dataset=get_dataset(dataset_name, image_size, mode="test"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    n_total = 0.0
    n_correct = 0.0

    model.train(False)
    for i, (t_img, t_label) in enumerate(dataloader):
        batch_size = len(t_label)
        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        class_output, _ = model(input_data=Variable(t_img, volatile=True), lambda_val=lambda_val)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.view_as(pred)).cpu().sum()
        n_total += batch_size

    accu = n_correct / n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
    return accu
