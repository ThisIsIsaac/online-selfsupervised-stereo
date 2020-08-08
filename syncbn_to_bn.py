
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from models import hsm
from utils import wandb_logger
from sync_batchnorm.sync_batchnorm import convert_model

if __name__ == "__main__":
    model = hsm(384, clean=False, level=1)

    model = nn.DataParallel(model)
    model = convert_model(model)

    model.cuda()

    # load model
    pretrained_dict = torch.load("weights/final-768px.tar")
    pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if ('disp' not in k)}
    model.load_state_dict(pretrained_dict['state_dict'], strict=False)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
