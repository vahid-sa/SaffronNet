import sys
import torch
from os import path as osp
sys.path.append(osp.abspath("../"))
sys.path.append(osp.abspath("./"))
from retinanet import my_model as model

device = "cuda:0"
state_dict_path = osp.expanduser('~/Saffron/init_fully_trained_weights/init_state_dict.pt')

retinanet = model.vgg7(num_classes=1, pretrained=True)
retinanet = retinanet.to(device=device)
retinanet = torch.nn.DataParallel(retinanet)
retinanet = retinanet.to(device=device)
optimizer = torch.optim.Adam(retinanet.parameters(), lr=1e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[6,8,9],
    gamma=0.1,
)

checkpoint = torch.load(state_dict_path)
retinanet.load_state_dict(checkpoint['model_state_dict'])
print("Molde loaded.")

for i in range(10):
    optimizer.step()
    print("optimizer", optimizer.param_groups[0]['lr'])
    scheduler.step()
