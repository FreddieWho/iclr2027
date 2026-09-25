"""A-wide backbone for N02 upgrade: channel-x4 ResNet18, random-init only.

Pretrained weights do not exist for wide nets, so the compute-separation
verdict (P1) is scoped to random-init (A / A-wide / B all random).
Channel x4 -> conv MACs ~16x (148M -> ~2.4G), at/above B (1.81G): conservative
for the compute hypothesis (if A-wide still << B, grid stands strongly).
"""
import torch
from torchvision.models.resnet import BasicBlock


def wide_resnet18(width=4, num_classes=1):
    w = width
    net = torch.nn.Sequential()
    conv1 = torch.nn.Conv2d(3, 64 * w, 7, stride=2, padding=3, bias=False)
    bn1 = torch.nn.BatchNorm2d(64 * w)
    layers = [
        ('conv1', conv1), ('bn1', bn1), ('relu', torch.nn.ReLU(inplace=True)),
    ]
    net = torch.nn.Module()
    net.conv1 = conv1
    net.bn1 = bn1
    net.relu = torch.nn.ReLU(inplace=True)
    net.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)

    def block_stack(inplanes, planes, blocks, stride=1):
        down = None
        if stride != 1 or inplanes != planes:
            down = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes))
        mods = [BasicBlock(inplanes, planes, stride, down)]
        for _ in range(1, blocks):
            mods.append(BasicBlock(planes, planes))
        return torch.nn.Sequential(*mods)

    net.layer1 = block_stack(64 * w, 64 * w, 2)
    net.layer2 = block_stack(64 * w, 128 * w, 2, stride=2)
    net.layer3 = block_stack(128 * w, 256 * w, 2, stride=2)
    net.layer4 = block_stack(256 * w, 512 * w, 2, stride=2)
    net.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    net.fc = torch.nn.Linear(512 * w, num_classes)

    def forward(x):
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        x = net.layer4(x)
        x = net.avgpool(x)
        return net.fc(x.flatten(1))
    net.forward = forward
    return net


def net_for_wide(init, stem, seed, width=4):
    assert init == 'random', init  # no pretrained weights exist for wide nets
    torch.manual_seed(seed)
    net = wide_resnet18(width=width)
    if stem == 'lowstride':
        net.maxpool = torch.nn.Identity()
    return net


def count_params(net):
    return sum(p.numel() for p in net.parameters())


def estimate_macs(net, res=64, device='cpu'):
    net = net.to(device).eval()
    mac, hooks = [], []

    def hook(m, i, o):
        mac.append(int(o.numel() * m.kernel_size[0] * m.kernel_size[1]
                       * m.in_channels / m.groups))
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    with torch.inference_mode():
        net(torch.zeros(1, 3, res, res, device=device))
    for h in hooks:
        h.remove()
    return sum(mac)


if __name__ == '__main__':
    import sys
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    for stem in ['standard', 'lowstride']:
        n = net_for_wide('random', stem, 803, width=w)
        print(stem, 'params', count_params(n),
              'macs64', estimate_macs(n, 64),
              'macs224', estimate_macs(n, 224))
