# https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html

import torch
import torch.nn as nn
import time
import os
from PIL import Image
from typing import Union, List, Dict, Any, cast
from torch.hub import load_state_dict_from_url

model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
}

cfgs: Dict[str, List[Union[str, int]]] = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}

class VGG(nn.Module):
    layer_num = 0
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def run(model):
    start = time.time()
    model.eval()
    if(isGPU != 0):
        model.to('cuda')
    # print (model)
    input_dog = Image.open("data/dog.jpg")
    input_cat = Image.open("data/cat.jpg")
    input_penguin = Image.open("data/penguin.jpg")
    input_bunny = Image.open("data/bunny.jpeg")
    preprocess = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_penguin_tensor = preprocess(input_penguin)
    input_dog_tensor = preprocess(input_dog)
    input_cat_tensor = preprocess(input_cat)
    input_bunny_tensor = preprocess(input_bunny)
    input_batch = input_dog_tensor.unsqueeze(0)
    for i in range (batch_size - 1):
        if i%4 == 0:
            input_batch =  torch.cat((input_batch, input_cat_tensor.unsqueeze(0)), 0)
        elif i%4 == 1:
            input_batch =  torch.cat((input_batch, input_penguin_tensor.unsqueeze(0)), 0)
        elif i%4 == 2:
            input_batch =  torch.cat((input_batch, input_bunny_tensor.unsqueeze(0)), 0)
        elif i%4 == 3:
            input_batch =  torch.cat((input_batch, input_dog_tensor.unsqueeze(0)), 0)
    
    end = time.time()

    if print_runtime_only == 0:
        print("Init Time taken: %3.6f"%(end-start))

    if(isGPU != 0):
        # move the input and model to GPU for speed if available
        start = time.time()
        input_batch = input_batch.to('cuda')
        end = time.time()
        if print_runtime_only == 0:
            print("GPU Move Time taken: %3.6f"%(end-start))

    output = model(input_batch)
    start = time.time()
    for i in range(1):
        with torch.no_grad():
            output = model(input_batch)
    end = time.time()
    if(isGPU != 0):
        torch.cuda.synchronize()
        output = output.cpu()
    if(isMKLDNN != 0):
        output = output.to_dense()

    if print_runtime_only == 0:
        print("Run Time taken: %3.6f"%((end-start)))
    else:
        print("%3.6f"%((end-start)))
    # for i in range(1000):
    #     print("%3.3e"%(output[0][i].item()))

    # probabilities = torch.nn.functional.softmax(output, dim=0)
    # dump_tensor_raw ("resnet50_layer73.bin", probabilities)
    
    if print_runtime_only == 0:
        for b in range(batch_size):
            print ("Batch " + str(b) + ":")
            probabilities = torch.nn.functional.softmax(output[b], dim=0)
            with open("data/imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            for i in range(top5_prob.size(0)):
                print("%d:"%(i+1), categories[top5_catid[i]], "- %4.2f%%"%(100*top5_prob[i].item()))
        end = time.time()
        print("Run Time taken (Output included): %3.6f"%((end-start)))

def dump_tensor (path, tensor, tensor_info_string):
    with open(path, "a") as f:
        f.write(tensor_info_string)
    np_arr = tensor.detach().numpy()
    np_arr.astype('float32').tofile(path + ".tmp")
    data_size = os.path.getsize(path + ".tmp")
    with open(path, "a") as f:
        f.write("DATA_SIZE:" + str(data_size) + "\n")
        f.write("DATA_START:\n")
    os.system ("cat " + path + ".tmp >> " + path)
    with open(path, "a") as f:
        f.write("DATA_END\n")
    os.system ("rm " + path + ".tmp")
    with open(path, "a") as f:
        f.write("LAYER_END\n")

def dump_data():
    model = vgg16(True, False)
    model.eval()
    path = "data/vgg16_data.bin"
    print (model)
    os.system ("echo ASPEN_DATA > " + path)
    dump_tensor (path, model.features[0].weight, "LAYER:1\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[0].bias, "LAYER:1\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[2].weight, "LAYER:2\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[2].bias, "LAYER:2\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[5].weight, "LAYER:3\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[5].bias, "LAYER:3\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[7].weight, "LAYER:4\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[7].bias, "LAYER:4\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[10].weight, "LAYER:5\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[10].bias, "LAYER:5\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[12].weight, "LAYER:6\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[12].bias, "LAYER:6\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[14].weight, "LAYER:7\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[14].bias, "LAYER:7\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[17].weight, "LAYER:8\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[17].bias, "LAYER:8\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[19].weight, "LAYER:9\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[19].bias, "LAYER:9\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[21].weight, "LAYER:10\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[21].bias, "LAYER:10\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[24].weight, "LAYER:11\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[24].bias, "LAYER:11\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[26].weight, "LAYER:12\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[26].bias, "LAYER:12\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.features[28].weight, "LAYER:13\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.features[28].bias, "LAYER:13\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.classifier[0].weight, "LAYER:14\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.classifier[0].bias, "LAYER:14\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.classifier[3].weight, "LAYER:15\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.classifier[3].bias, "LAYER:15\nTENSOR_TYPE:BIAS\n")
    dump_tensor (path, model.classifier[6].weight, "LAYER:16\nTENSOR_TYPE:WEIGHT\n")
    dump_tensor (path, model.classifier[6].bias, "LAYER:16\nTENSOR_TYPE:BIAS\n")
