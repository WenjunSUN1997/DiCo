import torch

class Translator(torch.nn.Module):
    def __init__(self, config, backbone_model):
        super(Translator, self).__init__()
        self.backbone_model = backbone_model

    def forward(self):
        pass