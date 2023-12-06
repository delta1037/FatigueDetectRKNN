import torch

from refer.session.session_com import InputSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchSession:
    def __init__(self, model_name):
        self.model_name = model_name
        self.backbone_mouth = torch.load(model_name)
        self.backbone_mouth.to(device)
        self.backbone_mouth.eval()

    @staticmethod
    def get_inputs():
        return [InputSet("input_name")]

    def run(self, arg_1, img):
        img = img["input_name"]
        outputs = self.backbone_mouth.forward(img.to(device)).to(device).detach().numpy()
        return outputs
