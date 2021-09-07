import torchvision.models as models

def get_vgg13(**kwargs):
    return models.vgg13_bn(False)

# def get_densenet():
#     return models.dense