def freeze_all_layers(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return None

def freeze_except_last_layers(model, n=2):
    
    layernames = list([name for name, _ in model.named_parameters() if "layer." in name])
    layerids = [name.split("layer.")[1].split(".")[0] for name in layernames]
    layerids = set([int(lid) for lid in layerids])
    layerids = sorted(list(layerids))
    lastn = layerids[-n:]
    lastn = ["layer." + str(lid) for lid in lastn]

    for name, param in model.named_parameters():
        lid = None
        if "layer." in name:
            lid = "layer." + name.split("layer.")[1].split(".")[0]
        if lid and lid in lastn:
            continue
        else:
            param.requires_grad = False
    
    return None
