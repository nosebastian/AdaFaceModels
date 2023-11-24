dependencies = ['torch', 'torchvision', 'gdown']

import torch
import torchvision.transforms as transforms
from typing import Literal
from adaface import IR_18, IR_50, IR_101
import gdown

TRANFRORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

_URLS = {
    'iresnet_18' : {
        'casia_webface' : 'https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view?usp=sharing', 
        'vgg_face2' : 'https://drive.google.com/file/d/1k7onoJusC0xjqfjB-hNNaxz9u6eEzFdv/view?usp=sharing', 
        'web_face_4m' : 'https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view?usp=sharing',
    },
    'iresnet_50' : {
        'casia_webface' : 'https://drive.google.com/file/d/1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2/view?usp=sharing', 
        'web_face_4m' : 'https://drive.google.com/file/d/1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN/view?usp=sharing',
        'ms1mv2' : 'https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing',
    },
    'iresnet_100' : {
        'ms1mv2_100' : 'https://drive.google.com/file/d/1m757p4-tUU5xlSHLaO04sqnhvqankimN/view?usp=sharing',
        'ms1mv3_100' : 'https://drive.google.com/file/d/1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing',
        'web_face_4m_100' : 'https://drive.google.com/file/d/18jQkqB0avFqWa0Pas52g54xNshUOQJpQ/view?usp=sharing',
        'web_face_12m_100' : 'https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view?usp=sharing',
    }
}

def _filter_state_dict_by_prefix(state_dict, prefix):
    return type(state_dict)([(k[len(prefix):], v) for k, v in state_dict.items() if k.startswith(prefix)])

def _load_model(model: Literal['iresnet_18', 'iresnet_50', 'iresnet_100'], pretrained: Literal[False, 'casia_webface', 'vgg_face2', 'web_face_4m'], **kwargs):
    global _URLS
    model_ = {
        'iresnet_18' : IR_18,
        'iresnet_50' : IR_50,
        'iresnet_100' : IR_101
    }[model]
    if pretrained == False:
        return model_(**kwargs)
    
    if not pretrained in _URLS[model]:
        raise ValueError(f'Pretrained model {pretrained} not available for {model}, available options are {list(URLS[model].keys())}')
    
    checkpoint_url = _URLS[model][pretrained]
    checkpoint_path = gdown.cached_download(checkpoint_url)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = _filter_state_dict_by_prefix(checkpoint['state_dict'], 'model.')
    
    model = model_(**kwargs)
    model.load_state_dict(state_dict)
    return model

def adaface_iresnet_18(pretrained: Literal[False, 'casia_webface', 'vgg_face2', 'web_face_4m'] = 'web_face_4m'):
    return _load_model('iresnet_18', pretrained, size=224)

def adaface_iresnet_50(pretrained: Literal[False, 'casia_webface', 'web_face_4m', 'ms1mv2'] = 'web_face_4m'):
    return _load_model('iresnet_50', pretrained, size=224)

def adaface_iresnet_100(pretrained: Literal[False, 'ms1mv2_100', 'ms1mv3_100', 'web_face_4m_100', 'web_face_12m_100'] = 'web_face_12m_100'):
    return _load_model('iresnet_100', pretrained, size=224)

