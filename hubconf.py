dependencies = ['torch', 'torchvision', 'gdown']

import torch
import torchvision.transforms as transforms
from typing import Literal
from adaface import IR_18, IR_50, IR_101, AdaFace
import gdown

def _reverse_channels(img):
    return img[[2, 1, 0], :, :]

TRANFRORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ConvertImageDtype(torch.float32),
    _reverse_channels,
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

_URLS = {
    'iresnet_18' : {
        'casia_webface' : 'https://drive.google.com/uc?id=1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe&export=download', 
        'vgg_face2' : 'https://drive.google.com/uc?id=1k7onoJusC0xjqfjB-hNNaxz9u6eEzFdv&export=download', 
        'web_face_4m' : 'https://drive.google.com/uc?id=1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y&export=download',
    },
    'iresnet_50' : {
        'casia_webface' : 'https://drive.google.com/uc?id=1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2&export=download', 
        'web_face_4m' : 'https://drive.google.com/uc?id=1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN&export=download',
        'ms1mv2' : 'https://drive.google.com/uc?id=1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI&export=download',
    },
    'iresnet_100' : {
        'ms1mv2_100' : 'https://drive.google.com/uc?id=1m757p4-tUU5xlSHLaO04sqnhvqankimN&export=download',
        'ms1mv3_100' : 'https://drive.google.com/uc?id=1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI&export=download',
        'web_face_4m_100' : 'https://drive.google.com/uc?id=18jQkqB0avFqWa0Pas52g54xNshUOQJpQ&export=download',
        'web_face_12m_100' : 'https://drive.google.com/uc?id=1dswnavflETcnAuplZj1IOKKP0eM8ITgT&export=download',
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
        raise ValueError(f'Pretrained model {pretrained} not available for {model}, available options are {list(_URLS[model].keys())}')
    
    checkpoint_url = _URLS[model][pretrained]
    checkpoint_path = gdown.cached_download(checkpoint_url)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = _filter_state_dict_by_prefix(checkpoint['state_dict'], 'model.')
    
    model = model_(**kwargs)
    model.load_state_dict(state_dict)
    
    head_state_dict = _filter_state_dict_by_prefix(checkpoint['state_dict'], 'head.')
    head = AdaFace(embedding_size=head_state_dict['kernel'].shape[0], classnum=head_state_dict['kernel'].shape[1])
    head.load_state_dict(head_state_dict)
    return model, head

def adaface_iresnet_18(pretrained: Literal[False, 'casia_webface', 'vgg_face2', 'web_face_4m'] = 'web_face_4m'):
    return _load_model('iresnet_18', pretrained, input_size=[112, 112])[0]

def adaface_iresnet_50(pretrained: Literal[False, 'casia_webface', 'web_face_4m', 'ms1mv2'] = 'web_face_4m'):
    return _load_model('iresnet_50', pretrained, input_size=[112, 112])[0]

def adaface_iresnet_100(pretrained: Literal[False, 'ms1mv2_100', 'ms1mv3_100', 'web_face_4m_100', 'web_face_12m_100'] = 'web_face_12m_100'):
    return _load_model('iresnet_100', pretrained, input_size=[112, 112])[0]

def adaface_iresnet_18_head(pretrained: Literal[False, 'casia_webface', 'vgg_face2', 'web_face_4m'] = 'web_face_4m'):
    return _load_model('iresnet_18', pretrained, input_size=[112, 112])[1]

def adaface_iresnet_50_head(pretrained: Literal[False, 'casia_webface', 'web_face_4m', 'ms1mv2'] = 'web_face_4m'):
    return _load_model('iresnet_50', pretrained, input_size=[112, 112])[1]

def adaface_iresnet_100_head(pretrained: Literal[False, 'ms1mv2_100', 'ms1mv3_100', 'web_face_4m_100', 'web_face_12m_100'] = 'web_face_12m_100'):
    return _load_model('iresnet_100', pretrained, input_size=[112, 112])[1]

def default_transform():
    return TRANFRORM
