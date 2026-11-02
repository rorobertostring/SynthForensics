__DEFAULT_TRANSFORMATION_CFG = {
    # 'resize': None,
    'post': {
        'blur': {
            'prob': 0.1,
            'sig': [0.0, 3.0]
        },
        'jpeg': {
            'prob': 0.1,
            'method': ['cv2', 'pil'],
            'qual': [30, 100]
        },
        'noise':{
            'prob': 0.0,
            'var': [0.01, 0.02]
        }
    },
    'crop': {
        'img_size': 224,
        'type': 'random'    # ['center', 'random'], according to 'train', 'val'or 'test' mode
    },
    'flip': True,    # set false when testing
    'normalize': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


__DEFAULT_TEST_TRANSFORMATION_CFG = {
    # 'resize': None,
    'post': {
        'blur': {
            'prob': 0.0,
            'sig': [0.0, 3.0]
        },
        'jpeg': {
            'prob': 0.0,
            'method': ['cv2', 'pil'],
            'qual': [30, 100]
        },
        'noise':{
            'prob': 0.0,
            'var': [0.01, 0.02]
        }
    },
    'crop': {
        'img_size': 224,
        'type': 'center'    # ['center', 'random'], according to 'train', 'val'or 'test' mode
    },
    'flip': False,    # set false when testing
    'normalize': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


def get_default_transformation_cfg(mode='train'):
    if mode == 'train':
        return __DEFAULT_TRANSFORMATION_CFG.copy()
    elif mode == 'test':
        return __DEFAULT_TEST_TRANSFORMATION_CFG.copy()
    else:
        raise ValueError(f'Expect mode in ["train", "test"]. Get "{mode}".')


def get_default_transformation(mode='train'):
    cfg = get_default_transformation_cfg()
    if mode == 'train':
        cfg['crop']['type'] = 'random'
    elif mode == 'test' or mode == 'val':
        cfg['crop']['type'] = 'center'
        cfg['flip'] = False
    return get_image_transformation_from_cfg(cfg)

