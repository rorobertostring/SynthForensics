from models import MMDet


def get_model(config):
    if config['model_name'] == 'MMDet':
        return MMDet(config)
    else:
        raise ValueError(f'Unsupported model: {config["model_name"]}')