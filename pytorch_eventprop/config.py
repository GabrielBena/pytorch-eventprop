

def get_flat_dict_from_nested(config):
    flat_dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat_dict.update(get_flat_dict_from_nested(value))
        else:
            flat_dict[key] = value
    return flat_dict