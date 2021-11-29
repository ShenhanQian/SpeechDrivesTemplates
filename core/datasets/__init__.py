from core.datasets.speech2gesture import Speech2GestureDataset


module_dict = {
    'speech2gesture': Speech2GestureDataset,
}


def get_dataset(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown dataset: %s' % name)
    else:
        return obj
