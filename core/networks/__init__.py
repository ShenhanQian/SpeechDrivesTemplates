from core.networks.keypoints_generation.generator import SequenceGeneratorCNN
from core.networks.keypoints_generation.discriminator import PoseSequenceDiscriminator
from core.networks.poses_reconstruction.autoencoder import Autoencoder, PoseSeqEncoder


module_dict = {
    'SequenceGeneratorCNN': SequenceGeneratorCNN,
    'PoseSequenceDiscriminator': PoseSequenceDiscriminator,
    'Autoencoder': Autoencoder,
    'PoseSeqEncoder': PoseSeqEncoder,
}


def get_model(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown model: %s' % name)
    else:
        return obj
