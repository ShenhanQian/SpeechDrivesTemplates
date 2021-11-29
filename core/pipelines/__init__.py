from core.pipelines.voice2pose import Voice2Pose
from core.pipelines.pose2pose import Pose2Pose


module_dict = {
    'Voice2Pose': Voice2Pose,
    'Pose2Pose': Pose2Pose,
}


def get_pipeline(name: str):
    obj = module_dict.get(name)
    if obj is None:
        raise KeyError('Unknown pipeline: %s' % name)
    else:
        return obj
