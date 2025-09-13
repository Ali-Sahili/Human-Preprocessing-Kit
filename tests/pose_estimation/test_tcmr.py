import warnings

warnings.filterwarnings("ignore")

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from humanproc_utils.pose_estimation_3d.TCMR.inference import tcmr_inference
from humanproc_utils.pose_estimation_3d.vibe.lib.utils.utils import load_yaml_as_config


if __name__ == "__main__":
    cfg = load_yaml_as_config("configs/pose_estimation/tcmr.yaml")
    cfg_pose = cfg.pose_estimation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg_pose.gpu)

    tcmr_inference(cfg_pose)
