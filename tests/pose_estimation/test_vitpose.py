import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from humanproc_utils.pose_estimation_2d.vitpose.inference import vitpose_inference


results = vitpose_inference(
    vid_file="inputs/sample_video.mp4", model_name="weights/vitpose-b-multi-coco.pth"
)


for person_id in results:
    print(results[person_id]["frames"].shape)
    print(results[person_id]["joints2d"].shape)
    print(results[person_id]["confidence"].shape)
    print(" ==================================== ")
