
import argparse

#-----------------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    # exp file
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.2, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=240, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser