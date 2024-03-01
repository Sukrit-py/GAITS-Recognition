import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from densepose import add_densepose_config, add_hrnet_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
)

def setup_cfg():
    cfg = get_cfg()
    add_densepose_config(cfg)
    add_hrnet_config(cfg)
    cfg.merge_from_file("configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    cfg.freeze()
    return cfg

def main():
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    visualizer = CompoundVisualizer(
        [ScoredBoundingBoxVisualizer(), DensePoseResultsFineSegmentationVisualizer(), DensePoseResultsContourVisualizer()]
    )

    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or specify video file path
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        outputs = predictor(frame)
        visualizer.visualize(frame, outputs)
        
        cv2.imshow("DensePose Human Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
