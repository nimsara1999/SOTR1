import os
import argparse
import time

from tqdm import tqdm

import cv2
import torch
import numpy as np

from adet.config import get_cfg
from adet.utils.visualizer import TextVisualizer
from demo.predictor import VisualizationDemo
from detectron2.utils.visualizer import Visualizer


class Visualization(VisualizationDemo):
    def __init__(self, cfg, score_threshold=0.3,):
        super(Visualization, self).__init__(cfg)
        self.score_threshold = score_threshold

    def run_on_image(self, image):
        vis_output = None
        t0=time.time()
        predictions = self.predictor(image)
        t1=time.time()
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        if self.vis_text:
            visualizer = TextVisualizer(image, self.metadata, instance_mode=self.instance_mode)
        else:
            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            instances = instances[instances.scores >= self.score_threshold]
            #labels = [self.metadata.thing_classes[x] for x in instances.pred_classes]
            colors = [self.metadata.thing_colors[x] for x in instances.pred_classes]
            colors = [[x/255 for x in lst] for lst in colors]
            vis_output = visualizer.overlay_instances(boxes=instances.pred_boxes, masks=instances.pred_masks, assigned_colors=colors, alpha=0.3)

        return instances, vis_output, t1-t0
    

def setup_cfg(model_cfg, model_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_cfg)
    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg


def main(video_path, out_dir, demo, model_name):
    cap = cv2.VideoCapture(video_path)

    idx = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        
        predictions, visualized_output, inf_time = demo.run_on_image(frame)
        visualized_output = visualized_output.get_image()
        
        txt="%s Inference: %dx%d  GPU: %s Inference time %.3fs" % (model_name,750,1333,torch.cuda.get_device_name(0), inf_time)
        cv2.putText(visualized_output,txt, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0),19)
        cv2.putText(visualized_output,txt, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255),9)

        cv2.imwrite(os.path.join(out_dir, 'img%08d.jpg' % idx), visualized_output[:,:,::-1])
        idx+=1; pbar.update(1)
        del frame

    cap.release()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cfg', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--score_threshold', type=int, default=0.2)
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    cfg = setup_cfg(args.model_cfg, args.model_path)
    demo = Visualization(cfg, score_threshold=args.score_threshold)
    
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    main(args.video_path, args.out_dir, demo, model_name)
