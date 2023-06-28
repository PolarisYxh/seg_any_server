from segment_anything import SamPredictor, sam_model_registry
import os
import cv2
import numpy as np
import os
class SAM:
    def __init__(self,model_type="vit_l",device="cpu") -> None:
        # model_type = "vit_l"
        rootpath = "./checkpoints"
        files = os.listdir(rootpath)
        file = list(filter(lambda name:model_type in name,files))[0]
        model_path = os.path.join(rootpath,file)
        sam = sam_model_registry[model_type](checkpoint=model_path)#b最小358mb，h最大2.4g，l中等1.2g
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        
    def get_mask(self,img,point_coords,point_labels):
        # img = cv2.imread("/home/shared/toolkits/Segment_anything_model/image_f.jpg")
        self.predictor.set_image(img)
        masks, scores, logits = self.predictor.predict(point_coords,point_labels,multimask_output=True)#,[329,329],0
        mask = masks[np.argmax(scores),:,:]
        # masked_img = np.zeros_like(img)
        # masked_img[masks] = img[masks]
        # cv2.imwrite("1.png",masked_img)
        return mask