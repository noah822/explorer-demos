import cv2
import numpy as np

from typing import Iterable

from core.interfaces import Observations
from perception.detection.detic.detic_perception import DeticPerception


class SemanticMoudle:
    def __init__(self, vocab: Iterable='coco', device_id: int=0):
        '''
        Args:
            vocab: name of classes to detect
            if set to None, default COCO vocabulary will be used
        '''
        assert vocab in ['coco', 'lvis', 'custom'], \
            'internally support `coco` and `lvis` or one can choose `custom`'
        self.vocab_content = '' if vocab != 'custom' else ','.join(vocab)

        self.model = DeticPerception(
            vocabulary=vocab,
            custom_vocabulary=self.vocab_content,
            sem_gpu_id=device_id
        )
        self.categories_mapping = self.model.categories_mapping    
        self.num_sem_categories = len(self.categories_mapping)
    
    def classname2index(self, class_name):
        pass
    def classindex2name(self, class_index):
        pass

    def predict(self, img: np.ndarray, accept_threshold=0.5):
        # alias for __call__
        return self(img, accept_threshold)

    def __call__(self, img: np.ndarray, accept_threshold=0.5):
       h, w = img.shape[:2]

       # configure other parameter to dummy value except for rgb image
       observe = Observations(gps=None, compass=None, rgb=img, depth=None)
       seg_res = self.model.predict(observe).task_observations

       instance_classes = [self.categories_mapping.get(raw_id, -1) for raw_id in seg_res['instance_classes']]
       # prepross instance detection result of detic
       # expected behavior: each pixel is associated with a feature vector 
       # of shape (1+num_sem_categories): the first dimension is associated with no object of interest
       # example: pixel(2, 2) is determined to be of first class (class-indexed as zero) of probability 0.8
       # then its pixel feature vector is (0., 0.8, 0., 0., ...)
       pixel_feature = np.zeros((h, w, self.num_sem_categories + 1))

       # clip instance with confidence less than threshold
       scores = seg_res['instance_scores']
       scores[scores < accept_threshold] = 0.
       scores = np.concatenate([[1.], scores])

       ins_id2cls_id = np.concatenate([[0], instance_classes]).astype(np.int32)

       # instance_map is of shape (h, w) with value spanning from [-1, num_instaces-1]
       # -1 denotes nothing
       masked_img_with_instance_id = seg_res['instance_map'].astype(np.int32) + 1
       masked_img_with_scores = scores[masked_img_with_instance_id]
       masked_img_with_class_id = ins_id2cls_id[masked_img_with_instance_id] 

       pixel_feature[
           np.arange(h).reshape(-1, 1),
           np.arange(w).reshape(1, -1),
           masked_img_with_class_id
       ] = 1
       pixel_feature *= masked_img_with_scores[..., np.newaxis]

       return {
           'annotated_frame' : seg_res['semantic_frame'],
           'pixel_features' : pixel_feature
       }


