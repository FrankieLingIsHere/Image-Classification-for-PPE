import torch
import torch.nn as nn
import torch.nn.functional as F
from .ssd import find_jaccard_overlap, cxcy_to_gcxgcy, xy_to_cxcy


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss function for SSD object detection
    
    This loss function combines:
    1. Classification loss (cross-entropy) for object categories
    2. Localization loss (smooth L1) for bounding box regression
    """
    
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.0):
        """
        Initialize MultiBox loss
        
        Args:
            priors_cxcy: prior boxes in center-size coordinates, a tensor of size (n_priors, 4)
            threshold: overlap threshold for matching ground truth boxes with priors
            neg_pos_ratio: ratio of negative samples to positive samples
            alpha: weighting factor for localization loss
        """
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation
        
        Args:
            predicted_locs: predicted locations/boxes w.r.t the prior boxes, (N, 8732, 4)
            predicted_scores: class scores for each of the prior boxes, (N, 8732, n_classes)
            boxes: ground truth object bounding boxes in boundary coordinates, list of N tensors
            labels: ground truth object labels, list of N tensors
            
        Returns:
            multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(predicted_locs.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(predicted_locs.device)
        
        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 8732)
            
            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)
            
            # For each object, find the prior that has the maximum overlap
            _, prior_for_each_object = overlap.max(dim=1)  # (n_objects)
            
            # Assign each object to the corresponding maximum-overlap-prior
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(predicted_locs.device)
            
            # To ensure that these priors qualify, artificially give them an overlap of greater than 0.5
            overlap_for_each_prior[prior_for_each_object] = 1.
            
            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)
            
            # Store
            true_classes[i] = label_for_each_prior
            
            # Encode center-size object coordinates w.r.t. the corresponding prior boxes' coordinates
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)
        
        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)
        
        # LOCALIZATION LOSS
        # Localization loss is computed only over positive (non-background) priors
        if positive_priors.sum() > 0:
            loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]).mean()
        else:
            loc_loss = torch.tensor(0.0, device=predicted_locs.device)
        
        # CONFIDENCE LOSS
        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE, we will take the hardest (neg_pos_ratio * n_positives) negative priors
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)
        
        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)
        
        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))
        
        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(predicted_locs.device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))
        
        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_pos.sum() + conf_loss_hard_neg.sum()) / n_positives.sum().float()  # (), scalar
        
        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (cx, cy, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def create_prior_boxes():
    """Create prior boxes for SSD300"""
    fmap_dims = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10, 'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
    
    obj_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375, 'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}
    
    aspect_ratios = {'conv4_3': [1., 2., 0.5],
                    'conv7': [1., 2., 3., 0.5, .333],
                    'conv8_2': [1., 2., 3., 0.5, .333],
                    'conv9_2': [1., 2., 3., 0.5, .333],
                    'conv10_2': [1., 2., 0.5],
                    'conv11_2': [1., 2., 0.5]}
    
    fmaps = list(fmap_dims.keys())
    
    prior_boxes = []
    
    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]
                
                for ratio in aspect_ratios[fmap]:
                    import math
                    prior_boxes.append([cx, cy, obj_scales[fmap] * math.sqrt(ratio), obj_scales[fmap] / math.sqrt(ratio)])
                    
                    # For aspect ratio of 1, add an extra box with scale s'
                    if ratio == 1.:
                        try:
                            additional_scale = math.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])
    
    prior_boxes = torch.FloatTensor(prior_boxes).clamp_(0, 1)  # (8732, 4)
    
    return prior_boxes


class PPELoss(MultiBoxLoss):
    """
    Specialized loss function for PPE detection
    
    This extends the standard MultiBox loss with PPE-specific considerations:
    - Higher penalty for missing critical PPE (hard hats, safety vests)
    - Weighted loss based on OSHA violation severity
    """
    
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.0, ppe_weights=None):
        """
        Initialize PPE-specific loss
        
        Args:
            priors_cxcy: prior boxes in center-size coordinates
            threshold: overlap threshold for matching ground truth boxes with priors
            neg_pos_ratio: ratio of negative samples to positive samples
            alpha: weighting factor for localization loss
            ppe_weights: class weights for different PPE violations (dict)
        """
        super(PPELoss, self).__init__(priors_cxcy, threshold, neg_pos_ratio, alpha)
        
        # Default PPE class weights (higher for critical safety violations)
        if ppe_weights is None:
            self.ppe_weights = {
                0: 1.0,   # background
                1: 1.0,   # person
                2: 2.0,   # hard_hat (critical)
                3: 2.0,   # safety_vest (critical)
                4: 1.5,   # safety_gloves
                5: 1.5,   # safety_boots
                6: 1.5,   # eye_protection
                7: 3.0,   # no_hard_hat (critical violation)
                8: 3.0,   # no_safety_vest (critical violation)
            }
        else:
            self.ppe_weights = ppe_weights
        
        # Convert to tensor
        max_class = max(self.ppe_weights.keys())
        weight_tensor = torch.ones(max_class + 1)
        for class_id, weight in self.ppe_weights.items():
            weight_tensor[class_id] = weight
        
        self.class_weights = weight_tensor

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation with PPE-specific weighting
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        
        # Move class weights to the same device as predictions
        self.class_weights = self.class_weights.to(predicted_scores.device)
        
        # Standard multibox loss computation
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(predicted_locs.device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(predicted_locs.device)
        
        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            
            if n_objects == 0:
                continue
                
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)  # (n_objects, 8732)
            
            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)
            
            # For each object, find the prior that has the maximum overlap
            _, prior_for_each_object = overlap.max(dim=1)  # (n_objects)
            
            # Assign each object to the corresponding maximum-overlap-prior
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(predicted_locs.device)
            
            # To ensure that these priors qualify, artificially give them an overlap of greater than 0.5
            overlap_for_each_prior[prior_for_each_object] = 1.
            
            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)
            
            # Store
            true_classes[i] = label_for_each_prior
            
            # Encode center-size object coordinates w.r.t. the corresponding prior boxes' coordinates
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)
        
        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)
        
        # LOCALIZATION LOSS
        if positive_priors.sum() > 0:
            loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]).mean()
        else:
            loc_loss = torch.tensor(0.0, device=predicted_locs.device)
        
        # WEIGHTED CONFIDENCE LOSS
        # Apply class weights to the cross entropy loss
        weighted_cross_entropy = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)
        
        # Find the loss for all priors with class weights
        conf_loss_all = weighted_cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)
        
        # Positive priors loss
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))
        
        # Hard negative mining
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(predicted_locs.device)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))
        
        # Average over positive priors
        total_positives = n_positives.sum().float()
        if total_positives > 0:
            conf_loss = (conf_loss_pos.sum() + conf_loss_hard_neg.sum()) / total_positives
        else:
            conf_loss = torch.tensor(0.0, device=predicted_scores.device)
        
        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss


if __name__ == "__main__":
    # Test loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create prior boxes
    priors_cxcy = create_prior_boxes()
    
    # Create loss function
    criterion = PPELoss(priors_cxcy)
    
    # Dummy data
    batch_size = 2
    n_classes = 9
    predicted_locs = torch.randn(batch_size, 8732, 4)
    predicted_scores = torch.randn(batch_size, 8732, n_classes)
    
    # Dummy ground truth
    boxes = [torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]), 
             torch.tensor([[0.2, 0.2, 0.4, 0.4]])]
    labels = [torch.tensor([2, 3]), torch.tensor([7])]
    
    # Compute loss
    loss = criterion(predicted_locs, predicted_scores, boxes, labels)
    print(f"PPE Loss: {loss.item():.4f}")
    print("Loss computation successful!")