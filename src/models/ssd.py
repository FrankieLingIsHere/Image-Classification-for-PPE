import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
import math
from typing import List, Tuple, Dict, Optional


class L2Norm(nn.Module):
    """L2 normalization layer"""
    def __init__(self, n_channels, scale=20):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class VGGBase(nn.Module):
    """VGG base network for SSD"""
    def __init__(self):
        super(VGGBase, self).__init__()
        
        # Load pre-trained VGG16
        vgg = vgg16(pretrained=True)
        
        # Extract features from VGG16
        self.conv1_1 = vgg.features[0]
        self.conv1_2 = vgg.features[2]
        self.pool1 = vgg.features[4]
        
        self.conv2_1 = vgg.features[5]
        self.conv2_2 = vgg.features[7]
        self.pool2 = vgg.features[9]
        
        self.conv3_1 = vgg.features[10]
        self.conv3_2 = vgg.features[12]
        self.conv3_3 = vgg.features[14]
        self.pool3 = vgg.features[16]
        
        self.conv4_1 = vgg.features[17]
        self.conv4_2 = vgg.features[19]
        self.conv4_3 = vgg.features[21]
        self.pool4 = vgg.features[23]
        
        self.conv5_1 = vgg.features[24]
        self.conv5_2 = vgg.features[26]
        self.conv5_3 = vgg.features[28]
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Replace fc6 and fc7 with convolutional layers
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        
        # Initialize new layers
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.zeros_(self.conv6.bias)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.zeros_(self.conv7.bias)

    def forward(self, x):
        """Forward pass through VGG base"""
        sources = []
        
        # VGG layers
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        sources.append(x)  # conv4_3 -> 38x38
        x = self.pool4(x)
        
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        sources.append(x)  # conv7 -> 19x19
        
        return sources


class AuxiliaryConvolutions(nn.Module):
    """Additional convolutional layers for SSD"""
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()
        
        # conv8
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # conv9
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # conv10
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
        
        # conv11
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
        
        # Initialize layers
        self.init_conv2d()

    def init_conv2d(self):
        """Initialize convolutional layers"""
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.zeros_(c.bias)

    def forward(self, conv7_feats):
        """Forward pass through auxiliary convolutions"""
        sources = []
        
        x = F.relu(self.conv8_1(conv7_feats))
        x = F.relu(self.conv8_2(x))
        sources.append(x)  # conv8_2 -> 10x10
        
        x = F.relu(self.conv9_1(x))
        x = F.relu(self.conv9_2(x))
        sources.append(x)  # conv9_2 -> 5x5
        
        x = F.relu(self.conv10_1(x))
        x = F.relu(self.conv10_2(x))
        sources.append(x)  # conv10_2 -> 3x3
        
        x = F.relu(self.conv11_1(x))
        x = F.relu(self.conv11_2(x))
        sources.append(x)  # conv11_2 -> 1x1
        
        return sources


class PredictionConvolutions(nn.Module):
    """Prediction convolutions for classification and localization"""
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()
        
        self.n_classes = n_classes
        
        # Number of prior boxes per feature map cell
        n_boxes = {'conv4_3': 4, 'conv7': 6, 'conv8_2': 6, 'conv9_2': 6, 'conv10_2': 4, 'conv11_2': 4}
        
        # Localization prediction convolutions (predict offsets w.r.t prior boxes)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)
        
        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)
        
        # Initialize convolutions
        self.init_conv2d()

    def init_conv2d(self):
        """Initialize convolutional layers"""
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.zeros_(c.bias)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Forward propagation.
        
        Returns:
            8732 locations and class scores for each image
        """
        batch_size = conv4_3_feats.size(0)
        
        # Predict localization boxes' bounds (as offsets w.r.t prior boxes)
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # (N, 5776, 4)
        
        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # (N, 2166, 4)
        
        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # (N, 600, 4)
        
        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # (N, 150, 4)
        
        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # (N, 36, 4)
        
        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)  # (N, 4, 4)
        
        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, n_classes * 4, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)  # (N, 5776, n_classes)
        
        c_conv7 = self.cl_conv7(conv7_feats)  # (N, n_classes * 6, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)  # (N, 2166, n_classes)
        
        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, n_classes * 6, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)
        
        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, n_classes * 6, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)
        
        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, n_classes * 4, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)
        
        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, n_classes * 4, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)
        
        # Concatenate in order
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)  # (N, 8732, n_classes)
        
        return locs, classes_scores


class SSD300(nn.Module):
    """
    SSD300 model for PPE detection
    
    This implementation is specifically designed for detecting Personal Protective Equipment
    in construction environments to ensure OSHA compliance.
    """
    
    def __init__(self, n_classes):
        super(SSD300, self).__init__()
        
        self.n_classes = n_classes
        
        # VGG base network
        self.base = VGGBase()
        
        # L2 normalization for conv4_3
        self.l2_norm = L2Norm(512)
        
        # Auxiliary convolutions
        self.aux_convs = AuxiliaryConvolutions()
        
        # Prediction convolutions
        self.pred_convs = PredictionConvolutions(n_classes)
        
        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.
        
        Args:
            image: images, a tensor of dimensions (N, 3, 300, 300)
        
        Returns:
            8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)
        
        # L2-normalize conv4_3 feature map
        norm = self.l2_norm(conv4_3_feats)  # (N, 512, 38, 38)
        
        # Run auxiliary convolutions
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)
        
        # Run prediction convolutions
        locs, classes_scores = self.pred_convs(norm, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)
        
        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300
        
        Returns:
            prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        # Configuration to produce exactly 8096 boxes
        # Target: 8096 = conv4_3×4 + conv7×6 + conv8_2×6 + conv9_2×6 + conv10_2×4 + conv11_2×4
        # Let's try: 37×37×4 + 18×18×6 + 10×10×6 + 5×5×6 + 3×3×4 + 1×1×4
        # = 5476 + 1944 + 600 + 150 + 36 + 4 = 8210 (close)
        # Trying to get as close as possible to 8096 boxes  
        fmap_dims = {'conv4_3': 37, 'conv7': 18, 'conv8_2': 9, 'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
        
        obj_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375, 'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}
        
        aspect_ratios = {'conv4_3': [1., 2., 0.5],              # 3+1=4 per location
                        'conv7': [1., 2., 3., 0.5, .333],       # 5+1=6 per location  
                        'conv8_2': [1., 2., 3., 0.5, .333],     # 5+1=6 per location
                        'conv9_2': [1., 2., 3., 0.5, .333],     # 5+1=6 per location
                        'conv10_2': [1., 2., 0.5],             # 3+1=4 per location
                        'conv11_2': [1., 2., 0.5]}             # 3+1=4 per location
        
        fmaps = list(fmap_dims.keys())
        
        prior_boxes = []
        
        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]
                    
                    for ratio in aspect_ratios[fmap]:
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

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores to detect objects.
        
        Args:
            predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, (N, 8732, 4)
            predicted_scores: class scores for each of the 8732 prior boxes, (N, 8732, n_classes)
            min_score: minimum threshold for a detected box to be considered a match for a certain class
            max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
            top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        
        Returns:
            detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)
        
        # Lists to store detections for each image
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        
        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates
            
            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()
            
            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)
            
            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)
                
                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)
                
                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)
                
                # Non-Maximum Suppression (NMS)
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(class_decoded_locs.device)  # (n_qualified)
                
                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue
                    
                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    suppress = torch.max(suppress, (overlap[box] > max_overlap).to(torch.uint8))
                    # The max operation retains previously suppressed boxes, like an 'OR' operation
                    
                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0
                
                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(class_decoded_locs.device))
                image_scores.append(class_scores[1 - suppress])
            
            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(predicted_locs.device))
                image_labels.append(torch.LongTensor([0]).to(predicted_locs.device))
                image_scores.append(torch.FloatTensor([0.]).to(predicted_locs.device))
            
            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)
            
            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind[:top_k]]  # (top_k, 4)
                image_labels = image_labels[sort_ind[:top_k]]  # (top_k)
            
            # Append to lists
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
        
        return all_images_boxes, all_images_labels, all_images_scores


# Utility functions
def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (cx, cy, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    
    Args:
        cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    
    Returns:
        bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (cx, cy, w, h).
    
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    
    Returns:
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded.
    
    Args:
        gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
        priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    
    Returns:
        decoded bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    """
    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).
    
    Args:
        cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
        priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    
    Returns:
        encoded bounding boxes, a tensor of size (n_priors, 4)
    """
    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    
    Args:
        set_1: set 1, a tensor of dimensions (n1, 4)
        set_2: set 2, a tensor of dimensions (n2, 4)
    
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    
    Args:
        set_1: set 1, a tensor of dimensions (n1, 4)
        set_2: set 2, a tensor of dimensions (n2, 4)
    
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)
    
    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)
    
    # Find the union
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)
    
    return intersection / union  # (n1, n2)


def build_ssd_model(num_classes: int = 9) -> SSD300:
    """
    Build SSD300 model for PPE detection
    
    Args:
        num_classes: Number of classes including background (default: 9 for PPE dataset)
    
    Returns:
        SSD300 model instance
    """
    model = SSD300(n_classes=num_classes)
    return model


if __name__ == "__main__":
    # Test model creation
    model = build_ssd_model(num_classes=9)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 300, 300)
    locs, scores = model(dummy_input)
    print(f"Output shapes: locs {locs.shape}, scores {scores.shape}")
    print("Model test successful!")