# Copyright (c) OpenMMLab. All rights reserved.
import torch


def bbox3d2roi(bbox_list):
    """Convert a list of bounding boxes to roi format."""
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes], dim=-1)
        else:
            rois = torch.zeros_like(bboxes)
        rois_list.append(rois)
    return torch.cat(rois_list, 0)


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a dict of cpu tensors."""
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu(),
    )
    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()
    return result_dict
