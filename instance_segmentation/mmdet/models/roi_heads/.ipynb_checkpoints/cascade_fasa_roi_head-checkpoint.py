from mmdet.core import bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import CascadeRoIHead


@HEADS.register_module()
class StandardFASACascadeRoIHead(CascadeRoIHead):

    def _bbox_forward(self,stage, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        if self.training:
            cls_score, bbox_pred, embedding = bbox_head(bbox_feats)

            bbox_results = dict(
                cls_score=cls_score,
                bbox_pred=bbox_pred,
                bbox_feats=bbox_feats,
                embedding=embedding)
        else:
            cls_score, bbox_pred = bbox_head(bbox_feats)

            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred,
                bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self,stage, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage,x, rois)

        bbox_targets = self.bbox_head[stage].get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg[stage])

        embedding = bbox_results.get('embedding', None)

        loss_bbox = self.bbox_head[stage].loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            *bbox_targets,
            embedding=embedding)

        bbox_results.update(loss_bbox=loss_bbox,rois=rois, bbox_targets=bbox_targets)
        return bbox_results
