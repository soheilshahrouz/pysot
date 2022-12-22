# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs




class SiamRPNTemplateMaker(nn.Module):
    def __init__(self, model):
        super(SiamRPNTemplateMaker, self).__init__()

        # build backbone
        self.featureExtract = model.backbone.features

        self.conv_reg1 = model.rpn_head.loc.conv_kernel
        self.conv_cls1 = model.rpn_head.cls.conv_kernel


    def forward(self, x):

        x_perm = x.permute((0, 3, 1, 2))
        x_f = self.featureExtract(x_perm)
        return x_f, self.conv_reg1(x_f), self.conv_cls1(x_f)



class AnchorLayer(nn.Module):
    def __init__(self, anc, mem_len):
        super(AnchorLayer, self).__init__()
        
        anc_tiled = np.tile(anc, (mem_len, 1, 1))
        self.anchor0 = nn.Parameter( torch.from_numpy(anc_tiled[:, 0, :]) )
        self.anchor1 = nn.Parameter( torch.from_numpy(anc_tiled[:, 1, :]) )
        self.anchor2 = nn.Parameter( torch.from_numpy(anc_tiled[:, 2, :]) )
        self.anchor3 = nn.Parameter( torch.from_numpy(anc_tiled[:, 3, :]) )

    def forward(self, delta):
      
        delta_0_out = delta[:, 0, :] * self.anchor2 + self.anchor0
        delta_1_out = delta[:, 1, :] * self.anchor3 + self.anchor1
        delta_2_out = torch.exp(delta[:, 2, :]) * self.anchor2
        delta_3_out = torch.exp(delta[:, 3, :]) * self.anchor3

        return delta_0_out, delta_1_out, delta_2_out, delta_3_out


class SiamRPNTHORForward(nn.Module):
    def __init__(self, model, anc, mem_len):
        super(SiamRPNTHORForward, self).__init__()

        self.mem_len = mem_len
        # build backbone
        self.featureExtract = model.backbone.features

        self.conv_reg2 = model.rpn_head.loc.conv_search
        self.conv_cls2 = model.rpn_head.cls.conv_search

        self.reg_head = model.rpn_head.loc.head
        self.cls_head = model.rpn_head.cls.head
        
        self.anchors = AnchorLayer(anc, mem_len)


    def forward(self, x, z_reg, z_cls):

        x_perm = x.permute((0, 3, 1, 2))
        x_f = self.featureExtract(x_perm)

        c_x = self.conv_reg2(x_f)
        r_x = self.conv_cls2(x_f)

        c_x = F.unfold(c_x.reshape(256, 1, 24, 24), (4, 4))
        r_x = F.unfold(r_x.reshape(256, 1, 24, 24), (4, 4))

        c_x = c_x.permute((2, 0, 1))
        r_x = r_x.permute((2, 0, 1))

        z_reg = z_reg.reshape(self.mem_len, 256, 4*4)
        z_cls = z_cls.reshape(self.mem_len, 256, 4*4)

        z_reg = list(torch.split(z_reg, 1))
        z_cls = list(torch.split(z_cls, 1))

        r_out = [torch.mul(r_x, zr).sum(2).permute((1, 0)).reshape(1, 256, 21, 21) for zr in z_reg]
        c_out = [torch.mul(c_x, zc).sum(2).permute((1, 0)).reshape(1, 256, 21, 21) for zc in z_cls]
        
        reg_corr = torch.cat(r_out)
        cls_corr = torch.cat(c_out)
        
        cls = self.cls_head( cls_corr )
        reg = self.reg_head( reg_corr )

        score = F.softmax( cls.view(self.mem_len, 2, -1), dim=1)
        delta = reg.view(self.mem_len, 4, -1)
        delta_0_out, delta_1_out, delta_2_out, delta_3_out = self.anchors(delta)

        return delta_0_out, delta_1_out, delta_2_out, delta_3_out, score[:, 1, :]


class SiamRPNForward(nn.Module):
    def __init__(self, model, anc):
        super(SiamRPNForward, self).__init__()

        # build backbone
        self.featureExtract = model.backbone.features

        self.conv_reg2 = model.rpn_head.loc.conv_search
        self.conv_cls2 = model.rpn_head.cls.conv_search

        self.reg_head = model.rpn_head.loc.head
        self.cls_head = model.rpn_head.cls.head
        
        self.anchors = AnchorLayer(anc, 1)


    def forward(self, x, z_reg, z_cls):

        x_perm = x.permute((0, 3, 1, 2))
        x_f = self.featureExtract(x_perm)

        c_x = self.conv_cls2(x_f)
        r_x = self.conv_reg2(x_f)

        c_x = F.unfold(c_x.reshape(256, 1, 24, 24), (4, 4))
        r_x = F.unfold(r_x.reshape(256, 1, 24, 24), (4, 4))

        c_x = c_x.permute((2, 0, 1))
        r_x = r_x.permute((2, 0, 1))

        z_reg = z_reg.reshape(256, 4*4)
        z_cls = z_cls.reshape(256, 4*4)

        r_out = torch.mul(r_x, z_reg).sum(2).permute((1, 0)).reshape(1, 256, 21, 21)
        c_out = torch.mul(c_x, z_cls).sum(2).permute((1, 0)).reshape(1, 256, 21, 21)

        reg = self.reg_head(r_out)
        cls = self.cls_head(c_out)

        score = F.softmax( cls.view(2, -1), dim=0)

        delta = reg.view(1, 4, -1)

        delta_0_out, delta_1_out, delta_2_out, delta_3_out = self.anchors(delta)


        return delta_0_out, delta_1_out, delta_2_out, delta_3_out, score[1, :]