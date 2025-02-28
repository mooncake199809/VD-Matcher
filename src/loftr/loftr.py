import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess, LocalFeatureTransformer_point
from .utils.coarse_matching import CoarseMatching
from .utils.obtain_corners import generate_corners
from .utils.fine_matching import FineMatching

class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.seed_num = config['coarse']['seed_num']
        self.seed_num_max = config['coarse']['seed_num_max']
        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])

        # img2img operation
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'], repeat_times=config['coarse']['repeat_time'])

        # point2point operation & point2img operation
        self.loftr_point_first = LocalFeatureTransformer_point(config['point'], repeat_times=config['point']['repeat_first'])

        self.use_base = config['point']['use_base']
        if self.use_base:
            self.loftr_point_second = LocalFeatureTransformer_point(config['point'], repeat_times=config['point']['repeat_second'])

        # img2img operation again
        self.loftr_point_last = LocalFeatureTransformer(config['point_last'], repeat_times=config['point_last']['repeat_last'])

        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine_first = LocalFeatureTransformer(config["fine"], repeat_times=config["fine"]["repeat_time"])

        if self.config["fine"]["use_base"]:
            self.loftr_fine_second = LocalFeatureTransformer(config["fine"], repeat_times=config["fine"]["repeat_time"])

        self.fine_matching = FineMatching()
        self.seed_selector = generate_corners()

    def calcul_assign_matrix_1(self, feat0, feat1, data):
        sim_matrix = torch.einsum("nlc,nsc->nls", feat0, feat1) / 0.1
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data["conf_matrix_1200_1"] = conf_matrix

    def calcul_assign_matrix_2(self, feat0, feat1, data):
        sim_matrix = torch.einsum("nlc,nsc->nls", feat0, feat1) / 0.1
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data["conf_matrix_1200_2"] = conf_matrix

    def forward(self, data):
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })
        
        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            "feat_c0": feat_c0,
            "feat_c1": feat_c1,
            "feat_f0": feat_f0,
            "feat_f1": feat_f1
        })

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })
        
        # 2. coarse-level loftr module
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)

        # img2img
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        # select keypoints
        if 'mask0' in data:
            b_index0_1, i_index_1 = self.seed_selector.generate_seed_forward(data, rearrange(feat_c0, 'n (h w) c -> n c h w', h = data['image0'].size(2) // 8), data['mask0'], [self.seed_num, self.seed_num_max])
            b_index0_1, j_index_1 = self.seed_selector.generate_seed_forward(data, rearrange(feat_c1, 'n (h w) c -> n c h w', h = data['image1'].size(2) // 8), data['mask1'], [self.seed_num, self.seed_num_max])
        else:
            b_index0_1, i_index_1 = self.seed_selector.generate_seed_forward(data, rearrange(feat_c0, 'n (h w) c -> n c h w', h = data['image0'].size(2) // 8), None, [self.seed_num, self.seed_num_max])
            b_index0_1, j_index_1 = self.seed_selector.generate_seed_forward(data, rearrange(feat_c1, 'n (h w) c -> n c h w', h = data['image1'].size(2) // 8), None, [self.seed_num, self.seed_num_max])
        data.update({"b_seed_idx1":b_index0_1[0], "i_seed_idx1":i_index_1[0], "j_seed_idx1":j_index_1[0],
                     "b_seed_idx1_max":b_index0_1[1], "i_seed_idx1_max":i_index_1[1], "j_seed_idx1_max":j_index_1[1]})

        featc0_edge, featc1_edge = feat_c0[data['b_seed_idx1'], data['i_seed_idx1']], \
                        feat_c1[data['b_seed_idx1'], data['j_seed_idx1']]
        featc0_edge_max, featc1_edge_max = feat_c0[data['b_seed_idx1_max'], data['i_seed_idx1_max']], \
                        feat_c1[data['b_seed_idx1_max'], data['j_seed_idx1_max']]

        mask_c0 = mask_c1 = None  # mask is useful in training
        mask0_edge = mask1_edge = None
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
            mask0_edge = mask_c0[data['b_seed_idx1'], data['i_seed_idx1']]
            mask1_edge = mask_c1[data['b_seed_idx1'], data['j_seed_idx1']]
            mask_c0 = mask_c0[data['b_seed_idx1_max'], data['i_seed_idx1_max']]
            mask_c1 = mask_c1[data['b_seed_idx1_max'], data['j_seed_idx1_max']]
            data['mask0'] = mask_c0
            data['mask1'] = mask_c1
        
        # point2point
        featc0_edge_max, featc1_edge_max, featc0_edge, featc1_edge = self.loftr_point_first(featc0_edge, featc1_edge, featc0_edge_max, featc1_edge_max, mask_c0, mask_c1, mask0_edge, mask1_edge)
        # self.calcul_assign_matrix_1(featc0_edge, featc1_edge, data)

        # point2point again
        if self.use_base:
            featc0_edge_max, featc1_edge_max, featc0_edge, featc1_edge = self.loftr_point_second(featc0_edge, featc1_edge, featc0_edge_max, featc1_edge_max, mask_c0, mask_c1, mask0_edge, mask1_edge)

        # img2img
        featc0_edge_max, featc1_edge_max = self.loftr_point_last(featc0_edge_max, featc1_edge_max, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(featc0_edge_max, featc1_edge_max, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, featc0_edge_max, featc1_edge_max, data)
        data.update({
            "feat_f0_unfold": feat_f0_unfold,
            "feat_f1_unfold": feat_f1_unfold
        })

        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine_first(feat_f0_unfold, feat_f1_unfold)
            if self.use_base:
                feat_f0_unfold, feat_f1_unfold = self.loftr_fine_second(feat_f0_unfold, feat_f1_unfold)

        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
