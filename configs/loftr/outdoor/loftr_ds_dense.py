from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False

cfg.TRAINER.CANONICAL_LR = 0.008
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
# cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

cfg.TRAINER.MSLR_MILESTONES = [6, 8, 12, 16, 20, 24, 28, 32, 35, 36, 37, 38, 39]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
