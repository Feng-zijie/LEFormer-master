_base_ = [
    '../_base_/models/leformer.py', '../_base_/datasets/qtpl_256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_240k.py'
]

model = dict(pretrained=None, decode_head=dict(num_classes=2))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0006, # 原始的 lr=0.00006
    betas=(0.9, 0.999),
    weight_decay=0.05, # 原始的 weight_decay=0.01
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=2.), # 原始的 'head': dict(lr_mult=10.)
            # 'backbone': dict(lr_mult=0.5) # 后来加的
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=16, workers_per_gpu=4)
