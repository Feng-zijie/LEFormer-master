# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='LEFormer',
        in_channels=3,
        embed_dims=32,
        num_stages=4,

        # work_dirs/10_DenseNet_layer*4_layers+heads
        # num_layers=[3, 3, 3, 4],
        # num_heads=[1, 2, 5, 6],

        # 原始的层数
        num_layers=[2, 2, 2, 3],
        num_heads=[1, 2, 5, 6],

        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        pool_numbers=1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 192],
        in_index=[0, 1, 2, 3],
        channels=192,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,

        # work_dirs/11_DenseNet_layer*4_loss_decode
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        #     dict(type='DiceLoss', loss_weight=1.0)
        # ]
        
        
        # 原始损失函数
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)

    ),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
