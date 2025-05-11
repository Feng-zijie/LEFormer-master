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

        # √的层数
        # num_layers=[2, 2, 3, 6],
         
        num_layers=[2, 2, 3, 6],
        num_heads=[1, 2, 5, 6],

        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        drop_rate=0.0,
        ffn_classes=3),
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
