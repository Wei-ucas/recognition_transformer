import os
resume_from = '/data1/ww/recognition/multi_instance_recognition.pytorch/cpks/no_crop/_200000.pth'
max_iters = 1000000
train_verbose = 100
val_iter = 1000
lr = 0.00005
save_iter = 100000
max_len = 25
grad_clip = 5
lr_step = 100000

save_name = './cpks/no_crop_1/'
if not os.path.exists(save_name):
    os.mkdir(save_name)

train_data_cfg = dict(
    data_dir='/data2/ww/SynthText',
    gt_dir='/data2/ww/SynthText/gt.mat',
    data_type='SynthText',
    num_instances=4,
    crop_ratio=0.0,
    crop_random=False,
    input_width=640,
    input_height=480,
    keep_ratio=False,
    max_len=max_len,
    batch_size=4,
    num_works=2,
    shuffle=True
)

# val_data_cfg = train_data_cfg
val_data_cfg = dict(
    data_dir='/data2/ww/icdar2013/test_images',
    gt_dir='/data2/ww/icdar2013/test_gt',
    data_type='ICDAR13',
    num_instances=4,
    crop_ratio=0.0,
    crop_random=False,
    input_width=640,
    input_height=1280,
    keep_ratio=True,
    max_len=max_len,
    batch_size=1,
    num_works=1,
    shuffle=False
)
from data_tools.data_utils import get_vocabulary
dim=512
seq_len = 25
voc_len = len(get_vocabulary("ALLCASES_SYMBOLS")[0])
model_cfg = dict(
    dim=512,
    seq_len = 25,
    voc_len = voc_len,
    num_instances=4,
    roi_size=(16,64),
    feature_channels=[256,512,1024,2048],
    fpn_out_channels=128,
    roi_feature_step=4,
    encoder_channels=512,
    embedding = dict(
        dim=dim,
        voc_len=voc_len,
        embedding_dim=512,
        pos_dim=seq_len,
        drop_rate=0.1,
    ),


    label_att = dict(
        dim=dim,
        n_heads=4,
        qdim=512,
        kdim=512,
        vdim=512,
        outdim=dim,
        dropout_rate=0.1
    ),

# '''img label att params'''
    att = dict(
        dim=dim,
        n_heads=4,
        qdim=512,
        kdim=512,
        vdim=512,
        outdim=dim,
        dropout_rate=0.1
    ),

    ff = dict(
        dim_ff=dim * 2,
        dim=dim
    )
)
