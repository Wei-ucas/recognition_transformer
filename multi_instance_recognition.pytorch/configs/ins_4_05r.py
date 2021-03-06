import os

resume_from = None
max_iters = 1000000
train_verbose = 100
val_iter = 1000
lr = 0.0001
save_iter = 100000
max_len = 25
grad_clip = 5
lr_step = 400000

save_name = './cpks/ins_4_05r/'
if not os.path.exists(save_name):
    os.mkdir(save_name)


train_data_cfg = dict(
    data_dir='/data2/ww/SynthText',
    gt_dir='/data2/ww/SynthText/gt.mat',
    data_type='SynthText',
    num_instances=4,
    crop_ratio=0.4,
    crop_random=True,
    input_width=640,
    input_height=320,
    max_len=max_len,
    batch_size=2,
    num_works=2,
    shuffle=True
)

# val_data_cfg = train_data_cfg
val_data_cfg = dict(
    data_dir='/data2/ww/15ICDAR/ch4_test_images',
    gt_dir='/data2/ww/15ICDAR/ch4_test_gt',
    data_type='ICDAR',
    crop_ratio=0.5,
    crop_random=True,
    num_instances=4,
    input_width=1280,
    input_height=960,
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
