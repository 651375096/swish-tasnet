import os
import argparse
import json
import pdb

import argparse

import torch
import torch.distributed as dist # 其一！导入库函数

# gpus = [0, 1, 2, 3] # TODO need to be updated based on real-world gpu numbers
# torch.cuda.set_device('cuda:{}'.format(gpus[0])) # 其二！设置当前default gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['RANK'] = '3'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '172.24.216.120'
os.environ['MASTER_PORT'] = '22'
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # 按照PCI BUS ID顺序从0开始排列GPU设备
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # 设置当前程序仅使用第0、1块GPU 运行
# device_ids = [0,1]#选中其中两块
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
# from src.Conv_TasNet.conv_tasnet import  TasNet
# import asteroid
# from asteroid.models import ConvTasNet, DPRNNTasNet, DPTNet
# from asteroid.data import LibriMix, WhamDataset, Wsj0mixDataset
# from asteroid.engine.optimizers import make_optimizer
# from asteroid.engine.syst em import System
# from asteroid.engine.schedulers import DPTNetScheduler
# from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
import asteroid
from src.asteroid.models import ConvTasNet, DPRNNTasNet, DPTNet
from src.asteroid.data import LibriMix, WhamDataset, Wsj0mixDataset
from src.asteroid.engine.optimizers import make_optimizer
from src.asteroid.engine.system import System
from src.asteroid.engine.schedulers import DPTNetScheduler
from src.asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from utils import make_dataloaders,make_txt_dataloaders, MultiTaskLossWrapper
def init_distributed_mode(args):
    # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
    # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
    if'RANK'in os.environ and'WORLD_SIZE'in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        # LOCAL_RANK代表某个机器上第几块GPU
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif'SLURM_PROCID'in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]
# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
# parser.add_argument("--corpus", default="wsj0-mix", choices=["LibriMix", "wsj0-mix"])

parser.add_argument("--corpus", default="wsj0-mix", choices=["LibriMix", "wsj0-mix","txt"])
parser.add_argument("--model", default="ConvTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet","CYTasNet"])
parser.add_argument("--strategy", default="from_scratch", choices=["from_scratch", "pretrained", "multi_task"])
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
parser.add_argument("--save_top_k", type=int, default=5, help="Save top k checkpoints. -1 for saving all checkpoints.")
parser.add_argument("--real_batch_size", type=int, default=0, help="Batch size on all gpu when using accumulate gradients.")
parser.add_argument("--resume_ckpt", default=None , help="Checkpoint path to load for resume-training")
##############

parser.add_argument('--local_rank', default=3 , type=int,
                    help='node rank for distributed training') # 其二，local_rank=current gpu
args = parser.parse_args()

dist.init_process_group(backend='nccl') # 其三，初始化dist，根据nccl as backend
torch.cuda.set_device(args.local_rank) # 其四，设置当前线程所在的gpu


args = parser.parse_args()
print(args.local_rank)


known_args = parser.parse_known_args()[0]


# parser.par se_known_args()[0] Namespace(corpus='wsj0-mix', exp_dir='exp/tmp',
# model='ConvTasNet', real_batch_size=0, resume_ckpt=None, save_top_k=5, strategy='from_scratch')

if known_args.strategy == "pretrained":###known_args.strategy==from_scratch
    parser.add_argument("--load_path", default='epoch=29-step=512249.ckpt', help="Checkpoint path to load for fine-tuning.")
elif known_args.strategy == "multi_task":
    parser.add_argument("--train_enh_dir", default=None, help="Multi-task data dir.")


def main(conf):

    train_enh_dir = None if conf["main_args"]["strategy"] != "multi_task" else conf["main_args"]["train_enh_dir"]
    batch_size = conf["training"]["batch_size"] if conf["main_args"]["real_batch_size"] == 0 else conf["main_args"]["real_batch_size"]
    accumulate_grad_batches=int(conf["training"]["batch_size"] / batch_size)

    # print(conf["data"]["train_dir"])==./voice_data/json/tr
    nsrv=conf["data"]["n_src"]
    print("开始train_loader",nsrv)



    train_loader, val_loader, train_set_infos = make_txt_dataloaders(
        corpus=conf["main_args"]["corpus"],#'wsj0-mix',
        train_dir=conf["data"]["train_dir"],#'./voice_data/json/tr'
        val_dir=conf["data"]["valid_dir"],
        train_enh_dir=train_enh_dir,#None
        task=conf["data"]["task"],#'sep_clean'
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],#(3.0,)
        batch_size=batch_size,
        num_workers=conf["training"]["num_workers"],#8,)
    )

    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader)
    #
    # train_loader = torch.utils.data.DataLoader(train_loader, batch_size=..., sampler=train_sampler)
    #
    # torch.cuda.set_device(args.local_rank)

    #conf["main_args"]["strategy"]==from_scratch
    if conf["main_args"]["strategy"] != "multi_task":
        conf["masknet"].update({"n_src": conf["data"]["n_src"]})
    else:
        conf["masknet"].update({"n_src": conf["data"]["n_src"]+1})



    model = getattr(asteroid, conf["main_args"]["model"])(**conf["filterbank"], **conf["masknet"])
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # model = nn.DataParallel(model)
    # model = model.cuda()
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
    # torch.distributed.init_process_group(backend="nccl")
    # model = DistributedDataParallel(model)  # device_ids will include all GPU devices by default
    # torch.distributed.init_process_group(backend="nccl")
    # model = DistributedDataParallel(model)  # device_ids will include all GPU devices by default
    # model = model.cuda()

    # model = nn.parallel.DistributedDataParallel(model)




    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    #
    # model = model.cuda()
    # model = nn.parallel.DistributedDataParallel(model)
    # model=torch.nn.DataParalle(model.cuda(), device_ids=[0, 1, 2, 3])
    # torch.distributed.init_process_group(backend="nccl")
    # model = DistributedDataParallel(model)    # model = torch.nn.DataParallel(model)
    # model=nn.DataParallel(model, device_ids=device_ids)  # 并行使用两块
    # enc_dim = 512, feature_dim = 128, sr = 16000, win = 2, layer = 8, stack = 3,
    # kernel = 3, num_spk = 2, causal = False
    # conf["main_args"]["model"]=ConvTasNet
    # conf["filterbank"]={'kernel_size': 18, 'n_filters': 512, 'stride': 8}
    # conf["masknet"]={'bn_chan': 128, 'hid_chan': 512, 'mask_act': 'relu', 'n_blocks': 8, 'n_repeats': 3, 'skip_chan': 128, 'n_src': 2}

    #conf["main_args"]["strategy"] ="from_scratch"
    if conf["main_args"]["strategy"] == "pretrained":#default="from_scratch"
        if conf["main_args"]["load_path"] is not None:
            all_states = torch.load(conf["main_args"]["load_path"], map_location="cpu")
            print("all_states",all_states)
            assert "state_dict" in all_states

            # If the checkpoint is not the serialized "best_model.pth", its keys
            # would start with "model.", which should be removed to avoid none
            # of the parameters are loaded.
            for key in list(all_states["state_dict"].keys()):
                if key.startswith("model"):
                    all_states["state_dict"][key.split('.', 1)[1]] = all_states["state_dict"][key]
                    del all_states["state_dict"][key]

            # For debugging, set strict=True to check whether only the following
            # parameters have different sizes (since n_src=1 for pre-training
            # and n_src=2 for fine-tuning):
            # for ConvTasNet: "masker.mask_net.1.*"
            # for DPRNNTasNet/DPTNet: "masker.first_out.1.*"


            if conf["main_args"]["model"] == "ConvTasNet":
                del all_states["state_dict"]["masker.mask_net.1.weight"]
                del all_states["state_dict"]["masker.mask_net.1.bias"]
            elif conf["main_args"]["model"] in ["DPRNNTasNet", "DPTNet"]:
                del all_states["state_dict"]["masker.first_out.1.weight"]
                del all_states["state_dict"]["masker.first_out.1.bias"]
            model.load_state_dict(all_states["state_dict"], strict=False)

    ##优化器
    optimizer = make_optimizer(model.parameters(), **conf["optim"])

    # 定义调度程序Define scheduler

    scheduler = None
    # print(conf["main_args"]["model"])==ConvTasNet]
    if conf["main_args"]["model"] == "DPTNet":
        steps_per_epoch = len(train_loader) // accumulate_grad_batches
        # print("steps_per_epoch",steps_per_epoch)
        conf["scheduler"]["steps_per_epoch"] = steps_per_epoch
        scheduler = {
            "scheduler": DPTNetScheduler(
                optimizer=optimizer,
                steps_per_epoch=steps_per_epoch,
                d_model=model.masker.mha_in_dim,
            ),
            "interval": "batch",
        }
    # conf["training"]["half_lr"]) == True
    elif conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)##降低高原上的LR

    # 实例化之后，保存参数。未来轻松加载。Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]#(conf["main_args"]["exp_dir"])==exp/tmp
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    if conf["main_args"]["strategy"] == "multi_task":#策略=“从头开始”strategy='from_scratch
        loss_func = MultiTaskLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    else:
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=conf["main_args"]["save_top_k"], verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))


    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    print("gpus",gpus)
    #################
    # distributed_backend = "dp" if torch.cuda.is_available() else None   # Don't use ddp for multi-task training
    distributed_backend = "dp" if torch.cuda.is_available() else None   # Don't use ddp for multi-task training
    # print("distributed_backend",distributed_backend)
    # distributed_backend = "dp"
   #####################


    # print("distributed_backend ",distributed_backend )
    print("开始trainer = pl.Trainer")
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,  # With some unknown problems
        #checkpoint_callback=checkpoint,
        #early_stop_callback=callbacks[1],
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        # fast_dev_run=True, # Useful for debugging
        # overfit_pct=0.001, # Useful for debugging
        gradient_clip_val=5.0,
        accumulate_grad_batches=accumulate_grad_batches,
        resume_from_checkpoint=conf["main_args"]["resume_ckpt"],

    )
    print("开始trainer.fit(system)")
    trainer.fit(system)
    print("结束trainer.fit(system)")

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
   ######################

    system.cpu()

#################

    to_save = system.model.serialize()
    to_save.update(train_set_infos)
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    print("结束保存")

def init_distributed_mode(args):
    # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
    # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        # LOCAL_RANK代表某个机器上第几块GPU
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True



if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from src.asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    model_type = known_args.model
    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open(f"local/{model_type}.yml", 'r', encoding='utf-8') as f:
        def_conf = yaml.safe_load(f)

        parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    # print(" plain_args", plain_args)
    print("arg_dic",arg_dic)
    main(arg_dic)
