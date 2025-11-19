import argparse
import os

import time

from copy import deepcopy

from PIL import Image
import numpy as np
import csv

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import config
import torch.nn.functional as F
import torch.nn as nn
from thop import profile

import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.custom_clip import get_coop, GradCAM, VClip
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.biovid_prompts import biosub_classes, raftrains_classes, raftests_classes
from data.stress_prompts import stresssub_classes
from data.bah_prompts import bahssub_classes

from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, create_target_folders, MetricLogger, load_bah_src_subs
from data.cls_to_names import *

from utils.visualize import *

from utils.cometml import comet_init, set_comet_exp_name
from data.action_units_prompts import AU_PROMPTS, CLASS_PROMPTS, CLASS_PROMPTS_AMBV
from clip import load, tokenize
from datasets.base_dataset import BaseDataset
from utils.reproducibility import get_default_seed

from tqdm import tqdm
from collections import OrderedDict

experiment = comet_init(config.COMET_PROJECT_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def entropy_loss(logits, class_weights=None):

    # weighted entropy to avoid collapse
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(class_weights * probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    ent = - (probs * log_probs).sum(dim=-1).mean()
    return ent

def load_txt_adapter_classifier(save_path, model, args, device):
    checkpoint = torch.load(save_path, map_location=device)
    if args.train_whole_clip_model:
        model.load_state_dict(checkpoint)
        print("[INFO] CLIP Model Loaded: ", save_path)
    else:
        if "text_adapter" in checkpoint:
            model.text_adapter.load_state_dict(checkpoint['text_adapter'])    
        if "au_classifier" in checkpoint:
            model.au_classifier.load_state_dict(checkpoint["au_classifier"])
        if "temporal_classifier" in checkpoint:
            model.temporal_classifier.load_state_dict(checkpoint["temporal_classifier"])
        if "temporal" in checkpoint:
            model.temporal.load_state_dict(checkpoint["temporal"])
        if "temporal_proj" in checkpoint:
            model.temporal_proj.load_state_dict(checkpoint["temporal_proj"])

    model = model.cuda(args.gpu)
    return model

def rebuild_optimizer(model, lr):
    trainable_params = model.au_prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    return optimizer

def main():
    args = parser.parse_args()
    set_random_seed(get_default_seed())

    if args.current_ds is config.BIOVID:
        '''
        BioVid Random Source subjects
        '''
        source_list_name = ['082208_w_45', '081714_m_36', '112610_w_60', '101908_m_61', '071709_w_23','082014_w_24', '110810_m_62', '080209_w_26', '101916_m_40', '110614_m_42',
        '101814_m_58', '112016_m_25', '071313_m_41', '102514_w_40', '100514_w_51', '101114_w_37', '100509_w_43', '082315_w_60', '112310_m_20', '120614_w_61', 
        '092714_m_64', '101514_w_36', '092813_w_24', '102414_w_58', '102309_m_61', '081617_m_27', '080609_w_27', '083114_w_55', '111313_m_64', '071614_m_20', 
        '101309_m_48', '071911_w_24', '102316_w_50', '100417_m_44', '083013_w_47', '083009_w_42', '080714_m_23', '101809_m_59', '082909_m_47', '101209_w_61', 
        '092014_m_56', '072414_m_23', '101015_w_43', '112909_w_20', '111609_m_65', '100117_w_36', '111409_w_63', '080709_m_24', '072714_m_23', '112914_w_51', 
        '120514_w_56', '083109_m_60', '110909_m_29', '091814_m_37', '071814_w_23', '092509_w_51', '112809_w_23', '100214_m_50', '102214_w_36', '082714_m_22', 
        '082109_m_53', '092808_m_51', '080309_m_29', '102008_w_22', '111914_w_63', '082809_m_26', '072514_m_27', '082814_w_46', '072609_w_23', '101216_m_40', 
        '091914_m_46', '100914_m_39', '112209_m_51', '092514_m_50', '092009_m_54', '082414_m_64', '080614_m_24']
        
        target_subject_list = ['081014_w_27','101609_m_36','112009_w_43','091809_w_43','071309_w_21','073114_m_25','080314_w_25','073109_w_28','100909_w_65','081609_w_40']

    elif args.current_ds is config.STRESS:
        '''
        StressID Random Source subjects
        '''
        source_list_name = ['9j3o','5f7t','9t6n','71i5','v8mh','bfl5','m8g5','45lx','tmvd','4woj',  # 10
                            'b2l8','j9h8','d4n6','9txq','g9j5','w2t5','2z7d','8g4y','6g6y','j1u8',  # 20
                            'c3m7','h7j3','chdf','qw5t','t6v9','a1k9','6k5f','i9t9','cxj0','r5s8',  # 30
                            '2ea4','2hpu','y8c3','kkf5','h8r2','iqyg','y9z6','f6q3','e5p4','k67g',  # 40
                            '8i4i', 'k2v7','4e8r','wssm']
        target_subject_list = ["kycf","uymz","h8s1","ctzy","p9i3","7h5u","g7r2","b9w0","r3zm","x1q3"]
    elif args.current_ds is config.BAH:
        '''
        BAH_DB Random Source subjects
        '''
        source_list_name = load_bah_src_subs(config.BAH_PATH_TRAIN)
        target_subject_list = ["82711", "82687", "82585", "82592", "82598", "82632", "82681", "82683", "82708", "82714"]

    print("===== Selected Conf Threshold: ", args.selection_p)

    file_name = 'logs/'+str(args.ctx_init)+'-au_topP='+str(args.au_topP)+'-stp='+str(args.tta_steps)+'-n_Ctx='+str(args.n_ctx)+'-landmarks='+str(args.use_landmarks)+'-sel_t='+str(args.selection_p)+'-num_land='+str(args.num_landmarks)+'.txt'


    if args.current_ds is config.BIOVID:
        args.srcs_file_name='lab_srcs77_biovid_ep10_bs8_sql16_str2_vid'
        args.pain_db_root_path = config.BIOVID_PATH
        args.test_sets = 'biosub0/biosub1/biosub2/biosub3/biosub4/biosub5/biosub6/biosub7/biosub8/biosub9'
        # args.test_sets = 'biosub0/biosub3/biosub6/biosub8'

        args.srcs_label_file_name= 'lab_srcs78_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
        args.srcs_label_val_file_name=None
        
    elif args.current_ds is config.STRESS:
        args.srcs_file_name='lab_srcs44_stress_ep20_bs8_sql16_str1_vid'
        args.pain_db_root_path = config.STRESS_PATH
        args.test_sets = 'stresssub0/stresssub1/stresssub2/stresssub3/stresssub4/stresssub5/stresssub6/stresssub7/stresssub8/stresssub9'
        args.srcs_label_file_name= 'stress_source_sub_labels'
        args.srcs_label_val_file_name= None
    elif args.current_ds is config.BAH:
        args.pain_db_root_path = config.BAH_DATASET_FRAMES_PATH
        args.test_sets = 'bahssub0/bahssub1/bahssub2/bahssub3/bahssub4/bahssub5/bahssub6/bahssub7/bahssub8/bahssub9'
        args.srcs_label_file_name= 'bah_source_sub_train_labels'
        args.srcs_label_val_file_name= 'bah_source_sub_val_labels'
    
    assert args.gpu is not None
    with experiment.train():
        experiment.log_parameter("arch", args.arch)
        experiment.log_parameter("Load train text adpt and classifier model", args.load_t_adpt_cl_mod)
        experiment.log_parameter("Train text adpt and classifier model", args.train_t_adpt_cl)
        experiment.log_parameter("Video seq length", args.seq_len)
        experiment.log_parameter("Video frame stride", args.frame_stride)
        experiment.log_parameter("Loss", 'CrossEntropyLoss')
        experiment.log_parameter("srcs_file_name", args.srcs_file_name)
        experiment.log_parameter("current_ds", args.current_ds)
        experiment.log_parameter("batch size", args.batch_size)
        experiment.log_parameter("source epochs", args.t_adap_epoch)
        experiment.log_parameter("seq length", args.seq_len)
        experiment.log_parameter("frame_stride", args.frame_stride)
        experiment.log_parameter("key_frame_sel", args.key_frame_sel)
        experiment.log_parameter("Key frames", args.key_frames)

        main_worker(args.gpu, args, file_name, source_list_name, target_subject_list)


def main_worker(gpu, args, file_name, source_list_name, target_subject_list):
    args.gpu = gpu
    set_random_seed(get_default_seed())
    print("Use GPU: {} for training".format(args.gpu))
    print("Train text adpt and classifier model: ", args.train_t_adpt_cl)
    print("Load train text adpt and classifier model: ", args.load_t_adpt_cl_mod)


    if args.current_ds is config.RAF_DB:
        classnames = raftrains_classes
    elif args.current_ds is config.STRESS:
        classnames = stresssub_classes 
    elif args.current_ds is config.BAH:
        classnames = bahssub_classes
    else:
        classnames = biosub_classes

    model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, 
                    num_aus=len(AU_PROMPTS), num_classes=len(classnames), au_prompts=AU_PROMPTS, is_video_clip=args.is_video_clip, frame_stride=args.frame_stride)

    for name, param in model.named_parameters():
        if args.train_t_adpt_cl:

            if any(k in name for k in ["text_adapter", "temporal_classifier", "temporal"]):
                param.requires_grad_(True)

        if args.adapt_tar_sub:
            if any(k in name for k in ["au_prompt_learner"]):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        if args.train_whole_clip_model:
            param.requires_grad_(True)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    print("=> Using TPT Augmentation {}".format(args.tpt))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    if args.train_t_adpt_cl:
        if not args.is_video_clip:
            trainable_params = (
                list(model.text_adapter.parameters()) +
                list(model.au_classifier.parameters())
            )
        else:
            trainable_params = (
                list(model.text_adapter.parameters()) +
                list(model.temporal.parameters()) +
                list(model.temporal_classifier.parameters()) 
            )
            print("[INFO] Training AU adapter, AU classifier, and temporal transformer (video mode).")
    elif args.adapt_tar_sub:
        if args.au_prompt_tune:
            trainable_params = model.au_prompt_learner.parameters()
            print("[INFO] Training Target subject-specific AU prompt tuning (personalization mode).")
        else:
            trainable_params = (
                list(model.text_adapter.parameters()) +
                list(model.temporal_classifier.parameters()) 
            )
            print("[INFO] Training Target with AU classifier (video mode).")
    elif args.train_whole_clip_model:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        print("[INFO] Training a While CLIP Model.")
    else:
        trainable_params = model.prompt_learner.parameters()
        print("[INFO] Training prompt learner only (image mode).")

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    '''
        Here Training AU Adapter and Classifier using Source Data
    '''
    if (args.train_t_adpt_cl or args.train_whole_clip_model) and args.current_ds is not config.RAF_DB:
        print(f"=== Source file name: {args.srcs_file_name}")
        srcs_loader, srcs_val_loader, srcs_test_loader = BaseDataset.load_pain_dataset(args.pain_db_root_path, 
                                args.srcs_label_file_name+('.csv' if args.current_ds is config.BAH else '.txt'), 
                                (args.srcs_label_val_file_name+'.csv' if args.current_ds is config.BAH else None), 
                                args.batch_size, BICUBIC, args.resolution, phase='src', 
                                seq_len=args.seq_len, frame_stride=args.frame_stride)
    source_model_path = os.path.join(config.WEIGHTS_FOLDER, args.current_ds, args.srcs_file_name+'.pth')

    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}

    for set_id in datasets:
        # comet create experiment name
        set_comet_exp_name(experiment, len(source_list_name), True, len(source_list_name), str())
        if not args.train_t_adpt_cl:
            target_file_path, target_weight_path, timestamp = create_target_folders(config.CURRENT_DIR, config.WEIGHTS_FOLDER, target_subject_list[int(set_id[-1])], args.top_timestamp if args.target_evaluation_only else None)

        if (args.tpt) and not args.use_landmarks:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size-1, 
                                            augmix=len(set_id)>1)
            batchsize = 1
            # batchsize = args.batch_size
        elif args.use_landmarks:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
            ])
            batchsize = args.batch_size
        else:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize])
            data_transform = AugMixAugmenter(base_transform, preprocess, n_views=0, 
                                            augmix=len(set_id)>1)
            if args.adapt_per_video:
                batchsize = 40
            else:
                batchsize = args.batch_size
        
        with open(file_name, "a") as f:  
            f.write("=> Model created: visual backbone {}".format(args.arch))
            f.write("\nPrompt: {}".format(args.ctx_init))
            f.write("\nSubject: {}".format(set_id))

        print("evaluating: {}".format(set_id))
        if 'bio' or 'stress' in set_id:
            sub_id = set_id[-1]
            set_id = set_id[:-1] 

        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = biosub_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        model.reset_classnames(classnames, args.arch)

        if args.is_video_clip:
            ''' Load Target data using file for Videos '''
            tar_sub_id = target_subject_list[int(sub_id)]
            tar_file_name = os.path.join(config.WEIGHTS_FOLDER, str(int(sub_id)+1)+'-'+tar_sub_id, 'files', 
                                tar_sub_id+('.csv' if args.current_ds is config.BAH else '.txt'))
            val_loader, _, _ = BaseDataset.load_pain_dataset(args.pain_db_root_path, tar_file_name, 
                            None, batchsize, BICUBIC, args.resolution, phase='tar', seq_len=args.seq_len, frame_stride=args.frame_stride)
        else:
            val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode, sub_id=sub_id if args.current_ds is config.BIOVID else '0', use_landmarks=args.use_landmarks, max_landmarks=args.num_landmarks)
            print("number of test samples: {}".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batchsize, shuffle=True,
                        num_workers=args.workers, pin_memory=True)     
        
        emoclip_model = None
        if args.load_t_adpt_cl_mod:
            saved_model = load_txt_adapter_classifier(source_model_path, model, args, device)
            print("[INFO] AU Adapter and Classifier Loaded: ", source_model_path)

            if args.eval_au_adpt_cl:
                evaluate_txt_adapter_n_au_classifier(args, saved_model, val_loader if args.current_ds is config.RAF_DB else srcs_val_loader)
                break
            elif args.eval_au_tar_sb:
                TSNE_ALL_SUB_LABLES = evaluate_txt_adapter_n_au_classifier(args, model, val_loader, sub_id, args.eval_au_tar_sb)
                continue
        # continue
        if args.train_t_adpt_cl or args.train_whole_clip_model:
            model = train_txt_adapter_n_au_classifier(args, model, emoclip_model, val_loader if args.current_ds is config.RAF_DB else srcs_loader, srcs_val_loader,
                                            scaler, optimizer, optim_state, classnames, source_model_path, num_epochs=args.t_adap_epoch, 
                                            save_path=source_model_path, train_clip_model=args.train_whole_clip_model)
            evaluate_txt_adapter_n_au_classifier(args, model, val_loader if args.current_ds is config.RAF_DB else srcs_val_loader)
            break
        if args.adapt_tar_sub:
            target_path = os.path.join(target_weight_path, 'adapt_model.pth')
            model = train_txt_adapter_n_au_classifier(args, model, emoclip_model, val_loader, None, scaler, optimizer, optim_state, classnames, source_model_path, 
                                    num_epochs=args.t_adap_epoch, save_path=target_path, train_clip_model=args.train_whole_clip_model)
            continue


def train_txt_adapter_n_au_classifier(args, model, emoclip_model, train_loader, val_loader, scaler, optimizer, optim_state, classnames, source_model_path,
                                      num_epochs=10, save_path="clip_au_model_vit32_au46.pth", train_clip_model=False):
    optimizer.load_state_dict(optim_state)

    if args.current_ds is config.BAH:
        class_prompt = CLASS_PROMPTS_AMBV
    else:
        class_prompt = CLASS_PROMPTS

    criterion = nn.CrossEntropyLoss()
    if args.adapt_tar_sub:
        model = load_txt_adapter_classifier(source_model_path, model, args, device)
        optimizer = rebuild_optimizer(model, args.lr)

    # count_trainable_params(model)
    times = []

    save_dict = {}
    best_acc, best_f1 = 0, 0
    best_model = deepcopy(model.state_dict())
    logger = MetricLogger(save_dir="metrics", filename="comparison_tran.csv")

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_samples, itera = 0, 0

        all_labels = []
        all_preds = []

        class_weights = [(epoch*0.05) + 0.05, 1 - ((epoch*0.05) + 0.05)]
        class_weights = None

        for images, labels, video_path in tqdm(train_loader):
            if args.adapt_per_video:
                model = load_txt_adapter_classifier(source_model_path, model, args, device)
                optimizer = rebuild_optimizer(model, args.lr)
            images = images.cuda(args.gpu, non_blocking=True) if args.current_ds is not config.RAF_DB else images[0].cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

            with torch.cuda.amp.autocast():
                if args.adapt_tar_sub:
                    logits, au_sim = model(images, AU_PROMPTS, class_prompt if args.include_cls_prompt else None, mode=("temporal" if args.is_video_clip else "au"), 
                                adapt_target=args.adapt_tar_sub, key_frame_sel=args.key_frame_sel, train_whole_clip=train_clip_model, key_frames=args.key_frames)   # forward pass

                    loss = entropy_loss(logits, class_weights)
                else:
                    start = time.time()
                    logits, au_sim = model(images, AU_PROMPTS, CLASS_PROMPTS if args.include_cls_prompt else None, mode=("temporal" if args.is_video_clip else "au"), 
                                adapt_target=args.adapt_tar_sub, train_whole_clip=train_clip_model)   # forward pass
                    loss = criterion(logits, labels)     # loss, DO NOT .item() here

                    if train_clip_model:
                        loss_t = criterion(logits, labels)
                        loss = (loss + loss_t) / 2
                
            itera = itera + 1
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize() 
            end = time.time()
            times.append(end - start)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)

            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())

            if itera == args.iter_limit:
                break

        # end of epoch metrics
        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)

        avg_time = sum(times) / len(times)
        print(f"Average per-batch time: {avg_time:.4f} sec")

        epoch_loss = running_loss / total_samples

        epoch_acc  = accuracy_score(all_labels, all_preds)
        epoch_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_uar  = recall_score(all_labels, all_preds, average='macro', zero_division=0)  # UAR

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f}  "
              f"WAR: {epoch_acc:.4f}  "
              f"UAR: {epoch_uar:.4f} "
              f"F1: {epoch_f1:.4f}  ")

        experiment.log_metric("Loss:", epoch_loss, epoch=epoch)
        experiment.log_metric("WAR:", epoch_acc, epoch=epoch)
        experiment.log_metric("UAR:", epoch_uar, epoch=epoch)
        experiment.log_metric("WAF1R:", epoch_f1, epoch=epoch)

        if best_acc < epoch_acc or best_f1 < epoch_f1:
            if best_acc < epoch_acc: 
                best_acc = epoch_acc
            if best_f1 < epoch_f1: 
                best_f1 = epoch_f1
            best_model = deepcopy(model)
            if args.is_video_clip:
                if train_clip_model:
                    torch.save(best_model.state_dict(), save_path)
                    print(f"[INFO] Saving Complete CLIP model")
                elif args.adapt_tar_sub:
                    print("[INFO] SKIPPING Not saving the weights")
                    continue
                    # if using the new temporal transformer
                    if hasattr(model, "temporal_classifier"):
                        save_dict["temporal_classifier"] = model.temporal_classifier.state_dict()
                    print(f"[INFO] Saving Target video model Subject-Specific Adapter")
                    torch.save(save_dict, save_path)
                else:
                    save_dict["text_adapter"] = model.text_adapter.state_dict()

                    # if using the new temporal transformer
                    if hasattr(model, "temporal"):
                        save_dict["temporal"] = model.temporal.state_dict()
                    if hasattr(model, "temporal_proj"):
                        save_dict["temporal_proj"] = model.temporal_proj.state_dict()
                    if hasattr(model, "temporal_classifier"):
                        save_dict["temporal_classifier"] = model.temporal_classifier.state_dict()
                    
                    print(f"[INFO] Saving video model components: {list(save_dict.keys())}")
                    torch.save(save_dict, save_path)
            else:
                # Save only the trainable parts
                best_model = deepcopy(model)
                torch.save({
                    'text_adapter': model.text_adapter.state_dict(),
                    'au_classifier': model.au_classifier.state_dict()
                }, save_path)
                print(f"Model saved to {save_path}")
    
    return best_model

def count_trainable_params(model):

    for name, module in [
        ("AU Adapter", model.text_adapter),
        ("Temporal (1D-CNN + GLU)", model.temporal),
        ("Emotion Classifier", model.temporal_classifier)
    ]:
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {params/1e6:.3f} M params")

    dummy_text = torch.randn(46, 512).to(device)
    flops_text, _ = profile(model.text_adapter, inputs=(dummy_text,), verbose=False)

    dummy_text = torch.randn(8, 512, 16).to(device)
    flops_temp, _ = profile(model.temporal, inputs=(dummy_text,), verbose=False)

    dummy_video = torch.randn(8, 46).to(device)
    flops, _ = profile(model.temporal_classifier, inputs=(dummy_video,), verbose=False)

    print(f"Text Adapter GFLOPs per batch: {flops_text / 1e9:.9f}")
    print(f"Temporal GFLOPs per batch: {flops_temp / 1e9:.9f}")
    print(f"Temporal_classifier GFLOPs per batch: {flops / 1e9:.9f}")


def evaluate_txt_adapter_n_au_classifier(args, model, data_loader, sub_id, eval_tar=False):
    model.eval()
    all_preds, all_labels = [], []
    itr = 0
    with torch.no_grad():
        for images, labels, frame_paths in tqdm(data_loader):
            if args.tpt and not is_video_clip:
                images = images[0].cuda(args.gpu, non_blocking=True)  
            else:
                images = images.cuda(args.gpu, non_blocking=True) 
            labels = labels.cuda(args.gpu, non_blocking=True)

            logits, _ = model(images, au_prompts=AU_PROMPTS, class_prompts=CLASS_PROMPTS if args.include_cls_prompt else None,
                    mode=("temporal" if args.is_video_clip else "au"))
            pred_prob = F.softmax(logits, dim=-1)
            preds = torch.argmax(pred_prob, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())


    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc  = accuracy_score(all_labels, all_preds) * 100.0
    f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100.0
    uar  = recall_score(all_labels, all_preds, average='macro', zero_division=0)  * 100.0

    print(f"Eval — WAR: {acc:.3f}  UAR: {uar:.3f}  F1(macro): {f1:.3f} ")

    return {'acc': acc, 'uar': uar, 'f1_macro': f1 }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', metavar='DIR', help='path to dataset root', default=config.BIOVID_SOURCE_DATASET_PATH)# BIOVID_SOURCE_DATASET_PATH or RAF_DATASET_CLS_PATH
    
    # Stress
    parser.add_argument('--test_sets', type=str, default='stresssub0/stresssub1/stresssub2/stresssub3/stresssub4/stresssub5/stresssub6/stresssub7/stresssub8/stresssub9', help='test dataset (multiple datasets split by slash)')
    
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/32')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N')
    parser.add_argument('-tar_b', '--tar_batch-size', default=32, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=500, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.5, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=10, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')

    parser.add_argument('--au_topP', type=int, default=8, help='Top-P AUs per class (by AU-class text similarity)')
    parser.add_argument('--dump_au_class_sim', action='store_true', default=True, help='Save AU-class similarity CSV')

    parser.add_argument('--ctx_init', default='a_person_with_an_expression_of_[CLS]', type=str, help='init tunable prompts')
    
    parser.add_argument('--use_landmarks', type=bool, default=False, help="to extract faical landmarks")
    parser.add_argument('--num_landmarks', type=int, default=12)
    parser.add_argument('--visualize_img', default=False, help='visualize_img')

    parser.add_argument('--train_whole_clip_model', type=bool, default=False, help="Train complete Clip model using au PROMPTS")

    parser.add_argument('--train_t_adpt_cl', type=bool, default=True, help="train_txt_adapt_classifer")
    parser.add_argument('--t_adap_epoch', default=10, type=int, help="Text and AU classifier training epochs")
    parser.add_argument('--iter_limit', default=10, type=int, help='Limit loop')

    parser.add_argument('--load_t_adpt_cl_mod', type=bool, default=False, help="load_t_adpt_cl_mod")
    parser.add_argument('--eval_au_adpt_cl', type=bool, default=False, help="eval_au_adpt_cl")
    parser.add_argument('--eval_au_tar_sb', type=bool, default=False, help="eval_au_tar_sb")
    
    parser.add_argument('--include_cls_prompt', type=bool, default=False, help="include_cls_prompt")

    # =========================================================== Target Subject Video Adaptation
    parser.add_argument('--adapt_tar_sub', type=bool, default=False, help="adapt_tar_sub")
    parser.add_argument('--adapt_per_video', type=bool, default=False, help="adapt_per_video")
    parser.add_argument('--au_prompt_tune', type=bool, default=False, help="au_prompt_tune")
    parser.add_argument('--key_frame_sel', type=bool, default=False, help="key_frame_sel")
    parser.add_argument('--key_frames', type=int, default=16, help="key_frame_sel")

    parser.add_argument('--current_ds', type=str, default=config.BIOVID)  # BIOVID or STRESS or BAH
    parser.add_argument('--pain_db_root_path', type=str, default=config.STRESS_PATH) # BIOVID_PATH or STRESS_PATH or 

    parser.add_argument('--srcs_label_file_name', default='bah_source_sub_labels', type=str, help='BAH Source file name to train AU adapter and classifier')
    parser.add_argument('--srcs_file_name', default='stress_src_ep10_bs8_sql16_str1_vid_46mixclsprompts', type=str, help='Source file name to train AU adapter and classifier')
    parser.add_argument('--srcs_label_val_file_name', default='bah_source_sub_val_labels', type=str, help='Source file name to val AU adapter and classifier')
    
    # =========================================================== Video
    parser.add_argument('--is_video_clip', type=bool, default=True, help="Train Clip on videos")
    parser.add_argument('--seq_len', default=20, type=int, help="Video / Sequence Length")
    parser.add_argument('--frame_stride', default=2, type=int, help="Frame stride")

    # ===========================================================
    parser.add_argument('--target_evaluation_only', type=bool, default=False)
    parser.add_argument('--top_timestamp', type=str, default='1754805485')

    main()