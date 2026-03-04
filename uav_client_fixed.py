#!/usr/bin/env python3
import asyncio 
import argparse 
import json 
import os 
import sys 
import time 
import traceback 
import math 
import random 
from typing import Dict ,Any ,Optional ,List ,Tuple ,cast ,Callable 
import aiohttp 
import pandas as pd 
import numpy as np 
import torch 
import torch .nn as nn 
import torch .optim as optim 
import torch .nn .functional as F 
from torch .utils .data import DataLoader ,TensorDataset ,WeightedRandomSampler 
from sklearn .metrics import accuracy_score ,f1_score ,classification_report ,confusion_matrix 
from sklearn .preprocessing import StandardScaler ,LabelEncoder ,RobustScaler 
import warnings 
warnings .filterwarnings ('ignore')
if not os .environ .get ("PYTORCH_ALLOC_CONF"):
    old_value =os .environ .get ("PYTORCH_CUDA_ALLOC_CONF")
    if old_value :
        os .environ ["PYTORCH_ALLOC_CONF"]=old_value 
        del os .environ ["PYTORCH_CUDA_ALLOC_CONF"]
    else :
        os .environ ["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
sys .path .append (os .path .dirname (os .path .abspath (__file__ )))
import config_fixed as config 
BASELINE_MODE =bool (getattr (config ,"BASELINE_MODE",False ))
_GLOBAL_PROTOTYPES_CACHE :Dict [str ,Any ]={"loaded":False ,"tensor":None ,"support":None }
_GAN_FILTER_MODEL_CACHE :Dict [str ,Any ]={"path":None ,"input_dim":None ,"num_classes":None ,"model":None }
def _parse_malicious_clients (raw :str )->List [int ]:
    if not raw :
        return []
    parts =[]
    for token in raw .split (","):
        token =token .strip ()
        if not token :
            continue 
        try :
            parts .append (int (token ))
        except Exception :
            pass 
    return parts 
def _get_malicious_set (total_clients :int )->List [int ]:
    attack_cfg =getattr (config ,"ATTACK_CONFIG",{})or {}
    if not attack_cfg .get ("enabled",False ):
        return []
    explicit =_parse_malicious_clients (attack_cfg .get ("malicious_clients",""))
    if explicit :
        return sorted (set (explicit ))
    ratio =float (attack_cfg .get ("malicious_ratio",0.0 ))
    if ratio <=0.0 :
        return []
    k =max (1 ,int (round (total_clients *ratio )))
    seed =int (attack_cfg .get ("seed",42 ))
    rng =random .Random (seed )
    return sorted (rng .sample (range (total_clients ),min (k ,total_clients )))
def _is_malicious_client (client_id :int )->bool :
    attack_cfg =getattr (config ,"ATTACK_CONFIG",{})or {}
    if not attack_cfg .get ("enabled",False ):
        return False 
    total_clients =int (getattr (config ,"NUM_CLIENTS",0 )or 0 )
    if total_clients <=0 :
        return False 
    return client_id in _get_malicious_set (total_clients )
def _resolve_label_value (label_value ,label_series :pd .Series ):
    unique_vals =set (label_series .unique ().tolist ())
    if label_value in unique_vals :
        return label_value 
    try :
        as_int =int (label_value )
        if as_int in unique_vals :
            return as_int 
    except Exception :
        pass 
    if isinstance (label_value ,str ):
        mapped =config .REVERSE_LABEL_MAPPING .get (label_value )
        if mapped in unique_vals :
            return mapped 
    try :
        as_int =int (label_value )
        mapped_name =config .LABEL_MAPPING .get (as_int )
        if mapped_name in unique_vals :
            return mapped_name 
    except Exception :
        pass 
    return None 
def _apply_label_flipping (train_df :pd .DataFrame ,client_id :int )->Tuple [pd .DataFrame ,int ]:
    attack_cfg =getattr (config ,"ATTACK_CONFIG",{})or {}
    lf_cfg =attack_cfg .get ("label_flipping",{})if attack_cfg else {}
    if not (attack_cfg .get ("enabled",False )and lf_cfg .get ("enabled",False )):
        return train_df ,0 
    if not _is_malicious_client (client_id ):
        return train_df ,0 
    label_col =getattr (config ,"LABEL_COL","label")
    if label_col not in train_df .columns :
        return train_df ,0 
    src_label =lf_cfg .get ("source_label","DDoS")
    tgt_label =lf_cfg .get ("target_label","BENIGN")
    label_series =cast (pd .Series ,train_df [label_col ])
    src_val =_resolve_label_value (src_label ,label_series )
    tgt_val =_resolve_label_value (tgt_label ,label_series )
    if src_val is None or tgt_val is None :
        print (f"[Client {client_id }] 標籤翻轉失敗：找不到標籤 {src_label } 或 {tgt_label }")
        return train_df ,0 
    flip_count =int ((train_df [label_col ]==src_val ).sum ())
    train_df .loc [train_df [label_col ]==src_val ,label_col ]=tgt_val 
    print (f"[Client {client_id }] 標籤翻轉已套用: {src_label } -> {tgt_label }, 影響 {flip_count } 筆")
    return train_df ,flip_count 
def _get_gan_aug_config ()->Dict [str ,Any ]:
    cfg =dict (getattr (config ,"GAN_AUGMENTATION_CONFIG",{})or {})
    enabled_env =os .environ .get ("GAN_AUG_ENABLED")
    if enabled_env is not None :
        cfg ["enabled"]=enabled_env .strip ().lower ()in ("1","true","yes","on")
    ratio_env =os .environ .get ("GAN_AUG_RATIO")
    if ratio_env is not None :
        cfg ["ratio"]=ratio_env .strip ()
    target_env =os .environ .get ("GAN_AUG_TARGET_LABEL")
    if target_env is not None :
        try :
            cfg ["target_label"]=int (target_env )
        except Exception :
            cfg ["target_label"]=target_env 
    min_ratio_env =os .environ .get ("GAN_AUG_MIN_CLASS_RATIO")
    if min_ratio_env is not None :
        try :
            cfg ["min_class_ratio"]=float (min_ratio_env )
        except Exception :
            pass 
    target_ratio_env =os .environ .get ("GAN_AUG_TARGET_RATIO")
    if target_ratio_env is not None :
        try :
            cfg ["target_class_ratio"]=float (target_ratio_env )
        except Exception :
            pass 
    max_new_env =os .environ .get ("GAN_AUG_MAX_NEW_SAMPLES")
    if max_new_env is not None :
        try :
            cfg ["max_new_samples"]=int (max_new_env )
        except Exception :
            pass 
    latent_env =os .environ .get ("GAN_AUG_LATENT_DIM")
    if latent_env is not None :
        try :
            cfg ["latent_dim"]=int (latent_env )
        except Exception :
            pass 
    gen_path_env =os .environ .get ("GAN_GENERATOR_PATH")
    if gen_path_env is not None :
        cfg ["generator_path"]=gen_path_env .strip ()
    gen_type_env =os .environ .get ("GAN_GENERATOR_TYPE")
    if gen_type_env is not None :
        cfg ["generator_type"]=gen_type_env .strip ().lower ()
    num_classes_env =os .environ .get ("GAN_NUM_CLASSES")
    if num_classes_env is not None :
        try :
            cfg ["num_classes"]=int (num_classes_env )
        except Exception :
            pass 
    device_env =os .environ .get ("GAN_AUG_DEVICE")
    if device_env is not None :
        cfg ["device"]=device_env .strip ().lower ()
    filter_enabled_env =os .environ .get ("GAN_AUG_FILTER_ENABLED")
    if filter_enabled_env is not None :
        cfg ["filter_enabled"]=filter_enabled_env .strip ().lower ()in ("1","true","yes","on")
    filter_min_env =os .environ .get ("GAN_AUG_FILTER_MIN_CONF")
    if filter_min_env is not None :
        try :
            cfg ["filter_min_conf"]=float (filter_min_env )
        except Exception :
            pass 
    filter_max_env =os .environ .get ("GAN_AUG_FILTER_MAX_CONF")
    if filter_max_env is not None :
        try :
            cfg ["filter_max_conf"]=float (filter_max_env )
        except Exception :
            pass 
    filter_model_env =os .environ .get ("GAN_AUG_FILTER_MODEL_PATH")
    if filter_model_env is not None :
        cfg ["filter_model_path"]=filter_model_env .strip ()
    return cfg 
class _WGANGenerator (nn .Module ):
    def __init__ (self ,latent_dim :int ,output_dim :int ,hidden_dims :Optional [List [int ]]=None ):
        super ().__init__ ()
        if hidden_dims is None :
            hidden_dims =[128 ,128 ]
        layers :List [nn .Module ]=[]
        in_dim =latent_dim 
        for h in hidden_dims :
            layers .append (nn .Linear (in_dim ,h ))
            layers .append (nn .ReLU (inplace =True ))
            in_dim =h 
        layers .append (nn .Linear (in_dim ,output_dim ))
        self .net =nn .Sequential (*layers )
    def forward (self ,z :torch .Tensor )->torch .Tensor :
        return self .net (z )
class _ConditionalWGANGenerator (nn .Module ):
    def __init__ (self ,latent_dim :int ,output_dim :int ,num_classes :int ,hidden_dims :Optional [List [int ]]=None ):
        super ().__init__ ()
        if hidden_dims is None :
            hidden_dims =[128 ,128 ]
        self .label_emb =nn .Embedding (num_classes ,num_classes )
        in_dim =latent_dim +num_classes 
        layers :List [nn .Module ]=[]
        for h in hidden_dims :
            layers .append (nn .Linear (in_dim ,h ))
            layers .append (nn .ReLU (inplace =True ))
            in_dim =h 
        layers .append (nn .Linear (in_dim ,output_dim ))
        self .net =nn .Sequential (*layers )
    def forward (self ,z :torch .Tensor ,labels :torch .Tensor )->torch .Tensor :
        label_vec =self .label_emb (labels )
        x =torch .cat ([z ,label_vec ],dim =1 )
        return self .net (x )
def _load_gan_generator (path :str ,latent_dim :int ,output_dim :int ,device :torch .device ,generator_type :str ="wgan",num_classes :int =0 )->Optional [nn .Module ]:
    if not path or not os .path .exists (path ):
        return None 
    if generator_type =="conditional"and num_classes >0 :
        gen =_ConditionalWGANGenerator (latent_dim =latent_dim ,output_dim =output_dim ,num_classes =num_classes )
    else :
        gen =_WGANGenerator (latent_dim =latent_dim ,output_dim =output_dim )
    state =torch .load (path ,map_location =device )
    if isinstance (state ,dict )and "state_dict"in state :
        state =state ["state_dict"]
    if isinstance (state ,dict ):
        cleaned ={k .replace ("module.",""):v for k ,v in state .items ()}
        gen .load_state_dict (cleaned ,strict =False )
    gen .to (device )
    gen .eval ()
    return gen 
def _apply_gan_augmentation (train_df :pd .DataFrame ,label_col :str ,feature_cols :List [str ],client_id :int )->pd .DataFrame :
    gan_cfg =_get_gan_aug_config ()
    if not gan_cfg .get ("enabled",False ):
        return train_df 
    if label_col not in train_df .columns :
        return train_df 
    target_label_cfg =gan_cfg .get ("target_label",1 )
    if isinstance (target_label_cfg ,str )and target_label_cfg .strip ().lower ()in ("auto_minority","auto-minority"):
        label_counts =train_df [label_col ].value_counts ()
        all_labels =list (getattr (config ,"LABEL_MAPPING",{}).keys ())
        counts :Dict [Any ,int ]={}
        if all_labels and all (isinstance (x ,type (all_labels [0 ]))for x in all_labels ):
            for lbl in all_labels :
                raw_value =label_counts .get (lbl ,0 )
                if raw_value is None :
                    raw_value =0
                counts [lbl ]=int (raw_value )
        else :
            for lbl ,cnt in label_counts .items ():
                counts [lbl ]=int (cnt )
        if not counts :
            return train_df 
        resolved_target =sorted (counts .items (),key =lambda kv :(kv [1 ],kv [0 ]))[0 ][0 ]
        print (f"[Client {client_id }] GAN 自動選擇最稀少類別: {resolved_target } (count={counts [resolved_target ]})")
    else :
        label_series =cast (pd .Series ,train_df [label_col ])
        resolved_target =_resolve_label_value (target_label_cfg ,label_series )
        if resolved_target is None :
            print (f"[Client {client_id }] GAN 增強略過：找不到目標類別 {target_label_cfg }")
            return train_df 
    total =len (train_df )
    if total <=0 :
        return train_df 
    target_count =int ((train_df [label_col ]==resolved_target ).sum ())
    min_ratio =float (gan_cfg .get ("min_class_ratio",0.0 )or 0.0 )
    if min_ratio >0 and target_count /max (total ,1 )>=min_ratio :
        print (f"[Client {client_id }] GAN 增強略過：目標類別比例已達 {min_ratio :.2%}")
        return train_df 
    ratio_cfg =gan_cfg .get ("ratio",0.0 )
    num_new =0 
    if isinstance (ratio_cfg ,str )and ratio_cfg .strip ().lower ()=="auto":
        target_ratio =float (gan_cfg .get ("target_class_ratio",0.1 )or 0.0 )
        if target_ratio <=0.0 or target_ratio >=1.0 :
            print (f"[Client {client_id }] GAN 增強略過：target_class_ratio 無效 ({target_ratio })")
            return train_df 
        current_ratio =target_count /max (total ,1 )
        if current_ratio >=target_ratio :
            print (f"[Client {client_id }] GAN 增強略過：目標類別比例已達 {target_ratio :.2%}")
            return train_df 
        required =(target_ratio *total -target_count )/(1.0 -target_ratio )
        num_new =max (0 ,int (math .ceil (required )))
    else :
        try :
            ratio =float (ratio_cfg )
        except Exception :
            ratio =0.0 
        num_new =max (0 ,int (total *ratio ))
    if num_new <=0 :
        return train_df 
    max_new =int (gan_cfg .get ("max_new_samples",0 )or 0 )
    if max_new >0 :
        num_new =min (num_new ,max_new )
    gen_path =str (gan_cfg .get ("generator_path","")or "").strip ()
    device_cfg =str (gan_cfg .get ("device","auto")).lower ()
    force_cpu =os .environ .get ("FORCE_CPU","0").strip ().lower ()in ("1","true","yes","on")
    if force_cpu :
        device =torch .device ("cpu")
    elif device_cfg =="auto":
        device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
    else :
        device =torch .device (device_cfg )
    latent_dim =int (gan_cfg .get ("latent_dim",32 )or 32 )
    generator_type =str (gan_cfg .get ("generator_type","wgan")or "wgan").lower ()
    num_classes =int (gan_cfg .get ("num_classes",len (getattr (config ,"LABEL_MAPPING",{}))or 0 )or 0 )
    generator =_load_gan_generator (gen_path ,latent_dim =latent_dim ,output_dim =len (feature_cols ),device =device ,generator_type =generator_type ,num_classes =num_classes )
    if generator is None :
        print (f"[Client {client_id }] GAN 增強略過：找不到生成器權重 {gen_path }")
        return train_df 
    with torch .no_grad ():
        z =torch .randn (num_new ,latent_dim ,device =device )
        if generator_type =="conditional"and num_classes >0 :
            label_tensor =torch .full ((num_new ,),int (resolved_target ),device =device ,dtype =torch .long )
            synth =generator (z ,label_tensor ).cpu ().numpy ()
        else :
            synth =generator (z ).cpu ().numpy ()
    columns_index =pd .Index (feature_cols )
    synth_df =pd .DataFrame (synth ,columns =columns_index )
    synth_df [label_col ]=resolved_target 
    filter_enabled =bool (gan_cfg .get ("filter_enabled",False ))
    if filter_enabled :
        filter_min =float (gan_cfg .get ("filter_min_conf",0.7 ))
        filter_max =float (gan_cfg .get ("filter_max_conf",0.9 ))
        filter_model_path =str (gan_cfg .get ("filter_model_path","")or "").strip ()
        filter_device =torch .device ("cpu")
        filter_model =_load_gan_filter_model (filter_model_path ,len (feature_cols ),int (gan_cfg .get ("num_classes",0 )or 0 ),filter_device )
        if filter_model is None :
            print (f"[Client {client_id }] GAN 過濾跳過：找不到/無法載入模型 {filter_model_path }")
        else :
            try :
                feats =torch .tensor (synth_df [feature_cols ].values ,dtype =torch .float32 ,device =filter_device )
                with torch .no_grad ():
                    logits =filter_model (feats )
                    logits =_extract_logits (logits )
                    probs =torch .softmax (logits ,dim =-1 )
                    conf =probs .max (dim =-1 ).values .cpu ().numpy ()
                keep_mask =(conf >=filter_min )&(conf <=filter_max )
                kept =int (keep_mask .sum ())
                if kept <=0 :
                    print (f"[Client {client_id }] GAN 過濾後無樣本保留（min={filter_min }, max={filter_max }），跳過增強")
                    return train_df 
                synth_df =synth_df .loc [keep_mask ].reset_index (drop =True )
                num_new =kept 
                print (f"[Client {client_id }] GAN 過濾完成：保留 {kept }/{len (conf )} 筆（min={filter_min }, max={filter_max }）")
            except Exception as e :
                print (f"[Client {client_id }] GAN 過濾失敗，改用未過濾樣本: {e }")
    train_df =pd .concat ([train_df ,synth_df ],ignore_index =True )
    train_df =train_df .sample (frac =1 ,random_state =42 +client_id ).reset_index (drop =True )
    print (f"[Client {client_id }] GAN 增強完成：新增 {num_new } 筆（label={resolved_target }）")
    return train_df 
def _poison_embedding (embedding :torch .Tensor ,client_id :int )->torch .Tensor :
    attack_cfg =getattr (config ,"ATTACK_CONFIG",{})or {}
    mp_cfg =attack_cfg .get ("model_poisoning",{})if attack_cfg else {}
    if not (attack_cfg .get ("enabled",False )and mp_cfg .get ("enabled",False )):
        return embedding 
    if not _is_malicious_client (client_id ):
        return embedding 
    method =str (mp_cfg .get ("method","gaussian")).lower ()
    sigma =float (mp_cfg .get ("sigma",0.5 ))
    replace_prob =float (mp_cfg .get ("replace_prob",1.0 ))
    if random .random ()>max (0.0 ,min (1.0 ,replace_prob )):
        return embedding 
    if method =="random":
        poisoned =torch .randn_like (embedding )
    else :
        noise =torch .randn_like (embedding )*sigma 
        poisoned =embedding +noise 
    print (f"[Client {client_id }] 模型中毒已套用: method={method }, sigma={sigma }")
    return poisoned 
def _apply_gradient_noise (model :nn .Module ,epoch_num :int ,batch_idx :int ,device :torch .device )->None :
    cfg =getattr (config ,"GRAD_NOISE_CONFIG",{})or {}
    if not cfg .get ("enabled",False ):
        return 
    base_sigma =float (cfg .get ("base_sigma",0.01 ))
    decay =float (cfg .get ("decay",0.0 ))
    min_sigma =float (cfg .get ("min_sigma",0.0 ))
    max_sigma =float (cfg .get ("max_sigma",max (base_sigma ,min_sigma )))
    mode =str (cfg .get ("mode","fixed")).lower ()
    ref_grad_norm =float (cfg .get ("ref_grad_norm",1.0 ))
    sigma =base_sigma 
    if decay >0 :
        sigma =max (min_sigma ,base_sigma /(1.0 +decay *max (0 ,epoch_num )))
    if mode =="adaptive":
        total_norm =0.0 
        for p in model .parameters ():
            if p .grad is None :
                continue 
            param_norm =p .grad .data .norm (2 ).item ()
            total_norm +=param_norm **2 
        grad_norm =total_norm **0.5 
        if ref_grad_norm >0 :
            sigma =sigma *(grad_norm /ref_grad_norm )
    if sigma <=0 :
        return 
    sigma =max (min_sigma ,min (max_sigma ,sigma ))
    for p in model .parameters ():
        if p .grad is None :
            continue 
        noise =torch .randn_like (p .grad ,device =device )*sigma 
        p .grad .add_ (noise )
    log_interval =int (cfg .get ("log_interval",50 ))
    if log_interval >0 and batch_idx %log_interval ==0 :
        print (f" 梯度噪聲已套用: sigma={sigma :.6f}, mode={mode }")
def _write_attack_metadata (result_dir :str ,client_id :int )->None :
    attack_cfg =getattr (config ,"ATTACK_CONFIG",{})or {}
    if not attack_cfg .get ("enabled",False ):
        return 
    if not result_dir :
        return 
    try :
        os .makedirs (result_dir ,exist_ok =True )
    except Exception :
        return 
    meta_path =os .path .join (result_dir ,"attack_metadata.json")
    if os .path .exists (meta_path ):
        return 
    try :
        total_clients =int (getattr (config ,"NUM_CLIENTS",0 )or 0 )
        malicious_clients =_get_malicious_set (total_clients )if total_clients >0 else []
        payload ={"timestamp":time .strftime ("%Y-%m-%d %H:%M:%S"),"malicious_ratio":float (attack_cfg .get ("malicious_ratio",0.0 )),"malicious_clients":malicious_clients ,"this_client_id":int (client_id ),"this_client_malicious":int (client_id )in set (malicious_clients ),"label_flipping":attack_cfg .get ("label_flipping",{}),"model_poisoning":attack_cfg .get ("model_poisoning",{}),"seed":int (attack_cfg .get ("seed",42 ))}
        with open (meta_path ,"w",encoding ="utf-8")as f :
            json .dump (payload ,f ,ensure_ascii =False ,indent =2 )
    except Exception :
        pass 
MODEL_HEAD_PREFIXES =tuple (getattr (config ,"MODEL_HEAD_PREFIXES",["output_layer"]))
def _is_head_layer (name :str )->bool :
    return any (name .startswith (prefix )for prefix in MODEL_HEAD_PREFIXES )
def _split_state_dict (state_dict :Dict [str ,torch .Tensor ])->Tuple [Dict [str ,torch .Tensor ],Dict [str ,torch .Tensor ]]:
    base ,head ={},{}
    for key ,value in state_dict .items ():
        if _is_head_layer (key ):
            head [key ]=value 
        else :
            base [key ]=value 
    return base ,head 
def _detect_architecture_from_weights (weights :Dict [str ,torch .Tensor ])->str :
    if not weights :
        return 'unknown'
    weight_keys =list (weights .keys ())
    transformer_features =['transformer_blocks','input_projection','positional_encoding','classifier.0.weight',]
    has_transformer =any (any (feat in key for feat in transformer_features )for key in weight_keys )
    dnn_features =['layers.0.weight','residual_layers','batch_norms','output_layer.weight',]
    has_dnn =any (any (feat in key for feat in dnn_features )for key in weight_keys )
    cnn_features =['conv','pool',]
    has_cnn =any (any (feat in key for feat in cnn_features )for key in weight_keys )
    if has_transformer :
        return 'transformer'
    elif has_dnn :
        return 'dnn'
    elif has_cnn :
        return 'cnn'
    else :
        return 'unknown'
def _detect_model_architecture (model :nn .Module )->str :
    model_keys =list (model .state_dict ().keys ())
    transformer_features =['transformer_blocks','input_projection','positional_encoding',]
    has_transformer =any (any (feat in key for feat in transformer_features )for key in model_keys )
    dnn_features =['layers.0.weight','residual_layers','batch_norms','output_layer.weight',]
    has_dnn =any (any (feat in key for feat in dnn_features )for key in model_keys )
    cnn_features =['conv','pool',]
    has_cnn =any (any (feat in key for feat in cnn_features )for key in model_keys )
    if has_transformer :
        return 'transformer'
    elif has_dnn :
        return 'dnn'
    elif has_cnn :
        return 'cnn'
    else :
        return 'unknown'
def _validate_architecture_compatibility (model :nn .Module ,weights :Dict [str ,torch .Tensor ],client_id :int ,raise_on_mismatch :bool =True )->bool :
    if not weights :
        return True 
    model_arch =_detect_model_architecture (model )
    weight_arch =_detect_architecture_from_weights (weights )
    if model_arch =='unknown'or weight_arch =='unknown':
        model_keys =set (model .state_dict ().keys ())
        weight_keys =set (weights .keys ())
        model_keys_filtered ={k for k in model_keys if not any (bn_key in k for bn_key in ['running_mean','running_var','num_batches_tracked'])and not k .startswith ('_teacher_model')}
        weight_keys_filtered ={k for k in weight_keys if not any (bn_key in k for bn_key in ['running_mean','running_var','num_batches_tracked'])and not k .startswith ('_teacher_model')}
        common_keys =model_keys_filtered &weight_keys_filtered 
        match_ratio =len (common_keys )/max (len (model_keys_filtered ),len (weight_keys_filtered ))if max (len (model_keys_filtered ),len (weight_keys_filtered ))>0 else 0.0 
        if match_ratio <0.3 :
            error_msg =(f"客戶端 {client_id } 模型架構與全局權重架構不匹配："f"模型鍵數={len (model_keys_filtered )}, 權重鍵數={len (weight_keys_filtered )}, "f"匹配鍵數={len (common_keys )}, 匹配率={match_ratio :.2%}")
            if raise_on_mismatch :
                raise ValueError (error_msg )
            else :
                print (f"[Client {client_id }] {error_msg }")
                return False 
        else :
            return True 
    if model_arch !=weight_arch :
        error_msg =(f"客戶端 {client_id } 模型架構與全局權重架構不匹配："f"模型架構={model_arch }, 權重架構={weight_arch }")
        if raise_on_mismatch :
            raise ValueError (error_msg )
        else :
            print (f"[Client {client_id }] {error_msg }")
            return False 
    return True 
def _apply_state_subset (model :nn .Module ,subset :Dict [str ,torch .Tensor ])->None :
    if not subset :
        return 
    own_state =model .state_dict ()
    skipped =[]
    for name ,tensor in subset .items ():
        if name not in own_state or not isinstance (tensor ,torch .Tensor ):
            continue 
        target =own_state [name ]
        if target .shape !=tensor .shape :
            skipped .append ((name ,tuple (tensor .shape ),tuple (target .shape )))
            continue 
        target .copy_ (tensor .to (target .device ))
    if skipped :
        try :
            client_id =getattr (getattr (model ,"args",None ),"client_id","?")
        except Exception :
            client_id ="?"
        print (f"[Client {client_id }] 偵測到 {len (skipped )} 個不相容權重層，已跳過以避免 shape mismatch：")
        for name ,src_shape ,dst_shape in skipped [:5 ]:
            print (f"  - {name }: global={src_shape } -> local={dst_shape }（已跳過）")
def _get_client_storage_dir (args )->str :
    base_dir =args .result_dir .strip ()if getattr (args ,"result_dir","").strip ()else config .LOG_DIR 
    client_dir =os .path .join (base_dir ,f"uav{args .client_id }")
    os .makedirs (client_dir ,exist_ok =True )
    return client_dir 
def _get_head_state_path (args )->str :
    return os .path .join (_get_client_storage_dir (args ),"head_state.pth")
def _save_local_head_state (args ,head_state :Dict [str ,torch .Tensor ])->None :
    if not head_state :
        return 
    path =_get_head_state_path (args )
    torch .save ({k :v .detach ().cpu ()for k ,v in head_state .items ()},path )
def _save_client_encoder_state (args ,model :nn .Module ,round_id :int )->None :
    if os .environ .get ("SAVE_CLIENT_ENCODER","1").strip ()=="0":
        return 
    try :
        exp_dir =os .environ .get ('EXPERIMENT_DIR',getattr (config ,'LOG_DIR','result'))
        os .makedirs (exp_dir ,exist_ok =True )
        encoder =model 
        state ={k :v .detach ().cpu ()for k ,v in encoder .state_dict ().items ()}
        payload ={"client_id":getattr (args ,"client_id",None ),"round":int (round_id ),"state_dict":state ,}
        client_dir =_get_client_storage_dir (args )
        per_client_path =os .path .join (client_dir ,"client_encoder_weights.pt")
        torch .save (payload ,per_client_path )
        print (f"[Client {args .client_id }] 💾 已保存 encoder: {per_client_path }")
        if int (getattr (args ,"client_id",0 ))==0 :
            root_path =os .path .join (exp_dir ,"client_encoder_weights.pt")
            torch .save (payload ,root_path )
            print (f"[Client {args .client_id }] 💾 已更新 root encoder: {root_path }")
    except Exception as e :
        print (f"[Client {getattr (args ,'client_id','?')}] 保存 encoder 失敗: {e }")
def _load_local_head_state (args )->Optional [Dict [str ,torch .Tensor ]]:
    path =_get_head_state_path (args )
    if not os .path .exists (path ):
        return None 
    try :
        loaded =torch .load (path ,map_location ="cpu")
        return {k :(v if isinstance (v ,torch .Tensor )else torch .tensor (v ))for k ,v in loaded .items ()}
    except Exception as exc :
        print (f"[Client {getattr (args ,'client_id','?')}] 無法載入本地 head 狀態: {exc }")
        return None 
def _extract_head_state (model :nn .Module )->Dict [str ,torch .Tensor ]:
    state =model .state_dict ()
    return {name :param .detach ().clone ().cpu ()for name ,param in state .items ()if _is_head_layer (name )}
class ClientCNN (nn .Module ):
    def __init__ (self ,input_dim :int ,num_classes :int )->None :
        super ().__init__ ()
        self .input_reshape =nn .Linear (input_dim ,128 )
        model_config =getattr (config ,'MODEL_CONFIG',{})
        channels =model_config .get ('channels',[32 ,64 ,128 ])
        kernel_sizes =model_config .get ('kernel_sizes',[3 ,3 ,3 ])
        dropout_rate =model_config .get ('dropout_rate',0.2 )
        self .conv_block1 =nn .Sequential (nn .Conv1d (1 ,channels [0 ],kernel_size =kernel_sizes [0 ],padding =1 ,bias =True ),nn .BatchNorm1d (channels [0 ]),nn .ReLU (inplace =True ),nn .Dropout (dropout_rate ))
        self .conv_block2 =nn .Sequential (nn .Conv1d (channels [0 ],channels [1 ],kernel_size =kernel_sizes [1 ],padding =1 ,bias =True ),nn .BatchNorm1d (channels [1 ]),nn .ReLU (inplace =True ),nn .Dropout (dropout_rate ))
        self .conv_block3 =nn .Sequential (nn .Conv1d (channels [1 ],channels [2 ],kernel_size =kernel_sizes [2 ],padding =1 ,bias =True ),nn .BatchNorm1d (channels [2 ]),nn .ReLU (inplace =True ),nn .Dropout (dropout_rate ))
        fc_hidden =model_config .get ('fc_hidden',128 )
        self .classifier =nn .Sequential (nn .Linear (channels [2 ],fc_hidden ,bias =True ),nn .ReLU (inplace =True ),nn .Dropout (dropout_rate ),nn .Linear (fc_hidden ,fc_hidden //2 ,bias =True ),nn .ReLU (inplace =True ),nn .Dropout (dropout_rate ),nn .Linear (fc_hidden //2 ,num_classes ,bias =True ))
    def forward (self ,x :torch .Tensor )->torch .Tensor :
        x =self .input_reshape (x )
        x =x .unsqueeze (1 )
        x =self .conv_block1 (x )
        x =self .conv_block2 (x )
        x =self .conv_block3 (x )
        x =torch .mean (x ,dim =2 )
        x =self .classifier (x )
        return x 
def build_model_from_config (input_dim :int ,num_classes :int )->nn .Module :
    model_type =config .MODEL_CONFIG .get ('type','dnn')
    if model_type =='transformer':
        try :
            from models .transformer import build_transformer 
            return build_transformer (input_dim =input_dim ,output_dim =num_classes ,d_model =config .MODEL_CONFIG .get ('d_model',128 ),num_layers =config .MODEL_CONFIG .get ('num_layers',2 ),num_heads =config .MODEL_CONFIG .get ('num_heads',4 ),d_ff =config .MODEL_CONFIG .get ('d_ff',None ),dropout =config .MODEL_CONFIG .get ('dropout_rate',0.3 ),max_seq_len =config .MODEL_CONFIG .get ('max_seq_len',input_dim ),use_positional_encoding =config .MODEL_CONFIG .get ('use_positional_encoding',True ))
        except ImportError as e :
            print (f"無法導入 Transformer 模型: {e }，使用本地 ClientCNN")
            return ClientCNN (input_dim ,num_classes )
    elif model_type =='dnn':
        try :
            from models .dnn import build_dnn 
            return build_dnn (input_dim =input_dim ,output_dim =num_classes ,hidden_dims =config .MODEL_CONFIG .get ('hidden_dims',[256 ,128 ,64 ]),dropout_rate =config .MODEL_CONFIG .get ('dropout_rate',0.3 ),use_batch_norm =config .MODEL_CONFIG .get ('use_batch_norm',True ),use_residual =config .MODEL_CONFIG .get ('use_residual',True ),activation =config .MODEL_CONFIG .get ('activation','relu'))
        except ImportError :
            print ("無法導入 DNN 模型，使用本地 ClientCNN")
            return ClientCNN (input_dim ,num_classes )
    elif model_type =='cnn':
        try :
            from models .cnn import build_cnn 
            return build_cnn (input_dim ,num_classes )
        except ImportError :
            print ("無法導入 CNN 模型，使用本地 ClientCNN")
            return ClientCNN (input_dim ,num_classes )
    else :
        print (f"未知模型類型 {model_type }，使用本地 ClientCNN")
        return ClientCNN (input_dim ,num_classes )
def _load_gan_filter_model (model_path :str ,input_dim :int ,num_classes :int ,device :torch .device )->Optional [nn .Module ]:
    if not model_path or not os .path .exists (model_path ):
        return None 
    cached =_GAN_FILTER_MODEL_CACHE 
    if (cached .get ("path")==model_path and cached .get ("input_dim")==input_dim and cached .get ("num_classes")==num_classes and cached .get ("model")is not None ):
        return cached ["model"]
    try :
        model =build_model_from_config (input_dim ,num_classes )
        state =torch .load (model_path ,map_location ="cpu")
        if isinstance (state ,dict )and "state_dict"in state :
            state =state ["state_dict"]
        elif isinstance (state ,dict )and "model_state_dict"in state :
            state =state ["model_state_dict"]
        if not isinstance (state ,dict ):
            return None 
        processed ={}
        for key ,value in state .items ():
            if isinstance (value ,list ):
                processed [key ]=torch .tensor (value )
            else :
                processed [key ]=value 
        model .load_state_dict (processed ,strict =False )
        model .to (device )
        model .eval ()
        _GAN_FILTER_MODEL_CACHE .update ({"path":model_path ,"input_dim":input_dim ,"num_classes":num_classes ,"model":model })
        return model 
    except Exception as e :
        print (f"  無法載入 GAN 過濾模型: {e }")
        return None 
def _extract_logits (output :Any )->torch .Tensor :
    if isinstance (output ,(tuple ,list ))and output :
        return cast (torch .Tensor ,output [-1 ])
    return cast (torch .Tensor ,output )
def _compute_kd_loss (student_logits :torch .Tensor ,teacher_logits :torch .Tensor ,temperature :float ,loss_type :str ,use_temperature :bool )->torch .Tensor :
    if use_temperature and temperature >0 :
        student_logits =student_logits /temperature 
        teacher_logits =teacher_logits /temperature 
    if loss_type =="kl_div":
        kd =F .kl_div (F .log_softmax (student_logits ,dim =-1 ),F .softmax (teacher_logits ,dim =-1 ),reduction ="batchmean")
        if use_temperature and temperature >0 :
            kd =kd *(temperature **2 )
        return kd 
    return F .mse_loss (student_logits ,teacher_logits )
def _build_teacher_model_from_weights (student_model :nn .Module ,raw_server_weights :Dict [str ,torch .Tensor ],device :torch .device )->Optional [nn .Module ]:
    if not raw_server_weights :
        return None 
    try :
        import copy 
        teacher =copy .deepcopy (student_model )
        processed ={}
        for key ,value in raw_server_weights .items ():
            if key .endswith ("num_batches_tracked"):
                if isinstance (value ,(int ,float ,np .integer ,np .floating )):
                    processed [key ]=torch .tensor (int (value ),dtype =torch .long )
                elif isinstance (value ,torch .Tensor ):
                    processed [key ]=value .long ()if value .dtype !=torch .long else value 
                else :
                    try :
                        processed [key ]=torch .tensor (int (value ),dtype =torch .long )
                    except (ValueError ,TypeError ):
                        processed [key ]=value 
            else :
                processed [key ]=value 
        if not processed :
            print (f"  server_weights 為空，無法建立 teacher")
            return None 
        student_keys =set (teacher .state_dict ().keys ())
        server_keys =set (processed .keys ())
        matched_keys =student_keys &server_keys 
        missing_keys =student_keys -server_keys 
        unexpected_keys =server_keys -student_keys 
        print (f"  GKD 權重匹配檢查（標準模型）:")
        print (f"    - 學生模型層數: {len (student_keys )}")
        print (f"    - 伺服器權重層數: {len (server_keys )}")
        print (f"    - 匹配層數: {len (matched_keys )}")
        if missing_keys :
            print (f"    - 缺失層數: {len (missing_keys )} (前5個: {list (missing_keys )[:5 ]})")
        if unexpected_keys :
            print (f"    - 額外層數: {len (unexpected_keys )} (前5個: {list (unexpected_keys )[:5 ]})")
        teacher .load_state_dict (processed ,strict =False )
        teacher .to (device )
        teacher .eval ()
        print (f"  成功建立標準模型 teacher（匹配 {len (matched_keys )}/{len (student_keys )} 層）")
        return teacher 
    except Exception as e :
        print (f"  建立標準模型 teacher 失敗，跳過 GKD: {e }")
        import traceback 
        print (f"  詳細錯誤: {traceback .format_exc ()}")
        return None 
class AdaptiveLabelSmoothingLoss (nn .Module ):
    def __init__ (self ,class_weights =None ,base_smoothing =0.1 ,class3_smoothing =0.2 ,num_classes =5 ):
        super ().__init__ ()
        self .class_weights =class_weights 
        self .base_smoothing =base_smoothing 
        self .class3_smoothing =class3_smoothing 
        self .num_classes =num_classes 
    def forward (self ,logits ,targets ):
        batch_size =targets .size (0 )
        device =targets .device 
        smooth_targets =torch .full ((batch_size ,self .num_classes ),self .base_smoothing /self .num_classes ,device =device )
        for i in range (batch_size ):
            true_label =targets [i ].item ()
            smoothing =self .class3_smoothing if true_label ==3 else self .base_smoothing 
            smooth_targets [i ].fill_ (smoothing /self .num_classes )
            smooth_targets [i ][true_label ]=1.0 -smoothing +(smoothing /self .num_classes )
        log_probs =nn .functional .log_softmax (logits ,dim =1 )
        loss =-torch .sum (smooth_targets *log_probs ,dim =1 )
        if self .class_weights is not None :
            weights =self .class_weights [targets ]
            loss =loss *weights 
        return loss .mean ()
def build_loss (class_weights =None ,label_smoothing =0.0 ):
    loss_config =getattr (config ,'LOSS_CONFIG',{})
    adaptive_label_smoothing =loss_config .get ('adaptive_label_smoothing',False )
    if adaptive_label_smoothing and label_smoothing >0 :
        num_classes =len (getattr (config ,'ALL_LABELS',[5 ]))
        return AdaptiveLabelSmoothingLoss (class_weights =class_weights ,base_smoothing =label_smoothing ,class3_smoothing =0.2 ,num_classes =num_classes )
    elif class_weights is not None :
        return nn .CrossEntropyLoss (weight =class_weights ,label_smoothing =label_smoothing )
    else :
        return nn .CrossEntropyLoss (label_smoothing =label_smoothing )
def _load_dynamic_class_weights_from_cloud (result_dir :str ,num_classes :int ,device :torch .device ):
    try :
        dyn_cfg =getattr (config ,"DYNAMIC_CLASS_WEIGHTING",{})or {}
        if not dyn_cfg .get ("enabled",False ):
            return None 
        if not result_dir or not result_dir .strip ():
            return None 
        filename =dyn_cfg .get ("filename","dynamic_class_weights.json")
        path =os .path .join (result_dir .strip (),filename )
        if not os .path .exists (path ):
            return None 
        import json 
        with open (path ,"r",encoding ="utf-8")as f :
            obj =json .load (f )or {}
        cw =obj .get ("class_weight")
        if not isinstance (cw ,dict ):
            return None 
        weights =[]
        for i in range (num_classes ):
            v =cw .get (str (i ),cw .get (i ,1.0 ))
            try :
                v =float (v )
            except Exception :
                v =1.0 
            weights .append (v )
        w_tensor =torch .tensor (weights ,dtype =torch .float32 ,device =device )
        print (f"[Client] 使用 Cloud 動態 class_weight（下一輪生效）：round={obj .get ('round')}, weights={weights }")
        return w_tensor 
    except Exception as e :
        print (f"[Client] 讀取 Cloud 動態 class_weight 失敗，改用既有權重策略: {e }")
        return None 
def update_round_learning_rate (optimizer :optim .Optimizer ,current_round :int )->Optional [float ]:
    import os 
    lr_config =getattr (config ,'LEARNING_RATE_SCHEDULER',{})
    if not lr_config .get ('enabled',False ):
        return None 
    mode =lr_config .get ('mode','').lower ()
    if mode not in ("manual","round"):
        return None 
    base_lr =optimizer .param_groups [0 ].get ('initial_lr',optimizer .param_groups [0 ].get ('lr',config .LEARNING_RATE ))
    max_lr =float (lr_config .get ('max_lr',base_lr ))
    min_lr =float (lr_config .get ('min_lr',base_lr *0.1 ))
    warmup_rounds =int (lr_config .get ('warmup_rounds',lr_config .get ('warmup_epochs',0 )))
    warmup_factor =float (lr_config .get ('warmup_factor',0.1 ))
    cycle_rounds =int (lr_config .get ('cycle_rounds',lr_config .get ('t_max_rounds',0 )))
    f1_based_decay =lr_config .get ('f1_based_decay',False )
    f1_threshold =float (lr_config .get ('f1_threshold',0.95 ))
    f1_decay_factor =float (lr_config .get ('f1_decay_factor',0.5 ))
    if warmup_rounds >0 and current_round <warmup_rounds :
        progress =(current_round +1 )/max (1 ,warmup_rounds )
        lr =max_lr *(warmup_factor +(1 -warmup_factor )*progress )
    elif cycle_rounds >0 :
        cycle_pos =(current_round -warmup_rounds )%cycle_rounds 
        cosine =(1 +math .cos (math .pi *cycle_pos /max (1 ,cycle_rounds )))/2 
        lr =min_lr +(max_lr -min_lr )*cosine 
    else :
        lr =max_lr 
    if f1_based_decay and os .environ .get ('HIGH_F1_LR_DECAY')=='1':
        try :
            current_global_f1 =float (os .environ .get ('CURRENT_GLOBAL_F1','0'))
            f1_high_threshold =float (lr_config .get ('f1_high_threshold',0.97 ))
            f1_high_decay_factor =float (lr_config .get ('f1_high_decay_factor',0.5 ))
            f1_ultra_threshold =float (lr_config .get ('f1_ultra_threshold',0.98 ))
            f1_ultra_decay_factor =float (lr_config .get ('f1_ultra_decay_factor',0.3 ))
            if current_global_f1 >f1_threshold :
                lr =lr *f1_decay_factor 
                print (f"  F1 第一階段衰減: F1={current_global_f1 :.4f} > {f1_threshold :.4f}，學習率 × {f1_decay_factor } = {lr :.6f}")
            if current_global_f1 >f1_high_threshold :
                lr =lr *f1_high_decay_factor 
                print (f"  F1 第二階段衰減: F1={current_global_f1 :.4f} > {f1_high_threshold :.4f}，學習率 × {f1_high_decay_factor } = {lr :.6f}")
            if current_global_f1 >f1_ultra_threshold :
                lr =lr *f1_ultra_decay_factor 
                print (f"  F1 第三階段衰減（精細微調）: F1={current_global_f1 :.4f} > {f1_ultra_threshold :.4f}，學習率 × {f1_ultra_decay_factor } = {lr :.6f}")
        except (ValueError ,TypeError ):
            pass 
    lr =max (min_lr ,min (max_lr ,lr ))
    for group in optimizer .param_groups :
        group ['lr']=lr 
    return lr 
def train_one_round (model :nn .Module ,train_loader :DataLoader ,optimizer :optim .Optimizer ,criterion :nn .Module ,device :torch .device ,global_weights =None ,current_round =0 ,epoch_num =0 ,total_epochs =1 ,teacher_model :Optional [nn .Module ]=None ,kd_cfg :Optional [Dict [str ,Any ]]=None )->Dict [str ,float ]:
    if total_epochs >1 :
        print (f"  開始訓練 Epoch {epoch_num +1 }/{total_epochs }，數據加載器大小: {len (train_loader )}")
    else :
        print (f"  開始訓練一輪，數據加載器大小: {len (train_loader )}")
    import sys 
    sys .stdout .flush ()
    updated_lr =update_round_learning_rate (optimizer ,current_round )
    if updated_lr is not None :
        print (f" Round-aware LR 已套用: lr={updated_lr :.6f} (round={current_round })")
        sys .stdout .flush ()
    kd_cfg =kd_cfg or {}
    kd_enabled =bool (kd_cfg .get ("enabled",False ))and (teacher_model is not None )
    kd_alpha =float (kd_cfg .get ("alpha",0.0 ))
    kd_temperature =float (kd_cfg .get ("temperature",4.0 ))
    kd_loss_type =str (kd_cfg .get ("loss_type","kl_div"))
    kd_use_temperature =bool (kd_cfg .get ("use_temperature",True ))
    kd_warmup_rounds =int (kd_cfg .get ("warmup_rounds",0 ))
    if current_round is not None and current_round <=kd_warmup_rounds :
        kd_enabled =False 
    if kd_enabled and teacher_model is not None :
        teacher_model .to (device )
        teacher_model .eval ()
    loss_guard_threshold =float (os .environ .get ("LOSS_EXPLOSION_THRESHOLD","10000"))
    max_bad_loss_batches =int (os .environ .get ("LOSS_EXPLOSION_MAX_BAD","8"))
    loss_backoff =float (os .environ .get ("LOSS_EXPLOSION_LR_BACKOFF","0.5"))
    loss_min_lr =float (os .environ .get ("LOSS_EXPLOSION_MIN_LR","1e-6"))
    bad_loss_count =0 
    fedprox_config =getattr (config ,'FEDPROX_CONFIG',{})
    fedprox_enabled =fedprox_config .get ('enabled',False )
    mu =fedprox_config .get ('mu',0.01 )
    adaptive_mu =fedprox_config .get ('adaptive_mu',True )
    mu_schedule =fedprox_config .get ('mu_schedule',{})
    max_rounds =getattr (config ,'MAX_ROUNDS',500 )
    mu_warmup_rounds =int (fedprox_config .get ('warmup_rounds',0 ))
    high_performance_mu =fedprox_config .get ('high_performance_mu',None )
    high_performance_threshold =fedprox_config .get ('high_performance_threshold',0.97 )
    if high_performance_mu is not None :
        try :
            current_global_f1 =float (os .environ .get ('CURRENT_GLOBAL_F1','0'))
            if current_global_f1 >high_performance_threshold :
                mu =high_performance_mu 
                print (f"  突破 0.98-0.99：F1={current_global_f1 :.4f} > {high_performance_threshold :.4f}，降低 FedProx mu 到 {mu :.4f}（允許更精細的調整）")
        except (ValueError ,TypeError ):
            pass 
    BREAKTHROUGH_ROUNDS =20 
    BREAKTHROUGH_START_ROUND =0 
    breakthrough_start =int (os .environ .get ("BREAKTHROUGH_START_ROUND",str (BREAKTHROUGH_START_ROUND )))
    if fedprox_enabled and breakthrough_start <=current_round <breakthrough_start +BREAKTHROUGH_ROUNDS :
        if high_performance_mu is None :
            mu =0.0 
            print (f"  破局配置 4.0：FedProx μ 暫時設為 0.0（輪次 {current_round }，允許模型跳出模式崩潰陷阱）")
        else :
            try :
                current_global_f1 =float (os .environ .get ('CURRENT_GLOBAL_F1','0'))
                if current_global_f1 <=high_performance_threshold :
                    mu =0.0 
                    print (f"  破局配置 4.0：FedProx μ 暫時設為 0.0（輪次 {current_round }，允許模型跳出模式崩潰陷阱）")
            except (ValueError ,TypeError ):
                mu =0.0 
                print (f"  破局配置 4.0：FedProx μ 暫時設為 0.0（輪次 {current_round }，允許模型跳出模式崩潰陷阱）")
    if fedprox_enabled and adaptive_mu and max_rounds >0 :
        effective_round =max (0 ,current_round -mu_warmup_rounds )
        total_span =max (1 ,max_rounds -mu_warmup_rounds )
        progress =min (1.0 ,effective_round /total_span )
        initial_mu =mu_schedule .get ('initial',mu )
        final_mu =mu_schedule .get ('final',mu *0.1 )
        decay_type =mu_schedule .get ('decay_type','cosine')
        if current_round <mu_warmup_rounds :
            mu =0.0 
            print (f" FedProx 暖啟中（round={current_round }/{mu_warmup_rounds }），暫停近端項")
        else :
            if decay_type =='cosine':
                mu =final_mu +(initial_mu -final_mu )*(1 +math .cos (math .pi *progress ))/2 
            else :
                mu =initial_mu +(final_mu -initial_mu )*progress 
            print (f" FedProx mu (輪次 {current_round }): {mu :.6f}")
    elif fedprox_enabled and current_round <mu_warmup_rounds :
        mu =0.0 
        print (f" FedProx 暖啟中（round={current_round }/{mu_warmup_rounds }），暫停近端項")
    if fedprox_enabled and global_weights is not None :
        global_params ={}
        for name ,param in model .named_parameters ():
            if name in global_weights :
                global_param =global_weights [name ]
                if isinstance (global_param ,torch .Tensor ):
                    global_params [name ]=global_param .to (device ).detach ()
                elif isinstance (global_param ,np .ndarray ):
                    global_params [name ]=torch .from_numpy (global_param ).float ().to (device ).detach ()
                else :
                    global_params [name ]=torch .tensor (global_param ,dtype =torch .float32 ).to (device ).detach ()
        print (f" FedProx 已啟用，mu={mu :.6f}，全局權重層數={len (global_params )}")
    else :
        global_params =None 
        if fedprox_enabled :
            print (f"  FedProx 已啟用但未提供全局權重，將不使用近端項")
    proto_cfg =getattr (config ,"PROTOTYPE_LOSS_CONFIG",{})
    proto_enabled =bool (proto_cfg .get ("enabled",False ))
    base_lambda_proto =float (proto_cfg .get ("lambda_proto",0.0 ))
    proto_log_every =int (proto_cfg .get ("log_every_n_batches",100 ))
    lambda_proto =base_lambda_proto 
    schedule_cfg =proto_cfg .get ("schedule",{})
    if proto_enabled and schedule_cfg .get ("enabled",False ):
        mode =str (schedule_cfg .get ("mode","cosine")).lower ()
        max_rounds =int (schedule_cfg .get ("max_rounds",getattr (config ,"MAX_ROUNDS",500 )))
        start_l =float (schedule_cfg .get ("start_lambda",base_lambda_proto ))
        end_l =float (schedule_cfg .get ("end_lambda",0.0 ))
        r =max (0 ,min (current_round ,max_rounds ))
        progress =r /max (1 ,max_rounds )
        if mode =="linear":
            lambda_proto =start_l +(end_l -start_l )*progress 
        else :
            import math as _math 
            cosine =(1 +_math .cos (_math .pi *progress ))/2.0 
            lambda_proto =end_l +(start_l -end_l )*cosine 
    use_proto =proto_enabled and lambda_proto >0.0 
    proto_tensor =None 
    if use_proto :
        try :
            import numpy as _np 
            proto_path =proto_cfg .get ("path",os .path .join ("model","global_prototypes.npy"))
            global _GLOBAL_PROTOTYPES_CACHE 
            if not _GLOBAL_PROTOTYPES_CACHE ["loaded"]:
                full_path =proto_path 
                if not os .path .isabs (full_path ):
                    full_path =os .path .join (os .path .dirname (os .path .abspath (__file__ )),full_path )
                if os .path .exists (full_path ):
                    print (f"  載入全局原型檔案: {full_path }")
                    data =_np .load (full_path ,allow_pickle =True ).item ()
                    protos =_np .asarray (data ["prototypes"],dtype =_np .float32 )
                    _GLOBAL_PROTOTYPES_CACHE ["tensor"]=torch .from_numpy (protos )
                    _GLOBAL_PROTOTYPES_CACHE ["support"]=_np .asarray (data .get ("support",[]))
                    _GLOBAL_PROTOTYPES_CACHE ["loaded"]=True 
                    print (f"  全局原型形狀: {_GLOBAL_PROTOTYPES_CACHE ['tensor'].shape }")
                else :
                    print (f"  找不到全局原型檔案: {full_path }，將停用 prototype loss")
                    use_proto =False 
            if _GLOBAL_PROTOTYPES_CACHE ["loaded"]:
                proto_tensor =_GLOBAL_PROTOTYPES_CACHE ["tensor"].to (device )
        except Exception as e :
            print (f"  載入全局原型失敗，停用 prototype loss: {e }")
            use_proto =False 
    grad_clip_cfg =getattr (config ,"GRADIENT_CLIPPING",{})
    grad_clip_enabled =bool (grad_clip_cfg .get ("enabled",False ))
    grad_clip_max_norm =float (grad_clip_cfg .get ("max_norm",1.0 ))
    grad_clip_norm_type =float (grad_clip_cfg .get ("norm_type",2 ))
    model .train ()
    total_loss =0.0 
    total_proximal_loss =0.0 
    total_proto_loss =0.0 
    correct =0 
    total =0 
    processed_batches =0 
    print (f"  模型設備: {next (model .parameters ()).device }, 目標設備: {device }")
    sys .stdout .flush ()
    log_interval =100 
    for batch_idx ,(data ,target )in enumerate (train_loader ):
        log_detail =(batch_idx %log_interval ==0 )
        if log_detail :
            print (f"  處理批次 {batch_idx }/{len (train_loader )}")
            sys .stdout .flush ()
        if log_detail :
            print (f"  批次 {batch_idx }: 數據形狀={data .shape }, 目標形狀={target .shape }")
            sys .stdout .flush ()
        data ,target =data .to (device ),target .to (device )
        if log_detail :
            print (f"  設備轉換完成: 數據設備={data .device }, 目標設備={target .device }")
            sys .stdout .flush ()
        optimizer .zero_grad ()
        if log_detail :
            print (f"  開始前向傳播...")
            sys .stdout .flush ()
        output =model (data )
        if log_detail :
            print (f"  前向傳播完成: 輸出形狀={output .shape }")
            sys .stdout .flush ()
        loss =criterion (output ,target )
        if log_detail :
            print (f"  損失計算完成: loss={loss .item ():.4f}")
            sys .stdout .flush ()
        if kd_enabled and kd_alpha >0 and teacher_model is not None :
            try :
                with torch .no_grad ():
                    teacher_out =teacher_model (data )
                student_logits =_extract_logits (output )
                teacher_logits =_extract_logits (teacher_out )
                kd_loss =_compute_kd_loss (student_logits ,teacher_logits ,kd_temperature ,kd_loss_type ,kd_use_temperature )
                loss =loss +kd_alpha *kd_loss 
                if log_detail :
                    print (f"  GKD loss: {kd_loss .item ():.6f} (α={kd_alpha :.4f}, T={kd_temperature :.2f})")
            except Exception as e :
                if log_detail :
                    print (f"  GKD 計算失敗，跳過: {e }")
        if use_proto and proto_tensor is not None and hasattr (model ,"get_embedding"):
            try :
                get_embedding_fn =cast (Callable [[torch .Tensor ],torch .Tensor ],getattr (model ,"get_embedding"))
                features =get_embedding_fn (data )
                if features .dim ()==1 :
                    features =features .unsqueeze (0 )
                if proto_tensor .dim ()==2 :
                    if proto_tensor .size (0 )>int (target .max ().item ()):
                        target_proto =proto_tensor [target .long ()]
                        proto_loss =((features -target_proto )**2 ).mean ()
                        loss =loss +lambda_proto *proto_loss 
                        total_proto_loss +=proto_loss .item ()
                        if batch_idx %proto_log_every ==0 :
                            print (f"  Prototype loss: {proto_loss .item ():.6f} (λ={lambda_proto :.4f})")
                    else :
                        if batch_idx %proto_log_every ==0 :
                            print ("  Prototype tensor 類別數不足，跳過 prototype loss")
                else :
                    if batch_idx %proto_log_every ==0 :
                        print ("  Prototype tensor 維度異常，跳過 prototype loss")
            except Exception as e :
                print (f"  計算 prototype loss 發生錯誤，該批次跳過: {e }")
        if fedprox_enabled and global_params is not None :
            proximal_loss :torch .Tensor =torch .tensor (0.0 ,device =device )
            skipped_layers =0 
            used_layers =0 
            for name ,param in model .named_parameters ():
                if name not in global_params :
                    continue 
                global_param =global_params [name ]
                if param .shape !=global_param .shape :
                    skipped_layers +=1 
                    continue 
                diff =param -global_param 
                proximal_loss +=(diff **2 ).sum ()
                used_layers +=1 
            if used_layers >0 :
                proximal_loss =mu *proximal_loss 
                proximal_loss_value =float (proximal_loss .item ())
                total_proximal_loss +=proximal_loss_value 
                loss =loss +proximal_loss 
                if log_detail :
                    print (f"   FedProx 近端項: {proximal_loss_value :.6f} (mu={mu :.6f}, used_layers={used_layers }, skipped_layers={skipped_layers })")
            elif log_detail :
                if global_params is None or len (global_params )==0 :
                    pass 
                elif skipped_layers >0 :
                    model_type =getattr (config ,'MODEL_CONFIG',{}).get ('type','unknown')
                    print (f"  FedProx 啟用但沒有任何 shape 相容的層（模型類型: {model_type }，可能是架構已變更），本輪不計算近端項")
                    print (f"     - 全局權重層數: {len (global_params )}")
                    print (f"     - 跳過的層數: {skipped_layers }")
                    print (f"     - 提示: 如果是第一次使用新架構，這是正常的；後續輪次應該能匹配")
                else :
                    print (f"  FedProx 啟用但沒有任何 shape 相容的層，本輪不計算近端項")
        client_diversity_penalty_enabled =getattr (config ,'CLIENT_DIVERSITY_PENALTY_ENABLED',True )
        client_diversity_weight =getattr (config ,'CLIENT_DIVERSITY_WEIGHT',0.1 )
        if client_diversity_penalty_enabled :
            try :
                output_probs =torch .softmax (output ,dim =-1 )
                entropy =-torch .sum (output_probs *torch .log (output_probs +1e-7 ),dim =-1 )
                avg_entropy =torch .mean (entropy )
                max_entropy =torch .log (torch .tensor (output .size (-1 ),dtype =torch .float32 ,device =device ))
                entropy_loss =1.0 -(avg_entropy /max_entropy )
                diversity_penalty =client_diversity_weight *entropy_loss 
                loss =loss +diversity_penalty 
                max_prob =torch .max (output_probs ,dim =-1 )[0 ]
                avg_max_prob =torch .mean (max_prob )
                avg_max_prob_value =float (avg_max_prob )
                avg_entropy_value =float (avg_entropy )
                max_entropy_value =float (max_entropy )
                diversity_penalty_value =float (diversity_penalty )
                if avg_max_prob_value >0.5 :
                    if log_detail :
                        print (f"  破局配置 3.3：客戶端檢測到預測分佈過於集中（平均最大機率={avg_max_prob_value :.4f}），應用多樣性懲罰（熵={avg_entropy_value :.4f}/{max_entropy_value :.4f}，損失={diversity_penalty_value :.4f}）")
            except Exception as e :
                if log_detail :
                    print (f"  客戶端多樣性懲罰計算失敗: {e }")
        loss_value =float (loss .item ())
        if (not math .isfinite (loss_value ))or (loss_value >loss_guard_threshold ):
            bad_loss_count +=1 
            print (f"  異常損失值: {loss_value :.4f}，跳過 batch 並降 LR (bad={bad_loss_count }/{max_bad_loss_batches })")
            for group in optimizer .param_groups :
                old_lr =float (group .get ("lr",0.0 ))
                new_lr =max (loss_min_lr ,old_lr *loss_backoff )
                group ["lr"]=new_lr 
            optimizer .zero_grad (set_to_none =True )
            if bad_loss_count >=max_bad_loss_batches :
                print ("  異常損失過多，提前結束本輪訓練以避免卡住")
                break 
            continue 
        loss .backward ()
        print (f"  反向傳播完成")
        sys .stdout .flush ()
        _apply_gradient_noise (model ,epoch_num ,batch_idx ,device )
        if grad_clip_enabled :
            grad_norm =torch .nn .utils .clip_grad_norm_ (model .parameters (),grad_clip_max_norm ,grad_clip_norm_type )
            if log_detail :
                print (f" 梯度裁剪: norm={float (grad_norm ):.6f}, max_norm={grad_clip_max_norm }")
                sys .stdout .flush ()
        if batch_idx %100 ==0 :
            total_grad_norm =0.0 
            param_count =0 
            zero_grad_count =0 
            max_grad =-float ('inf')
            min_grad =float ('inf')
            for name ,param in model .named_parameters ():
                if param .grad is not None :
                    grad_norm =param .grad .norm ().item ()
                    total_grad_norm +=grad_norm 
                    param_count +=1 
                    max_grad =max (max_grad ,param .grad .abs ().max ().item ())
                    min_grad =min (min_grad ,param .grad .abs ().min ().item ())
                    if grad_norm <1e-8 :
                        zero_grad_count +=1 
                else :
                    zero_grad_count +=1 
            if param_count >0 :
                avg_grad_norm =total_grad_norm /param_count 
                print (f"  梯度統計 (批次 {batch_idx }): 平均範數={avg_grad_norm :.6f}, 最大梯度={max_grad :.6f}, 最小梯度={min_grad :.6f}, 零梯度層數={zero_grad_count }")
                if avg_grad_norm <1e-6 :
                    print (f"  警告：梯度範數極小，可能存在梯度消失問題")
                if zero_grad_count >param_count *0.5 :
                    print (f"  警告：超過50%的參數梯度為零，訓練可能無效")
        optimizer .step ()
        print (f"  優化器步驟完成")
        sys .stdout .flush ()
        total_loss +=loss_value 
        processed_batches +=1 
        _ ,predicted =torch .max (output .data ,1 )
        total +=target .size (0 )
        correct +=(predicted ==target ).sum ().item ()
    if processed_batches ==0 :
        print ("  本輪未完成任何有效 batch，回傳 0 loss/acc")
        avg_loss =0.0 
        avg_proximal_loss =0.0 
        avg_proto_loss =0.0 
        accuracy =0.0 
    else :
        avg_loss =total_loss /processed_batches 
        avg_proximal_loss =total_proximal_loss /processed_batches if total_proximal_loss >0 else 0.0 
        avg_proto_loss =total_proto_loss /processed_batches if total_proto_loss >0 else 0.0 
        accuracy =correct /max (1 ,total )
    if fedprox_enabled and avg_proximal_loss >0 :
        print (f" FedProx 平均近端損失: {avg_proximal_loss :.6f}")
    if use_proto and avg_proto_loss >0 :
        print (f"  Prototype 平均損失: {avg_proto_loss :.6f}")
    if hasattr (model ,'_weights_before_training'):
        prev_weights =cast (Dict [str ,torch .Tensor ],getattr (model ,'_weights_before_training'))
        weight_changes ={}
        for name ,param in model .named_parameters ():
            if name in prev_weights :
                old_weight =prev_weights [name ]
                if param .shape ==old_weight .shape :
                    weight_diff =(param .data -old_weight ).abs ().mean ().item ()
                    weight_norm =param .data .norm ().item ()
                    weight_changes [name ]={'mean_diff':weight_diff ,'norm':weight_norm ,'relative_change':weight_diff /(weight_norm +1e-8 )}
        if weight_changes :
            avg_change =sum (w ['mean_diff']for w in weight_changes .values ())/len (weight_changes )
            avg_relative =sum (w ['relative_change']for w in weight_changes .values ())/len (weight_changes )
            print (f"  訓練權重更新: 平均絕對變化={avg_change :.6f}, 平均相對變化={avg_relative :.6f}")
            if avg_change <1e-6 :
                print (f"  警告：權重更新極小，可能未有效訓練")
    if total_epochs >1 :
        print (f"  Epoch {epoch_num +1 }/{total_epochs } 訓練完成: 平均損失={avg_loss :.4f}, 準確率={accuracy :.4f}")
    else :
        print (f"  訓練完成: 平均損失={avg_loss :.4f}, 準確率={accuracy :.4f}")
    import sys 
    sys .stdout .flush ()
    return {'loss':avg_loss ,'acc':accuracy ,'ce_loss':avg_loss -avg_proximal_loss -avg_proto_loss ,'fedprox_loss':avg_proximal_loss ,'proto_loss':avg_proto_loss }
def evaluate (model :nn .Module ,val_loader :DataLoader ,criterion :nn .Module ,device :torch .device )->Dict [str ,Any ]:
    model .eval ()
    total_loss =0.0 
    all_predictions :List [int ]=[]
    all_targets :List [int ]=[]
    alpha_local =getattr (config ,"JOINT_PREDICTION_ALPHA_LOCAL",0.5 )
    alpha_agg =getattr (config ,"JOINT_PREDICTION_ALPHA_AGGREGATOR",0.25 )
    alpha_global =getattr (config ,"JOINT_PREDICTION_ALPHA_GLOBAL",0.25 )
    alpha_sum =float (alpha_local )+float (alpha_agg )+float (alpha_global )
    if alpha_sum >0 :
        alpha_local /=alpha_sum 
        alpha_agg /=alpha_sum 
        alpha_global /=alpha_sum 
    has_global_head =(hasattr (model ,"_global_head_state")and isinstance (getattr (model ,"_global_head_state"),dict )and len (getattr (model ,"_global_head_state"))>0 )
    agg_head_state :Optional [Dict [str ,torch .Tensor ]]=None 
    try :
        aggregator_id_env =os .environ .get ("AGGREGATOR_ID")
        agg_id :Optional [int ]=None 
        if aggregator_id_env is not None :
            try :
                agg_id =int (aggregator_id_env )
            except ValueError :
                agg_id =None 
        if agg_id is None and hasattr (config ,"NUM_AGGREGATORS")and hasattr (config ,"NUM_CLIENTS"):
            try :
                client_id_env =os .environ .get ("CLIENT_ID")
                if client_id_env is not None :
                    cid =int (client_id_env )
                    per_agg =max (1 ,int (math .ceil (config .NUM_CLIENTS /config .NUM_AGGREGATORS )))
                    agg_id =min (config .NUM_AGGREGATORS -1 ,cid //per_agg )
            except Exception :
                agg_id =None 
        if agg_id is not None :
            experiment_dir =os .environ .get ("EXPERIMENT_DIR",config .LOG_DIR )
            agg_model_path =os .path .join (experiment_dir ,"aggregator_models",f"aggregator_{agg_id }_latest.pt")
            if os .path .exists (agg_model_path ):
                payload =torch .load (agg_model_path ,map_location ="cpu")
                if isinstance (payload ,dict )and "state_dict"in payload :
                    full_state =payload ["state_dict"]
                    agg_head_state ={name :(tensor if isinstance (tensor ,torch .Tensor )else torch .tensor (tensor ))for name ,tensor in full_state .items ()if _is_head_layer (name )}
    except Exception as load_agg_exc :
        print (f"[Client] 載入聚合器 latest 權重失敗，僅使用 local/global：{load_agg_exc }")
        agg_head_state =None 
    use_joint_fusion =has_global_head or agg_head_state is not None 
    with torch .no_grad ():
        total_samples =0 
        for data ,target in val_loader :
            data ,target =data .to (device ),target .to (device )
            batch_size =data .size (0 )
            total_samples +=batch_size 
            output_local =model (data )
            loss =criterion (output_local ,target )
            loss_value =float (loss .item ())
            if not (0 <=loss_value <1000 ):
                print (f"[Client] 檢測到異常損失值: {loss_value :.2f}，進行裁剪")
                loss_value =min (loss_value ,1000.0 )
            total_loss +=loss_value *batch_size 
            output_for_pred =output_local 
            if use_joint_fusion :
                try :
                    local_head_state =_extract_head_state (model )
                    logits_local =output_local 
                    logits_global =None 
                    logits_agg =None 
                    if has_global_head and alpha_global >0.0 :
                        global_head_state =getattr (model ,"_global_head_state",{})
                        if global_head_state :
                            _apply_state_subset (model ,global_head_state )
                            logits_global =model (data )
                            _apply_state_subset (model ,local_head_state )
                    if agg_head_state is not None and alpha_agg >0.0 :
                        _apply_state_subset (model ,agg_head_state )
                        logits_agg =model (data )
                        _apply_state_subset (model ,local_head_state )
                    probs_local =torch .softmax (logits_local ,dim =1 )
                    probs_fused =alpha_local *probs_local 
                    if logits_agg is not None :
                        probs_agg =torch .softmax (logits_agg ,dim =1 )
                        probs_fused =probs_fused +alpha_agg *probs_agg 
                    if logits_global is not None :
                        probs_global =torch .softmax (logits_global ,dim =1 )
                        probs_fused =probs_fused +alpha_global *probs_global 
                    _ ,predicted =torch .max (probs_fused ,1 )
                except Exception as fusion_exc :
                    print (f"[Client] 聯合預測融合失敗，回退為本地評估: {fusion_exc }")
                    _ ,predicted =torch .max (output_for_pred .data ,1 )
            else :
                _ ,predicted =torch .max (output_for_pred .data ,1 )
            all_predictions .extend (predicted .detach ().cpu ().numpy ())
            all_targets .extend (target .detach ().cpu ().numpy ())
    if not all_targets :
        return {"val_loss":0.0 ,"val_acc":0.0 ,"f1_score":0.0 }
    avg_loss =total_loss /max (1 ,total_samples )
    accuracy =accuracy_score (all_targets ,all_predictions )
    f1 =f1_score (all_targets ,all_predictions ,average ='weighted')
    result :Dict [str ,Any ]={'val_loss':float (avg_loss ),'val_acc':float (accuracy ),'f1_score':float (f1 )}
    try :
        from sklearn .metrics import precision_recall_fscore_support 
        precision ,recall ,f1_macro ,support =precision_recall_fscore_support (all_targets ,all_predictions ,average ='macro',zero_division ="0")
        f1_micro =f1_score (all_targets ,all_predictions ,average ='micro')
        num_classes =config .MODEL_CONFIG .get ('num_classes',5 )
        cm =confusion_matrix (all_targets ,all_predictions ,labels =list (range (num_classes )))
        auc_micro =0.0 
        auc_macro =0.0 
        try :
            model .eval ()
            all_probas =[]
            all_targets_for_auc =[]
            with torch .no_grad ():
                for data ,target in val_loader :
                    data =data .to (device )
                    output_local =model (data )
                    probas =torch .softmax (output_local ,dim =1 )
                    all_probas .extend (probas .cpu ().numpy ())
                    all_targets_for_auc .extend (target .numpy ())
            if all_probas :
                from sklearn .metrics import roc_auc_score 
                from sklearn .preprocessing import label_binarize 
                y_true_binary =label_binarize (all_targets_for_auc ,classes =list (range (5 )))
                auc_micro =roc_auc_score (y_true_binary ,all_probas ,average ='micro',multi_class ='ovr')
                auc_macro =roc_auc_score (y_true_binary ,all_probas ,average ='macro',multi_class ='ovr')
        except Exception as auc_e :
            print (f"[Client] 計算 AUC 失敗: {auc_e }")
        result ['f1_macro']=float (f1_macro )
        result ['f1_micro']=float (f1_micro )
        result ['precision_macro']=float (precision )
        result ['recall_macro']=float (recall )
        result ['auc_micro']=float (auc_micro )
        result ['auc_macro']=float (auc_macro )
        result ['confusion_matrix']=cm .tolist ()
        result ['predictions']=all_predictions 
        result ['targets']=all_targets 
    except Exception as e :
        print (f"[Client] 計算詳細指標失敗: {e }")
    return result 
async def register_client (session :aiohttp .ClientSession ,aggregator_url :str ,client_id :int )->bool :
    try :
        url =f"{aggregator_url }/register_client"
        data =aiohttp .FormData ()
        data .add_field ('client_id',str (client_id ))
        async with session .post (url ,data =data )as resp :
            if resp .status ==200 :
                print (f" 客戶端 {client_id } 註冊成功")
                return True 
            else :
                print (f" 客戶端 {client_id } 註冊失敗: {resp .status }")
                return False 
    except Exception as e :
        print (f" 客戶端 {client_id } 註冊異常: {e }")
        return False 
async def get_federated_status (session :aiohttp .ClientSession ,aggregator_url :str )->Dict [str ,Any ]:
    try :
        url =f"{aggregator_url }/federated_status"
        async with session .get (url ,timeout =aiohttp .ClientTimeout (total =10 ))as resp :
            if resp .status ==200 :
                result =await resp .json ()
                current_round =result .get ('current_round',result .get ('round_count',0 ))
                selected_clients =result .get ('selected_clients',[])
                print (f"[Client] 獲取 federated_status: round={current_round }, selected_clients={selected_clients }")
                return result 
            else :
                error_text =await resp .text ()
                print (f"[Client] Error getting status: HTTP {resp .status }, {error_text }")
                return {}
    except asyncio .TimeoutError :
        print (f"[Client] federated_status 查詢超時")
        return {}
    except Exception as e :
        print (f"[Client] Error getting status: {e }")
        return {}
async def get_global_weights (session :aiohttp .ClientSession ,aggregator_url :str )->Optional [Dict [str ,Any ]]:
    try :
        url =f"{aggregator_url }/get_global_weights"
        async with session .get (url )as resp :
            if resp .status ==200 :
                data =await resp .json ()
                if isinstance (data ,dict )and "global_weights"not in data :
                    raw_global =data 
                    raw_server =None 
                    model_version =None 
                    server_model_version =None 
                else :
                    raw_global =data .get ("global_weights",{})
                    raw_server =data .get ("server_weights")if isinstance (data ,dict )else None 
                    model_version =data .get ("model_version")
                    server_model_version =data .get ("server_model_version")
                def _to_tensors (raw :Optional [Dict [str ,Any ]])->Dict [str ,torch .Tensor ]:
                    if not raw :
                        return {}
                    converted ={}
                    for key ,value in raw .items ():
                        if isinstance (value ,list ):
                            converted [key ]=torch .tensor (value )
                        else :
                            converted [key ]=value 
                    return converted 
                return {"global_weights":_to_tensors (raw_global ),"server_weights":_to_tensors (raw_server )if raw_server is not None else {},"model_version":model_version ,"server_model_version":server_model_version }
            else :
                print (f"Error getting global weights: {resp .status }")
                return None 
    except Exception as e :
        print (f"Error getting global weights: {e }")
        return None 
async def upload_weights (session :aiohttp .ClientSession ,aggregator_url :str ,client_id :int ,weights :Dict [str ,torch .Tensor ],sample_count :int ,val_stats :Dict [str ,float ],round_id :int ,max_retries :int =3 ,base_delay :float =1.0 ,metrics_payload :Optional [Dict [str ,Any ]]=None ,)->Tuple [bool ,int ,str ]:
    for attempt in range (max_retries ):
        try :
            url =f"{aggregator_url }/upload_federated_weights"
            weights_data ={}
            for key ,tensor in weights .items ():
                weights_data [key ]=tensor .cpu ().tolist ()
            data =aiohttp .FormData ()
            data .add_field ('client_id',str (client_id ))
            data .add_field ('data_size',str (sample_count ))
            data .add_field ('round_id',str (round_id ))
            data .add_field ('commit_id','')
            data .add_field ('attack_ratio','0.0')
            data .add_field ('client_version','1.0')
            data .add_field ('model_signature','')
            data .add_field ('train_time_ms','0')
            import io 
            import pickle 
            weights_buffer =io .BytesIO ()
            pickle .dump (weights_data ,weights_buffer )
            weights_buffer .seek (0 )
            data .add_field ('weights',weights_buffer ,filename ='weights.pkl',content_type ='application/octet-stream')
            if metrics_payload :
                try :
                    data .add_field ('metrics',json .dumps (metrics_payload ),content_type ='application/json')
                except Exception as e :
                    print (f"[Client {client_id }] 無法附加 metrics: {e }")
            async with session .post (url ,data =data )as resp :
                if resp .status ==200 :
                    result =await resp .json ()
                    accepted =result .get ('accepted',False )
                    server_round =result .get ('server_round',-1 )
                    reason =result .get ('reason','unknown')
                    print (f"[Client {client_id }] 📤 上傳回應: accepted={accepted }, server_round={server_round }, reason={reason }")
                    if accepted :
                        return True ,resp .status ,f"Success: {result .get ('message','Accepted')}"
                    else :
                        if attempt <max_retries -1 :
                            try :
                                await sync_client_state (session ,aggregator_url ,client_id ,round_id )
                            except Exception :
                                pass 
                            delay =base_delay *(2 **attempt )
                            print (f"[Client {client_id }]  上傳未被接受，{delay :.1f}s 後重試...")
                            await asyncio .sleep (delay )
                            continue 
                        return False ,409 ,f"Rejected: {reason }"
                else :
                    error_text =await resp .text ()
                    print (f"[Client {client_id }] 上傳失敗: HTTP {resp .status }, {error_text }")
                    return False ,resp .status ,error_text 
        except Exception as e :
            print (f"[Client {client_id }] 上傳異常 (attempt {attempt +1 }/{max_retries }): {e }")
            if attempt <max_retries -1 :
                delay =base_delay *(2 **attempt )
                print (f"[Client {client_id }]  等待 {delay :.1f}s 後重試...")
                await asyncio .sleep (delay )
            else :
                return False ,0 ,f"Max retries exceeded: {str (e )}"
    return False ,0 ,"All retries failed"
async def sync_client_state (session :aiohttp .ClientSession ,aggregator_url :str ,client_id :int ,last_confirmed_round :int )->Optional [Dict ]:
    try :
        url =f"{aggregator_url }/sync_state"
        data =aiohttp .FormData ()
        data .add_field ('client_id',str (client_id ))
        data .add_field ('last_confirmed_round',str (last_confirmed_round ))
        async with session .post (url ,data =data )as resp :
            if resp .status ==200 :
                result =await resp .json ()
                print (f"[Client {client_id }] 狀態同步回應: {result }")
                return result 
            else :
                error_text =await resp .text ()
                print (f"[Client {client_id }] 狀態同步失敗: HTTP {resp .status }, {error_text }")
                return None 
    except Exception as e :
        print (f"[Client {client_id }] 狀態同步異常: {e }")
        return None 
# pyright: ignore[reportGeneralTypeIssues]
async def main():
    parser =argparse .ArgumentParser (description ='UAV Client for Federated Learning')
    parser .add_argument ('--client_id',type =int ,required =True ,help ='Client ID')
    parser .add_argument ('--aggregator_url',type =str ,required =True ,help ='Aggregator URL')
    parser .add_argument ('--data_path',type =str ,required =False ,help ='Data file path (deprecated, using processed data)')
    parser .add_argument ('--cloud_url',type =str ,required =True ,help ='Cloud Server URL')
    parser .add_argument ('--result_dir',type =str ,default ='',help ='Result directory for metrics')
    args =parser .parse_args ()
    import sys 
    sys .stdout .flush ()
    sys .stderr .flush ()
    print (" 優化後的聯邦學習配置已加載")
    print (" 核心參數:")
    print (f"  - 學習率: {config .LEARNING_RATE }")
    print (f"  - 批次大小: {config .BATCH_SIZE }")
    print (f"  - 本地訓練輪數: {config .LOCAL_EPOCHS }")
    print (f"  - 最大輪數: {config .MAX_ROUNDS }")
    print (f"  - 客戶端數量: {config .NUM_CLIENTS }")
    print (f"  - 聚合器數量: {config .NUM_AGGREGATORS }")
    print (f"  - 模型類型: {config .MODEL_TYPE }")
    print (" 配置驗證通過")
    sys .stdout .flush ()
    force_cpu =os .environ .get ("FORCE_CPU","1").strip ().lower ()in ("1","true","yes","on")
    force_cuda =os .environ .get ("FORCE_CUDA","0").strip ().lower ()in ("1","true","yes","on")
    if force_cpu :
        device =torch .device ("cpu")
        print (f" 強制使用CPU設備（FORCE_CPU=1）")
    elif force_cuda and torch .cuda .is_available ():
        device =torch .device ("cuda")
        print (f" 強制使用CUDA設備（FORCE_CUDA=1）")
    else :
        try :
            from utils .cuda_memory_manager import get_memory_manager ,print_memory_status 
            memory_manager =get_memory_manager ()
            device =memory_manager .get_optimal_device ()
            print_memory_status ()
        except ImportError :
            device =torch .device ("cpu")
            if torch .cuda .is_available ():
                try :
                    torch .cuda .empty_cache ()
                    memory_allocated =torch .cuda .memory_allocated ()
                    memory_reserved =torch .cuda .memory_reserved ()
                    memory_total =torch .cuda .get_device_properties (0 ).total_memory 
                    print (f" CUDA記憶體狀態: 已分配={memory_allocated /1024 **3 :.2f}GB, 已保留={memory_reserved /1024 **3 :.2f}GB, 總計={memory_total /1024 **3 :.2f}GB")
                    if memory_reserved /memory_total <0.3 :
                        device =torch .device ("cuda")
                        print (f" 使用CUDA設備: {device }")
                    else :
                        print (f"CUDA記憶體使用率過高({memory_reserved /memory_total *100 :.1f}%)，使用CPU")
                        device =torch .device ("cpu")
                except RuntimeError as e :
                    print (f"CUDA錯誤，使用CPU: {e }")
                    device =torch .device ("cpu")
            else :
                print (f"Using device: {device }")
    metrics_dir =args .result_dir .strip ()or os .getcwd ()
    _write_attack_metadata (metrics_dir ,args .client_id )
    curve_csv =None 
    if args .result_dir .strip ():
        try :
            legacy_client_dir =os .path .join (args .result_dir .strip (),f"uav{args .client_id }")
            os .makedirs (legacy_client_dir ,exist_ok =True )
            curve_csv =os .path .join (legacy_client_dir ,f"uav{args .client_id }_curve.csv")
        except Exception :
            curve_csv =None 
    if curve_csv and not os .path .isfile (curve_csv ):
        try :
            with open (curve_csv ,"w",encoding ="utf-8")as f :
                f .write ("round,loss,acc,val_loss,val_acc,joint_f1,alpha,label_flip_count,data_size,upload_ok,status_code,server_status,server_msg,train_time_ms\n")
        except Exception :
            pass 
    print (f"[Client {args .client_id }]使用預處理後的數據...")
    import sys 
    sys .stdout .flush ()
    processed_data_dir =config .DATA_PATH 
    client_data_dir =os .path .join (processed_data_dir ,f"uav{args .client_id }")
    def _choose_scaled_path (base_dir :str ,name :str )->str :
        prefer_scaled =bool (os .environ .get ("PREFER_SCALED_DATA",str (getattr (config ,"PREFER_SCALED_DATA",True ))).strip ()!="0")
        scaled_path =os .path .join (base_dir ,f"{name }_scaled.csv")
        raw_path =os .path .join (base_dir ,f"{name }.csv")
        if prefer_scaled and os .path .exists (scaled_path ):
            return scaled_path 
        return raw_path 
    train_path =_choose_scaled_path (client_data_dir ,"train")
    test_path =_choose_scaled_path (client_data_dir ,"test")
    client_eval_source =os .environ .get ("CLIENT_EVAL_SOURCE",getattr (config ,"CLIENT_EVAL_SOURCE","local_split")).strip ().lower ()
    if client_eval_source not in ("global_test","local_split"):
        print (f"[Client {args .client_id }] 未知 CLIENT_EVAL_SOURCE={client_eval_source }，改用 local_split")
        client_eval_source ="local_split"
    if not os .path .exists (train_path ):
        error_msg =f" [Client {args .client_id }] 預處理數據不存在: {train_path }\n"
        error_msg +=f"請先運行 preprocess_data.py 生成預處理數據\n"
        error_msg +=f"檢查的數據目錄: {client_data_dir }\n"
        error_msg +=f"配置的 DATA_PATH: {config .DATA_PATH }\n"
        print (error_msg )
        sys .stdout .flush ()
        if args .result_dir .strip ():
            try :
                client_log_path =os .path .join (args .result_dir .strip (),f"uav{args .client_id }",f"client_{args .client_id }.log")
                with open (client_log_path ,'a',encoding ='utf-8')as f :
                    f .write (error_msg )
            except Exception :
                pass 
        sys .exit (1 )
    if client_eval_source =="global_test"and not os .path .exists (test_path ):
        error_msg =f" [Client {args .client_id }] 測試數據不存在: {test_path }\n"
        error_msg +=f"請先運行 preprocess_data.py 生成預處理數據\n"
        error_msg +=f"檢查的數據目錄: {client_data_dir }\n"
        error_msg +=f"配置的 DATA_PATH: {config .DATA_PATH }\n"
        print (error_msg )
        sys .stdout .flush ()
        if args .result_dir .strip ():
            try :
                client_log_path =os .path .join (args .result_dir .strip (),f"uav{args .client_id }",f"client_{args .client_id }.log")
                with open (client_log_path ,'a',encoding ='utf-8')as f :
                    f .write (error_msg )
            except Exception :
                pass 
        sys .exit (1 )
    print (f"[Client {args .client_id }] 📂 開始讀取訓練數據: {train_path }")
    sys .stdout .flush ()
    try :
        max_train_samples =config .DATA_CONFIG .get ('max_train_samples',None )
        if max_train_samples :
            train_df =pd .read_csv (train_path ,nrows =max_train_samples )
            print (f"[Client {args .client_id }] 訓練數據讀取完成: {len (train_df )} 行 (限制為最多 {max_train_samples } 行)")
        else :
            train_df =pd .read_csv (train_path )
            print (f"[Client {args .client_id }] 訓練數據讀取完成: {len (train_df )} 行")
        label_col =config .LABEL_COL 
        if label_col in train_df .columns :
            min_samples_per_class =int (os .environ .get ("MIN_SAMPLES_PER_CLASS","50"))
            if min_samples_per_class <=0 :
                print (f"[Client {args .client_id }] 已停用本地重採樣（MIN_SAMPLES_PER_CLASS={min_samples_per_class }）")
                min_samples_per_class =0 
            class_counts =train_df [label_col ].value_counts ()
            oversampled_dfs =[]
            for class_label ,count in class_counts .items ():
                class_data =train_df [train_df [label_col ]==class_label ]
                if min_samples_per_class >0 and 0 <count <min_samples_per_class :
                    repeat_factor =(min_samples_per_class +count -1 )//count 
                    oversampled_class =pd .concat ([class_data ]*repeat_factor ,ignore_index =True )
                    oversampled_class =oversampled_class .head (min_samples_per_class )
                    oversampled_dfs .append (oversampled_class )
                    print (f"[Client {args .client_id }]類別 {class_label } 樣本數不足 ({count } < {min_samples_per_class })，重採樣至 {len (oversampled_class )} 個樣本")
                else :
                    oversampled_dfs .append (class_data )
            if oversampled_dfs :
                train_df =pd .concat (oversampled_dfs ,ignore_index =True )
                train_df =train_df .sample (frac =1 ,random_state =42 +args .client_id ).reset_index (drop =True )
                print (f"[Client {args .client_id }] 重採樣完成，最終訓練樣本數: {len (train_df )}")
        if label_col in train_df .columns :
            feature_cols =[col for col in train_df .columns if col !=label_col ]
            train_df =_apply_gan_augmentation (train_df ,label_col ,feature_cols ,args .client_id )
        sys .stdout .flush ()
    except Exception as e :
        error_msg =f" [Client {args .client_id }] 讀取訓練數據失敗: {e }\n"
        import traceback 
        error_msg +=f"Traceback: {traceback .format_exc ()}\n"
        print (error_msg )
        sys .stdout .flush ()
        if args .result_dir .strip ():
            try :
                client_log_path =os .path .join (args .result_dir .strip (),f"uav{args .client_id }",f"client_{args .client_id }.log")
                with open (client_log_path ,'a',encoding ='utf-8')as f :
                    f .write (error_msg )
            except Exception :
                pass 
        sys .exit (1 )
    label_flip_count =0 
    if client_eval_source !="global_test":
        eval_ratio =float (os .environ .get ("CLIENT_LOCAL_EVAL_RATIO",str (getattr (config ,"CLIENT_LOCAL_EVAL_RATIO",0.2 ))))
        eval_ratio =max (0.05 ,min (eval_ratio ,0.5 ))
        if len (train_df )<2 :
            val_df =train_df .copy ()
            train_df =train_df .copy ()
            print (f"[Client {args .client_id }] 訓練樣本過少，使用相同資料作為驗證集")
        else :
            try :
                from sklearn .model_selection import train_test_split 
                train_df ,val_df =train_test_split (train_df ,test_size =eval_ratio ,stratify =train_df ['label']if 'label'in train_df .columns else None ,random_state =42 )
            except Exception as split_e :
                print (f"[Client {args .client_id }] 本地分割失敗，改用隨機切分: {split_e }")
                val_size =max (1 ,int (len (train_df )*eval_ratio ))
                val_df =train_df .sample (n =val_size ,random_state =42 )
                train_df =train_df .drop (val_df .index )
        train_df ,label_flip_count =_apply_label_flipping (train_df ,args .client_id )
        train_df =train_df .reset_index (drop =True )
        test_df =val_df .reset_index (drop =True )
        print (f"[Client {args .client_id }] 使用本地切分驗證集: train={len (train_df )}, val={len (test_df )} (ratio={eval_ratio })")
        print (f"[Client {args .client_id }] 已停用全域測試集評估，改由雲端統一評估")
        sys .stdout .flush ()
    else :
        train_df ,label_flip_count =_apply_label_flipping (train_df ,args .client_id )
    print (f"[Client {args .client_id }] 📂 開始讀取測試數據: {test_path }")
    sys .stdout .flush ()
    try :
        file_size_mb =os .path .getsize (test_path )/(1024 *1024 )
        print (f"[Client {args .client_id }] 測試數據文件大小: {file_size_mb :.2f} MB")
        sys .stdout .flush ()
        max_test_samples =config .DATA_CONFIG .get ('max_test_samples',None )
        data_usage_ratio =config .DATA_CONFIG .get ('data_usage_ratio',1.0 )
        if file_size_mb >100 :
            print (f"[Client {args .client_id }] 測試數據文件較大，使用分塊讀取...")
            if max_test_samples :
                print (f"[Client {args .client_id }] 將限制測試數據為最多 {max_test_samples :,} 行")
            elif data_usage_ratio <1.0 :
                print (f"[Client {args .client_id }] 將使用 {data_usage_ratio *100 :.1f}% 的數據")
            sys .stdout .flush ()
            chunks =[]
            if file_size_mb >1000 :
                chunk_size =20000 
            elif file_size_mb >500 :
                chunk_size =30000 
            else :
                chunk_size =50000 
            print (f"[Client {args .client_id }] 使用chunk size: {chunk_size :,} 行")
            sys .stdout .flush ()
            total_chunks =0 
            total_rows =0 
            start_time =time .time ()
            for i ,chunk in enumerate (pd .read_csv (test_path ,chunksize =chunk_size )):
                if max_test_samples :
                    if total_rows >=max_test_samples :
                        break 
                    remaining =max_test_samples -total_rows 
                    if remaining <len (chunk ):
                        chunk =chunk .head (remaining )
                        chunks .append (chunk )
                        total_rows +=len (chunk )
                        total_chunks =i +1 
                        break 
                    else :
                        chunks .append (chunk )
                        total_rows +=len (chunk )
                elif data_usage_ratio <1.0 :
                    sample_size =int (len (chunk )*data_usage_ratio )
                    if sample_size >0 :
                        chunk =chunk .sample (n =min (sample_size ,len (chunk )),random_state =42 )
                        chunks .append (chunk )
                        total_rows +=len (chunk )
                else :
                    chunks .append (chunk )
                    total_rows +=len (chunk )
                total_chunks =i +1 
                if total_chunks %3 ==0 :
                    elapsed_time =time .time ()-start_time 
                    if elapsed_time >0 :
                        rows_per_sec =total_rows /elapsed_time 
                        print (f"[Client {args .client_id }] 已讀取 {total_rows :,} 行 (約 {total_rows /1000000 :.1f}M 行, {rows_per_sec /1000 :.1f}K 行/秒)...")
                    else :
                        print (f"[Client {args .client_id }] 已讀取 {total_rows :,} 行 (約 {total_rows /1000000 :.1f}M 行)...")
                    sys .stdout .flush ()
                if max_test_samples and total_rows >=max_test_samples :
                    break 
            if chunks :
                print (f"[Client {args .client_id }] 正在合併 {total_chunks } 個數據塊...")
                sys .stdout .flush ()
                test_df_full =pd .concat (chunks ,ignore_index =True )
                total_time =time .time ()-start_time 
                print (f"[Client {args .client_id }] 測試數據讀取完成: {len (test_df_full ):,} 行 (共 {total_chunks } 個塊, 耗時 {total_time :.1f}秒)")
                if max_test_samples and len (test_df_full )>max_test_samples :
                    try :
                        from sklearn .model_selection import train_test_split 
                        test_df ,_ =train_test_split (test_df_full ,train_size =max_test_samples ,stratify =test_df_full ['label'],random_state =42 )
                        print (f"[Client {args .client_id }] 測試數據分層採樣完成: {len (test_df )} 行 (從 {len (test_df_full ):,} 行中採樣)")
                    except Exception as stratify_e :
                        print (f"[Client {args .client_id }] 分層採樣失敗，改用隨機採樣: {stratify_e }")
                        test_df =test_df_full .sample (n =max_test_samples ,random_state =42 )
                        print (f"[Client {args .client_id }] 測試數據隨機採樣完成: {len (test_df )} 行")
                elif data_usage_ratio <1.0 :
                    sample_size =int (len (test_df_full )*data_usage_ratio )
                    try :
                        from sklearn .model_selection import train_test_split 
                        test_df ,_ =train_test_split (test_df_full ,train_size =sample_size ,stratify =test_df_full ['label'],random_state =42 )
                        print (f"[Client {args .client_id }] 測試數據分層採樣完成: {len (test_df )} 行 (從 {len (test_df_full ):,} 行中採樣 {data_usage_ratio *100 :.1f}%)")
                    except Exception as stratify_e :
                        test_df =test_df_full .sample (n =sample_size ,random_state =42 )
                        print (f"[Client {args .client_id }] 測試數據隨機採樣完成: {len (test_df )} 行")
                else :
                    test_df =test_df_full 
            else :
                test_df =pd .DataFrame ()
                print (f"[Client {args .client_id }] 測試數據為空")
        else :
            if max_test_samples :
                test_df_full =pd .read_csv (test_path )
                if len (test_df_full )>max_test_samples :
                    try :
                        from sklearn .model_selection import train_test_split 
                        test_df ,_ =train_test_split (test_df_full ,train_size =max_test_samples ,stratify =test_df_full ['label'],random_state =42 )
                        print (f"[Client {args .client_id }] 測試數據讀取完成: {len (test_df )} 行 (分層採樣，限制為最多 {max_test_samples } 行)")
                    except Exception as stratify_e :
                        print (f"[Client {args .client_id }] 分層採樣失敗，改用隨機採樣: {stratify_e }")
                        test_df =test_df_full .sample (n =max_test_samples ,random_state =42 )
                        print (f"[Client {args .client_id }] 測試數據讀取完成: {len (test_df )} 行 (隨機採樣，限制為最多 {max_test_samples } 行)")
                else :
                    test_df =test_df_full 
                    print (f"[Client {args .client_id }] 測試數據讀取完成: {len (test_df )} 行 (數據量少於限制，使用全部)")
            elif data_usage_ratio <1.0 :
                test_df_full =pd .read_csv (test_path )
                sample_size =int (len (test_df_full )*data_usage_ratio )
                try :
                    from sklearn .model_selection import train_test_split 
                    test_df ,_ =train_test_split (test_df_full ,train_size =sample_size ,stratify =test_df_full ['label'],random_state =42 )
                    print (f"[Client {args .client_id }] 測試數據讀取完成: {len (test_df )} 行 (分層採樣，從 {len (test_df_full )} 行中採樣 {data_usage_ratio *100 :.1f}%)")
                except Exception as stratify_e :
                    print (f"[Client {args .client_id }] 分層採樣失敗，改用隨機採樣: {stratify_e }")
                    test_df =test_df_full .sample (n =sample_size ,random_state =42 )
                    print (f"[Client {args .client_id }] 測試數據讀取完成: {len (test_df )} 行 (隨機採樣，從 {len (test_df_full )} 行中採樣 {data_usage_ratio *100 :.1f}%)")
            else :
                test_df =pd .read_csv (test_path )
                print (f"[Client {args .client_id }] 測試數據讀取完成: {len (test_df )} 行")
        sys .stdout .flush ()
    except Exception as e :
        error_msg =f" [Client {args .client_id }] 讀取測試數據失敗: {e }\n"
        import traceback 
        error_msg +=f"Traceback: {traceback .format_exc ()}\n"
        print (error_msg )
        sys .stdout .flush ()
        if args .result_dir .strip ():
            try :
                client_log_path =os .path .join (args .result_dir .strip (),f"uav{args .client_id }",f"client_{args .client_id }.log")
                with open (client_log_path ,'a',encoding ='utf-8')as f :
                    f .write (error_msg )
            except Exception :
                pass 
        sys .exit (1 )
    print (f"[Client {args .client_id }] 開始分離特徵和標籤...")
    sys .stdout .flush ()
    feature_cols =[col for col in train_df .columns if col !='label']
    if 'label'in train_df .columns :
        label_counts =train_df ['label'].value_counts ().sort_index ()
        print (f"[Client {args .client_id }] 訓練數據標籤分佈: {dict (label_counts )}")
        print (f"[Client {args .client_id }] 訓練數據總樣本數: {len (train_df )}, 類別數: {len (label_counts )}")
        sys .stdout .flush ()
    data_usage_ratio =config .DATA_CONFIG .get ('data_usage_ratio',1.0 )
    if data_usage_ratio <1.0 and not config .DATA_CONFIG .get ('max_train_samples',None ):
        sample_size =int (len (train_df )*data_usage_ratio )
        train_df =train_df .sample (n =sample_size ,random_state =42 )
        print (f"[Client {args .client_id }] 訓練數據採樣: {len (train_df )} 行 (使用 {data_usage_ratio *100 :.1f}%)")
        sys .stdout .flush ()
    try :
        max_eval_samples =int (os .environ .get ("CLIENT_EVAL_MAX_SAMPLES","0"))
    except Exception :
        max_eval_samples =0 
    if max_eval_samples >0 and len (test_df )>max_eval_samples :
        try :
            from sklearn .model_selection import train_test_split 
            _ ,test_df =train_test_split (test_df ,train_size =max_eval_samples ,stratify =test_df ['label'],random_state =42 )
            print (f"[Client {args .client_id }]驗證集下採樣: {len (test_df )} 行 (分層)")
        except Exception as e :
            test_df =test_df .sample (n =max_eval_samples ,random_state =42 )
            print (f"[Client {args .client_id }]驗證集下採樣: {len (test_df )} 行 (隨機, {e })")
        sys .stdout .flush ()
    X_train_raw =train_df [feature_cols ].values .astype (np .float32 )
    X_val_raw =test_df [feature_cols ].values .astype (np .float32 )
    use_robust_scaler =os .environ .get ("USE_ROBUST_SCALER","0").strip ().lower ()in ("1","true","yes","on")
    if use_robust_scaler :
        scaler =RobustScaler ()
        X_train =scaler .fit_transform (X_train_raw ).astype (np .float32 )
        X_val =scaler .transform (X_val_raw ).astype (np .float32 )
        print (f"[Client {args .client_id }] 已應用特徵標準化（RobustScaler - 對離群值更穩健）")
    else :
        scaler =StandardScaler ()
        X_train =scaler .fit_transform (X_train_raw ).astype (np .float32 )
        X_val =scaler .transform (X_val_raw ).astype (np .float32 )
        print (f"[Client {args .client_id }] 已應用特徵標準化（StandardScaler）")
    use_l2_normalization =os .environ .get ("USE_L2_NORMALIZATION","0").strip ().lower ()in ("1","true","yes","on")
    if use_l2_normalization :
        train_norms =np .linalg .norm (X_train ,axis =1 ,keepdims =True )+1e-7 
        val_norms =np .linalg .norm (X_val ,axis =1 ,keepdims =True )+1e-7 
        X_train =(X_train /train_norms ).astype (np .float32 )
        X_val =(X_val /val_norms ).astype (np .float32 )
        print (f"[Client {args .client_id }] 已應用 L2 Normalization（進一步穩定特徵量級）")
    print (f"[Client {args .client_id }]   標準化前 - 均值範圍: [{X_train_raw .mean (axis =0 ).min ():.4f}, {X_train_raw .mean (axis =0 ).max ():.4f}]")
    print (f"[Client {args .client_id }]   標準化前 - 標準差範圍: [{X_train_raw .std (axis =0 ).min ():.4f}, {X_train_raw .std (axis =0 ).max ():.4f}]")
    print (f"[Client {args .client_id }]   標準化後 - 均值範圍: [{X_train .mean (axis =0 ).min ():.4f}, {X_train .mean (axis =0 ).max ():.4f}]")
    print (f"[Client {args .client_id }]   標準化後 - 標準差範圍: [{X_train .std (axis =0 ).min ():.4f}, {X_train .std (axis =0 ).max ():.4f}]")
    sys .stdout .flush ()
    y_train_raw =train_df ['label'].values 
    y_val_raw =test_df ['label'].values 
    unique_labels =np .unique (np .concatenate ([y_train_raw ,y_val_raw ]))
    num_classes =len (unique_labels )
    label_mapping ={old_label :new_label for new_label ,old_label in enumerate (unique_labels )}
    print (f"[Client {args .client_id }]標籤映射: {label_mapping }")
    sys .stdout .flush ()
    y_train =np .array ([label_mapping [label ]for label in y_train_raw ])
    y_val =np .array ([label_mapping [label ]for label in y_val_raw ])
    loss_config =getattr (config ,'LOSS_CONFIG',{})
    class_weights_enabled =loss_config .get ('class_weights_enabled',False )
    label_smoothing =float (loss_config .get ('label_smoothing',0.0 ))
    max_class_weight =float (loss_config .get ('max_class_weight',3.0 ))
    print (f"[Client {args .client_id }]損失函數配置: class_weights_enabled={class_weights_enabled }, label_smoothing={label_smoothing }, max_class_weight={max_class_weight }")
    class_weights_tensor =None 
    class_weights_dict =None 
    if class_weights_enabled :
        from sklearn .utils .class_weight import compute_class_weight 
        try :
            unique_classes =np .unique (y_train )
            print (f"[Client {args .client_id }] 訓練數據中實際存在的類別: {unique_classes .tolist ()}")
            if len (unique_classes )>1 :
                class_weight_strategy =loss_config .get ('class_weight_strategy','balanced')
                min_class_weight =float (loss_config .get ('min_class_weight',1.0 ))
                class_counts ={}
                for cls in unique_classes :
                    class_counts [int (cls )]=int (np .sum (y_train ==cls ))
                print (f"[Client {args .client_id }] 類別樣本數: {class_counts }")
                if class_weight_strategy =='balanced':
                    class_weights_partial =compute_class_weight ('balanced',classes =unique_classes ,y =y_train )
                elif class_weight_strategy =='inverse_frequency':
                    total_samples =len (y_train )
                    n_classes_present =len (unique_classes )
                    class_weights_partial =[]
                    for cls in unique_classes :
                        count =class_counts [int (cls )]
                        weight =total_samples /(n_classes_present *count )if count >0 else 1.0 
                        class_weights_partial .append (weight )
                    class_weights_partial =np .array (class_weights_partial )
                elif class_weight_strategy =='sqrt_inverse_frequency':
                    total_samples =len (y_train )
                    n_classes_present =len (unique_classes )
                    class_weights_partial =[]
                    for cls in unique_classes :
                        count =class_counts [int (cls )]
                        weight =np .sqrt (total_samples /(n_classes_present *count ))if count >0 else 1.0 
                        class_weights_partial .append (weight )
                    class_weights_partial =np .array (class_weights_partial )
                elif class_weight_strategy =='adaptive':
                    total_samples =len (y_train )
                    n_classes_present =len (unique_classes )
                    max_count =max (class_counts .values ())
                    min_count =min (class_counts .values ())
                    imbalance_ratio =max_count /min_count if min_count >0 else 1.0 
                    print (f"[Client {args .client_id }] 類別不平衡比例: {imbalance_ratio :.2f}")
                    if imbalance_ratio >10.0 :
                        class_weights_partial =compute_class_weight ('balanced',classes =unique_classes ,y =y_train )
                    elif imbalance_ratio >5.0 :
                        class_weights_partial =[]
                        for cls in unique_classes :
                            count =class_counts [int (cls )]
                            weight =total_samples /(n_classes_present *count )if count >0 else 1.0 
                            class_weights_partial .append (weight )
                        class_weights_partial =np .array (class_weights_partial )
                    else :
                        class_weights_partial =[]
                        for cls in unique_classes :
                            count =class_counts [int (cls )]
                            weight =np .sqrt (total_samples /(n_classes_present *count ))if count >0 else 1.0 
                            class_weights_partial .append (weight )
                        class_weights_partial =np .array (class_weights_partial )
                elif class_weight_strategy =='adaptive_enhanced':
                    total_samples =len (y_train )
                    n_classes_present =len (unique_classes )
                    max_count =max (class_counts .values ())
                    min_count =min (class_counts .values ())
                    imbalance_ratio =max_count /min_count if min_count >0 else 1.0 
                    rare_class_threshold =float (loss_config .get ('rare_class_threshold',0.1 ))
                    rare_class_boost =float (loss_config .get ('rare_class_boost',1.5 ))
                    print (f"[Client {args .client_id }] 類別不平衡比例: {imbalance_ratio :.2f}")
                    if imbalance_ratio >10.0 :
                        base_weights =compute_class_weight ('balanced',classes =unique_classes ,y =y_train )
                    elif imbalance_ratio >5.0 :
                        base_weights =[]
                        for cls in unique_classes :
                            count =class_counts [int (cls )]
                            weight =total_samples /(n_classes_present *count )if count >0 else 1.0 
                            base_weights .append (weight )
                        base_weights =np .array (base_weights )
                    else :
                        base_weights =[]
                        for cls in unique_classes :
                            count =class_counts [int (cls )]
                            weight =np .sqrt (total_samples /(n_classes_present *count ))if count >0 else 1.0 
                            base_weights .append (weight )
                        base_weights =np .array (base_weights )
                    class_weights_partial =[]
                    for idx ,cls in enumerate (unique_classes ):
                        count =class_counts [int (cls )]
                        class_ratio =count /total_samples if total_samples >0 else 0.0 
                        base_weight =base_weights [idx ]
                        if class_ratio <rare_class_threshold :
                            enhanced_weight =base_weight *rare_class_boost 
                            print (f"[Client {args .client_id }]類別 {cls } 為極少數類別 (比例={class_ratio :.4f})，權重增強: {base_weight :.3f} -> {enhanced_weight :.3f}")
                            class_weights_partial .append (enhanced_weight )
                        else :
                            class_weights_partial .append (base_weight )
                    class_weights_partial =np .array (class_weights_partial )
                else :
                    class_weights_partial =compute_class_weight ('balanced',classes =unique_classes ,y =y_train )
                class_weights =np .ones (num_classes ,dtype =np .float32 )
                for cls ,weight in zip (unique_classes ,class_weights_partial ):
                    class_weights [int (cls )]=float (weight )
                class_weights =np .clip (class_weights ,min_class_weight ,max_class_weight )
                zero_f1_boost =float (loss_config .get ('zero_f1_boost',1.0 ))
                if zero_f1_boost >1.0 :
                    train_total_samples =len (y_train )
                    benign_ratio =class_counts .get (0 ,0 )/train_total_samples if train_total_samples >0 else 0.0 
                    ddos_ratio =class_counts .get (1 ,0 )/train_total_samples if train_total_samples >0 else 0.0 
                    if 0 in unique_classes and class_weights [0 ]<max_class_weight *0.8 :
                        if benign_ratio <0.05 :
                            boost_factor =zero_f1_boost 
                        elif benign_ratio <0.10 :
                            boost_factor =1.0 +(zero_f1_boost -1.0 )*0.7 
                        else :
                            boost_factor =1.0 +(zero_f1_boost -1.0 )*0.4 
                        old_weight =class_weights [0 ]
                        new_weight =min (max_class_weight ,class_weights [0 ]*boost_factor )
                        class_weights [0 ]=new_weight 
                        print (f"[Client {args .client_id }]BENIGN 類別權重提升: {old_weight :.3f} -> {new_weight :.3f} (boost={boost_factor :.2f}, ratio={benign_ratio :.4f})")
                    if 1 in unique_classes and class_weights [1 ]<max_class_weight *0.8 :
                        if ddos_ratio <0.05 :
                            boost_factor =zero_f1_boost 
                        elif ddos_ratio <0.10 :
                            boost_factor =1.0 +(zero_f1_boost -1.0 )*0.7 
                        else :
                            boost_factor =1.0 +(zero_f1_boost -1.0 )*0.4 
                        old_weight =class_weights [1 ]
                        new_weight =min (max_class_weight ,class_weights [1 ]*boost_factor )
                        class_weights [1 ]=new_weight 
                        print (f"[Client {args .client_id }]DDoS 類別權重提升: {old_weight :.3f} -> {new_weight :.3f} (boost={boost_factor :.2f}, ratio={ddos_ratio :.4f})")
                min_guaranteed_weight =float (loss_config .get ('min_class_weight',1.0 ))*1.2 
                for cls_idx in range (num_classes ):
                    if class_weights [cls_idx ]<min_guaranteed_weight :
                        old_weight =class_weights [cls_idx ]
                        class_weights [cls_idx ]=min_guaranteed_weight 
                        if old_weight <min_guaranteed_weight *0.8 :
                            print (f"[Client {args .client_id }]類別 {cls_idx } 權重提升到最低保證: {old_weight :.3f} -> {class_weights [cls_idx ]:.3f}")
                class_weights =np .clip (class_weights ,min_class_weight ,max_class_weight )
                weight_mean =float (np .mean (class_weights ))
                weight_std =float (np .std (class_weights ))
                weight_min =float (np .min (class_weights ))
                weight_max =float (np .max (class_weights ))
                class_weights_dict ={i :float (weight )for i ,weight in enumerate (class_weights )}
                print (f"[Client {args .client_id }]類別權重策略: {class_weight_strategy }")
                print (f"[Client {args .client_id }]類別權重統計: mean={weight_mean :.3f}, std={weight_std :.3f}, min={weight_min :.3f}, max={weight_max :.3f}")
                print (f"[Client {args .client_id }]類別權重（已限制範圍 [{min_class_weight }, {max_class_weight }]）: {class_weights_dict }")
                class_weights_tensor =torch .tensor (class_weights ,dtype =torch .float32 )
                print (f"[Client {args .client_id }]類別權重 tensor: {class_weights_tensor .tolist ()}")
            else :
                print (f"[Client {args .client_id }] 訓練數據只有一個類別，無法計算類別權重，使用均勻權重")
                class_weights_tensor =None 
                class_weights_dict =None 
        except Exception as e :
            print (f"[Client {args .client_id }] 計算類別權重失敗: {e }，使用均勻權重")
            traceback .print_exc ()
            class_weights_tensor =None 
            class_weights_dict =None 
    else :
        print (f"[Client {args .client_id }]類別權重已禁用（LOSS_CONFIG['class_weights_enabled']=False）")
    print (f"[Client {args .client_id }] 預處理數據載入完成:")
    print (f"  - 訓練數據: {len (X_train )} 樣本")
    print (f"  - 驗證數據: {len (X_val )} 樣本")
    print (f"  - 特徵維度: {X_train .shape [1 ]}")
    print (f"  - 類別數: {num_classes }")
    print (f"  - 原始標籤: {unique_labels }")
    print (f"  - 映射後標籤範圍: [{y_train .min ()}, {y_train .max ()}]")
    print (f"Loaded data for client {args .client_id }: {len (X_train )} train, {len (X_val )} val")
    train_dataset =TensorDataset (torch .tensor (X_train ,dtype =torch .float32 ),torch .tensor (y_train ,dtype =torch .long ))
    val_dataset =TensorDataset (torch .tensor (X_val ,dtype =torch .float32 ),torch .tensor (y_val ,dtype =torch .long ))
    train_batch_size =int (getattr (config ,"TRAIN_BATCH_SIZE",getattr (config ,"BATCH_SIZE",64 )))
    eval_batch_size =int (getattr (config ,"EVAL_BATCH_SIZE",train_batch_size ))
    print (f"[Client {args .client_id }]批次大小設定: train={train_batch_size }, eval={eval_batch_size }")
    use_balanced_sampler =True 
    if use_balanced_sampler :
        try :
            class_counts =pd .Series (y_train ).value_counts ().sort_index ()
            total_samples =len (y_train )
            class_weights ={}
            for cls in range (num_classes ):
                count =class_counts .get (cls ,1 )
                if cls in [0 ,1 ]:
                    min_batch_ratio =0.10 
                    target_count =max (1 ,int (train_batch_size *min_batch_ratio ))
                    class_weights [cls ]=total_samples /max (count ,target_count )*2.0 
                else :
                    class_weights [cls ]=total_samples /max (count ,1 )
            sample_weights =[class_weights [y_train [i ]]for i in range (len (y_train ))]
            sampler =WeightedRandomSampler (weights =sample_weights ,num_samples =len (sample_weights ),replacement =True )
            train_loader =DataLoader (train_dataset ,batch_size =train_batch_size ,sampler =sampler ,drop_last =True )
            print (f"[Client {args .client_id }] 破局配置 3.6：已啟用平衡採樣器（類別 0 和 1 權重={class_weights .get (0 ,1.0 ):.2f}）")
        except Exception as e :
            print (f"[Client {args .client_id }] 平衡採樣器創建失敗: {e }，回退到隨機採樣")
            train_loader =DataLoader (train_dataset ,batch_size =train_batch_size ,shuffle =True ,drop_last =True )
    else :
        train_loader =DataLoader (train_dataset ,batch_size =train_batch_size ,shuffle =True ,drop_last =True )
    val_loader =DataLoader (val_dataset ,batch_size =eval_batch_size ,shuffle =False ,drop_last =False )
    input_dim =len (feature_cols )
    class_counts =pd .Series (y_train ).value_counts ().sort_index ()
    class_proportions =[class_counts .get (i ,0 )/len (y_train )for i in range (num_classes )]
    print (f"[Client {args .client_id }] 破局配置 3.6：類別比例: {dict (zip (range (num_classes ),class_proportions ))}")
    try :
        print (f"[Client {args .client_id }]  構建模型: input_dim={input_dim }, num_classes={num_classes }")
        sys .stdout .flush ()
        model =build_model_from_config (input_dim ,num_classes ).to (device )
        if hasattr (model ,'client_model'):
            model .client_model ._class_proportions =class_proportions 
            print (f"[Client {args .client_id }] 破局配置 3.6：已設置類別比例到 ClientModel")
        elif hasattr (model ,'_class_proportions'):
            model ._class_proportions =class_proportions 
            print (f"[Client {args .client_id }] 破局配置 3.6：已設置類別比例到模型")
        with torch .no_grad ():
            total_norm =sum (p .norm ().item ()for p in model .parameters ())
            all_params =torch .cat ([p .flatten ()for p in model .parameters ()])
            weight_mean =all_params .mean ().item ()
            weight_std =all_params .std ().item ()
            weight_min =all_params .min ().item ()
            weight_max =all_params .max ().item ()
            print (f"[Client {args .client_id }] 模型初始化權重統計:")
            print (f"  - 總範數: {total_norm :.4f}")
            print (f"  - 均值: {weight_mean :.6f}")
            print (f"  - 標準差: {weight_std :.6f}")
            print (f"  - 範圍: [{weight_min :.6f}, {weight_max :.6f}]")
            output_layer_weight_norm =None 
            output_layer_bias_norm =None 
            for name ,param in model .named_parameters ():
                if 'output_layer'in name and 'weight'in name :
                    output_layer_weight_norm =param .norm ().item ()
                    print (f"[Client {args .client_id }] 輸出層權重範數: {output_layer_weight_norm :.6f}")
                elif 'output_layer'in name and 'bias'in name :
                    output_layer_bias_norm =param .norm ().item ()
                    print (f"[Client {args .client_id }] 輸出層偏置範數: {output_layer_bias_norm :.6f}")
            if total_norm <1.0 :
                print (f"[Client {args .client_id }] 警告：模型權重總範數過小 ({total_norm :.4f})，可能導致輸出接近零")
            if abs (weight_mean )>0.1 :
                print (f"[Client {args .client_id }] 警告：模型權重均值偏離零 ({weight_mean :.6f})，可能影響初始化效果")
            if weight_std <0.01 :
                print (f"[Client {args .client_id }] 警告：模型權重標準差過小 ({weight_std :.6f})，可能導致梯度消失")
            if output_layer_bias_norm is not None and output_layer_bias_norm <0.15 :
                print (f"[Client {args .client_id }] 警告：輸出層偏置範數過小 ({output_layer_bias_norm :.6f})，可能導致模型性能差")
                for name ,param in model .named_parameters ():
                    if 'output_layer'in name and 'bias'in name :
                        fan_in =param .size (0 )if len (param .shape )>0 else 1 
                        bound =max (0.2 ,1.0 /math .sqrt (fan_in )if fan_in >0 else 0.2 )
                        nn .init .uniform_ (param ,-bound ,bound )
                        new_norm =param .norm ().item ()
                        print (f"[Client {args .client_id }] 已重新初始化輸出層偏置: 新範數={new_norm :.6f}")
        print (f"[Client {args .client_id }] 模型構建成功")
        sys .stdout .flush ()
    except Exception as e :
        import traceback 
        error_msg =f" [Client {args .client_id }] 模型構建失敗: {e }\n"
        import traceback 
        error_msg +=f"Traceback: {traceback .format_exc ()}\n"
        print (error_msg )
        sys .stdout .flush ()
        if args .result_dir .strip ():
            try :
                client_log_path =os .path .join (args .result_dir .strip (),f"uav{args .client_id }",f"client_{args .client_id }.log")
                with open (client_log_path ,'a',encoding ='utf-8')as f :
                    f .write (error_msg )
            except Exception :
                pass 
        sys .exit (1 )
    global_class_weight_cfg =getattr (config ,"CLASS_WEIGHT_CONFIG",{})
    use_global_weights =False 
    if class_weights_tensor is None and global_class_weight_cfg .get ("enabled",False ):
        try :
            import numpy as _np 
            cw_path =None 
            if args .result_dir .strip ():
                exp_cw_path =os .path .join (args .result_dir .strip (),"model","global_class_weights.npy")
                if os .path .exists (exp_cw_path ):
                    cw_path =exp_cw_path 
                    print (f"[Client {args .client_id }] 從實驗目錄讀取全局類別權重（備選）: {exp_cw_path }")
            if cw_path is None or not os .path .exists (cw_path ):
                cw_path =global_class_weight_cfg .get ("path",os .path .join ("model","global_class_weights.npy"))
                if not os .path .isabs (cw_path ):
                    if args .result_dir .strip ():
                        exp_cw_path =os .path .join (args .result_dir .strip (),cw_path )
                        if os .path .exists (exp_cw_path ):
                            cw_path =exp_cw_path 
                        else :
                            cw_path =os .path .join (os .path .dirname (os .path .abspath (__file__ )),cw_path )
                    else :
                        cw_path =os .path .join (os .path .dirname (os .path .abspath (__file__ )),cw_path )
            if os .path .exists (cw_path ):
                cw_obj =_np .load (cw_path ,allow_pickle =True ).item ()
                weights =_np .asarray (cw_obj .get ("weights"),dtype =_np .float32 )
                if global_class_weight_cfg .get ("normalize",True ):
                    weights =weights /(weights .mean ()+1e-8 )
                if len (weights )!=num_classes :
                    print (f"[Client {args .client_id }] 全局類別權重長度 ({len (weights )}) 與實際類別數 ({num_classes }) 不匹配，調整權重")
                    if len (weights )>num_classes :
                        weights =weights [:num_classes ]
                        print (f"[Client {args .client_id }]截斷權重到 {num_classes } 個類別: {weights .tolist ()}")
                    else :
                        padded_weights =np .ones (num_classes ,dtype =np .float32 )
                        padded_weights [:len (weights )]=weights 
                        weights =padded_weights 
                        print (f"[Client {args .client_id }]填充權重到 {num_classes } 個類別: {weights .tolist ()}")
                class_weights_tensor =torch .tensor (weights ,dtype =torch .float32 )
                use_global_weights =True 
                print (f"[Client {args .client_id }] 使用全域 class weights（備選）: {weights .tolist ()}")
            else :
                print (f"[Client {args .client_id }] 找不到全域 class weights 檔案: {cw_path }，使用均勻權重")
        except Exception as e :
            print (f"[Client {args .client_id }] 載入全域 class weights 失敗，使用均勻權重: {e }")
    if class_weights_tensor is not None and not use_global_weights :
        print (f"[Client {args .client_id }] 優先使用本地計算的類別權重（基於本地訓練數據分佈）")
    if class_weights_tensor is not None :
        device =next (model .parameters ()).device 
        class_weights_tensor =class_weights_tensor .to (device )
        dyn_w =_load_dynamic_class_weights_from_cloud (args .result_dir .strip (),num_classes ,device )
        if dyn_w is not None :
            class_weights_tensor =dyn_w 
            print (f"[Client {args .client_id }] 已套用 Cloud 動態 class_weight（覆寫本地/全域權重）")
        criterion =build_loss (class_weights =class_weights_tensor ,label_smoothing =label_smoothing )
        print (f"[Client {args .client_id }]使用類別權重損失函數 (label_smoothing={label_smoothing })")
    else :
        device =next (model .parameters ()).device 
        dyn_w =_load_dynamic_class_weights_from_cloud (args .result_dir .strip (),num_classes ,device )
        if dyn_w is not None :
            class_weights_tensor =dyn_w 
            criterion =build_loss (class_weights =class_weights_tensor ,label_smoothing =label_smoothing )
            print (f"[Client {args .client_id }]使用 Cloud 動態類別權重損失函數 (label_smoothing={label_smoothing })")
        else :
            criterion =build_loss (label_smoothing =label_smoothing )
            print (f"[Client {args .client_id }]使用標準損失函數（無類別權重，label_smoothing={label_smoothing }）")
    learning_rate =getattr (config ,'LEARNING_RATE',0.01 )
    if learning_rate <1e-5 :
        print (f"[Client {args .client_id }] 警告：學習率過小 ({learning_rate })，可能導致訓練緩慢")
    elif learning_rate >0.1 :
        print (f"[Client {args .client_id }] 警告：學習率過大 ({learning_rate })，可能導致訓練不穩定")
    weight_decay =float (os .environ .get ("WEIGHT_DECAY",getattr (config ,"WEIGHT_DECAY",1e-4 )))
    optimizer =optim .Adam (model .parameters (),lr =learning_rate ,weight_decay =weight_decay )
    print (f"[Client {args .client_id }]優化器配置: lr={learning_rate }, weight_decay={weight_decay }")
    for group in optimizer .param_groups :
        group .setdefault ('initial_lr',learning_rate )
    lr_scheduler_config =getattr (config ,'LEARNING_RATE_SCHEDULER',{})
    scheduler_mode =lr_scheduler_config .get ('mode','').lower ()
    manual_round_scheduler =lr_scheduler_config .get ('enabled',False )and scheduler_mode in ("manual","round")
    if lr_scheduler_config .get ('enabled',False )and not manual_round_scheduler :
        scheduler_type =lr_scheduler_config .get ('scheduler_type','step')
        if scheduler_type =='cosine':
            from torch .optim .lr_scheduler import CosineAnnealingLR 
            max_epochs =getattr (config ,'MAX_ROUNDS',500 )*getattr (config ,'LOCAL_EPOCHS',1 )
            scheduler =CosineAnnealingLR (optimizer ,T_max =max_epochs ,eta_min =lr_scheduler_config .get ('min_lr',1e-4 ))
            print (f"[Client {args .client_id }]使用 CosineAnnealingLR 學習率調度器 (T_max={max_epochs }, eta_min={lr_scheduler_config .get ('min_lr',1e-4 )})")
        elif scheduler_type =='step':
            from torch .optim .lr_scheduler import StepLR 
            step_size =lr_scheduler_config .get ('step_size',5 )
            gamma =lr_scheduler_config .get ('gamma',0.8 )
            scheduler =StepLR (optimizer ,step_size =step_size ,gamma =gamma )
            print (f"[Client {args .client_id }]使用 StepLR 學習率調度器 (step_size={step_size }, gamma={gamma })")
        else :
            scheduler =None 
            print (f"[Client {args .client_id }] 未知的學習率調度器類型: {scheduler_type }")
    else :
        scheduler =None 
        if manual_round_scheduler :
            print (f"[Client {args .client_id }]啟用 round-aware 學習率調整（mode={scheduler_mode or 'manual'}）")
    print (f"[Client {args .client_id }]使用學習率: {learning_rate }")
    print (f"[Client {args .client_id }] 客戶端初始化完成，準備連接聚合器...")
    sys .stdout .flush ()
    last_attempted_round =-1 
    last_confirmed_round =-1 
    idle_ticks =0 
    async with aiohttp .ClientSession ()as session :
        await register_client (session ,args .aggregator_url ,args .client_id )
        while True :
            try :
                max_retries =3 
                status =None 
                for retry in range (max_retries ):
                    status =await get_federated_status (session ,args .aggregator_url )
                    if status and status .get ('current_round',status .get ('round_count',0 ))>0 :
                        break 
                    if retry <max_retries -1 :
                        await asyncio .sleep (0.5 )
                if not status :
                    print (f"[Client {args .client_id }] 無法獲取 federated_status，等待後重試...")
                    await asyncio .sleep (2 )
                    continue 
                current_round =int (status .get ('current_round',status .get ('round_count',0 )))
                selected_raw =status .get ('selected_clients',[])or []
                selected =set (int (x )for x in selected_raw )
                is_selected =args .client_id in selected 
                print (f"[Client {args .client_id }] Round {current_round }: selected={is_selected }, selected_clients={selected_raw }")
                if is_selected :
                    print (f"[Client {args .client_id }] 被選中參與第 {current_round } 輪訓練")
                else :
                    print (f"[Client {args .client_id }]  未被選中（選中列表: {selected_raw }）")
                import sys 
                sys .stdout .flush ()
                if current_round >last_confirmed_round +1 :
                    print (f"[Client {args .client_id }] 輪次落後過多（{last_confirmed_round } → {current_round }），立即同步")
                    last_confirmed_round =current_round -1 
                should_train =False 
                training_round =current_round 
                print (f"[Client {args .client_id }] 輪次檢查: current_round={current_round }, last_confirmed_round={last_confirmed_round }, last_attempted_round={last_attempted_round }, is_selected={is_selected }")
                max_rounds =getattr (config ,'MAX_ROUNDS',500 )
                if current_round >max_rounds :
                    print (f" 輪次 {current_round } 超過最大輪數 {max_rounds }，停止訓練")
                    should_train =False 
                    training_round =current_round 
                    break 
                if is_selected and (current_round >last_confirmed_round or (current_round ==1 and last_confirmed_round ==-1 )):
                    should_train =True 
                    training_round =current_round 
                    print (f"[Client {args .client_id }] 安全訓練：聚合器輪次 {current_round } > 已確認輪次 {last_confirmed_round }")
                elif is_selected and current_round ==last_confirmed_round :
                    if current_round ==1 or current_round ==0 :
                        should_train =True 
                        training_round =current_round if current_round >0 else 1 
                        print (f"[Client {args .client_id }] 第1輪訓練：聚合器輪次 {current_round } = 已確認輪次 {last_confirmed_round }")
                    elif last_attempted_round <current_round :
                        should_train =True 
                        training_round =current_round 
                        print (f"[Client {args .client_id }] 補訓練：聚合器輪次 {current_round } = 已確認輪次 {last_confirmed_round }，但 last_attempted_round={last_attempted_round } < current_round，需要訓練")
                    else :
                        should_train =False 
                        print (f"[Client {args .client_id }]  等待輪次推進：聚合器輪次 {current_round } = 已確認輪次 {last_confirmed_round }，且已訓練過（last_attempted_round={last_attempted_round }）")
                elif is_selected and current_round <last_confirmed_round :
                    last_confirmed_round =current_round -1 
                    should_train =True 
                    training_round =current_round 
                    print (f"[Client {args .client_id }]輪次重置並訓練：客戶端輪次超前，重置到 {last_confirmed_round }")
                else :
                    should_train =False 
                    if is_selected :
                        print (f"[Client {args .client_id }]  被選中但無需訓練：聚合器輪次 {current_round } <= 已確認輪次 {last_confirmed_round }")
                    else :
                        print (f"[Client {args .client_id }]  未被選中：聚合器輪次 {current_round }（選中列表: {selected_raw }）")
                print (f" 最終決定: should_train={should_train }, training_round={training_round }")
                import sys 
                sys .stdout .flush ()
                if should_train :
                    client_kd_cfg =getattr (config ,"CLIENT_KD_CONFIG",{})or {}
                    client_kd_enabled =bool (client_kd_cfg .get ("enabled",False ))
                    print (f"[Client {args .client_id }] GKD 配置檢查: CLIENT_KD_CONFIG={client_kd_cfg }, enabled={client_kd_enabled }")
                    needs_global =(not hasattr (model ,'_has_global_weights'))or (not model ._has_global_weights )
                    has_server_weights =hasattr (model ,'_has_server_weights')and model ._has_server_weights 
                    current_server_version =getattr (model ,'_server_weights_version',None )
                    needs_server =client_kd_enabled and (not has_server_weights or current_server_version is None )
                    check_server_update =client_kd_enabled and has_server_weights and current_server_version is not None 
                    if client_kd_enabled :
                        print (f"[Client {args .client_id }] GKD 狀態檢查: enabled={client_kd_enabled }, has_server_weights={has_server_weights }, current_version={current_server_version }, needs_server={needs_server }, check_update={check_server_update }")
                    else :
                        print (f"[Client {args .client_id }] GKD 未啟用: client_kd_enabled={client_kd_enabled }, CLIENT_KD_CONFIG={client_kd_cfg }")
                    if needs_global or needs_server or check_server_update :
                        if needs_global :
                            print (f"[Client {args .client_id }] 模型尚未載入全局權重，嘗試獲取...")
                        if needs_server :
                            print (f"[Client {args .client_id }] GKD 已啟用，嘗試補拉 server_weights...")
                        if check_server_update :
                            print (f"[Client {args .client_id }] GKD 已啟用，檢查 server_weights 版本更新（當前 v={current_server_version }）...")
                        weights_payload =await get_global_weights (session ,args .aggregator_url )
                        if not weights_payload or not isinstance (weights_payload ,dict ):
                            if needs_global :
                                print (f"[Client {args .client_id }] 無法獲取全局權重，跳過本輪訓練（避免不同初始值問題）")
                                await asyncio .sleep (5 )
                                continue 
                            else :
                                print (f"[Client {args .client_id }] 無法獲取 server_weights，GKD 本輪跳過")
                                model ._has_server_weights =False 
                        else :
                            try :
                                raw_global_weights =weights_payload .get ("global_weights",{})if needs_global else {}
                                raw_server_weights =weights_payload .get ("server_weights",{})if (needs_server or check_server_update )else {}
                                server_model_version =weights_payload .get ("server_model_version")
                                teacher_model =None 
                                if needs_server or check_server_update :
                                    if check_server_update and server_model_version is not None :
                                        if server_model_version <=current_server_version :
                                            print (f"[Client {args .client_id }] server_weights 版本未更新（v={server_model_version } <= 當前 v={current_server_version }），跳過")
                                        elif not raw_server_weights :
                                            print (f"[Client {args .client_id }] server_weights 版本已更新（v={server_model_version } > 當前 v={current_server_version }），但權重為空")
                                        else :
                                            print (f"[Client {args .client_id }] server_weights 版本更新（v={current_server_version } → v={server_model_version }），重新載入...")
                                            teacher_model =_build_teacher_model_from_weights (model ,raw_server_weights ,device )
                                            if teacher_model is not None :
                                                model ._teacher_model =teacher_model 
                                                model ._has_server_weights =True 
                                                model ._server_weights_version =server_model_version 
                                                print (f"[Client {args .client_id }] 已更新 teacher 模型權重（GKD，v={server_model_version }）")
                                            else :
                                                print (f"[Client {args .client_id }] 無法建立 teacher，保持舊版本")
                                    elif needs_server :
                                        if raw_server_weights :
                                            print (f"[Client {args .client_id }] 收到 server_weights（keys={len (raw_server_weights )}，v={server_model_version })")
                                            teacher_model =_build_teacher_model_from_weights (model ,raw_server_weights ,device )
                                            if teacher_model is not None :
                                                model ._teacher_model =teacher_model 
                                                model ._has_server_weights =True 
                                                model ._server_weights_version =server_model_version 
                                                print (f"[Client {args .client_id }] 已載入 teacher 模型權重（GKD）")
                                            else :
                                                print (f"[Client {args .client_id }] 無法建立 teacher，GKD 將跳過")
                                                model ._has_server_weights =False 
                                        else :
                                            print (f"[Client {args .client_id }] 未收到 server_weights（GKD 將無法啟動）")
                                            model ._has_server_weights =False 
                                if needs_global :
                                    processed_weights ={}
                                    for key ,value in raw_global_weights .items ():
                                        if key .endswith ('num_batches_tracked'):
                                            if isinstance (value ,(int ,float ,np .integer ,np .floating )):
                                                processed_weights [key ]=torch .tensor (int (value ),dtype =torch .long )
                                            elif isinstance (value ,torch .Tensor ):
                                                processed_weights [key ]=value .long ()if value .dtype !=torch .long else value 
                                            else :
                                                try :
                                                    processed_weights [key ]=torch .tensor (int (value ),dtype =torch .long )
                                                except (ValueError ,TypeError ):
                                                    processed_weights [key ]=value 
                                        else :
                                            processed_weights [key ]=value 
                                    try :
                                        _validate_architecture_compatibility (model ,processed_weights ,args .client_id ,raise_on_mismatch =True )
                                        print (f"[Client {args .client_id }] 架構驗證通過：模型架構與全局權重架構匹配")
                                    except ValueError as arch_error :
                                        print (f"[Client {args .client_id }] {arch_error }")
                                        weight_arch =_detect_architecture_from_weights (processed_weights )
                                        model_arch =_detect_model_architecture (model )
                                        print (f"[Client {args .client_id }] 檢測到架構不匹配，嘗試重新構建模型...")
                                        print (f"  - 當前模型架構: {model_arch }")
                                        print (f"  - 全局權重架構: {weight_arch }")
                                        if weight_arch !='unknown':
                                            try :
                                                print (f"[Client {args .client_id }]檢測到全局權重是 {weight_arch } 架構，但客戶端模型是 {model_arch } 架構")
                                                print (f"[Client {args .client_id }] 無法自動切換架構，請確保客戶端配置與全局權重架構一致")
                                                print (f"[Client {args .client_id }] 建議：重新啟動客戶端，確保 MODEL_CONFIG['type'] = '{weight_arch }'")
                                                raise ValueError (f"架構不匹配：客戶端模型是 {model_arch }，但全局權重是 {weight_arch }。請重新啟動客戶端以使用正確的架構。")
                                            except ValueError :
                                                raise 
                                            except Exception as rebuild_error :
                                                print (f"[Client {args .client_id }] 處理架構不匹配時發生錯誤: {rebuild_error }")
                                                raise ValueError (f"無法處理架構不匹配問題: {rebuild_error }")
                                        else :
                                            raise ValueError (f"無法檢測全局權重架構，無法自動修復。請檢查全局權重是否有效。")
                                    base_weights ,head_weights =_split_state_dict (processed_weights )
                                    _apply_state_subset (model ,base_weights )
                                    model ._global_head_state ={name :tensor .detach ().clone ().cpu ()for name ,tensor in head_weights .items ()}
                                    if not getattr (model ,"_head_initialized",False ):
                                        _apply_state_subset (model ,head_weights )
                                        _save_local_head_state (args ,head_weights )
                                        model ._head_initialized =True 
                                    else :
                                        local_head_state =_load_local_head_state (args )
                                        if local_head_state :
                                            _apply_state_subset (model ,local_head_state )
                                    apply_server_weights =True 
                                    if client_kd_enabled and teacher_model is not None :
                                        apply_server_weights =False 
                                        print (f"[Client {args .client_id }] 已使用 server_weights 作為 teacher，避免覆蓋 student 權重")
                                    if raw_server_weights and apply_server_weights :
                                        _apply_state_subset (model ,raw_server_weights )
                                    model ._fedprox_global_weights =processed_weights 
                                    output_layer_fixed =False 
                                    for name ,param in model .named_parameters ():
                                        if 'output_layer'in name and 'bias'in name :
                                            bias_norm =param .norm ().item ()
                                            if bias_norm <0.15 :
                                                print (f"[Client {args .client_id }] 輸出層 bias 範數過小 ({bias_norm :.6f})，重新初始化")
                                                fan_in =param .size (0 )if len (param .shape )>0 else 1 
                                                bound =1.0 /math .sqrt (fan_in )if fan_in >0 else 0.1 
                                                bound =max (bound ,0.2 )
                                                nn .init .uniform_ (param ,-bound ,bound )
                                                new_norm =param .norm ().item ()
                                                print (f"[Client {args .client_id }] 輸出層 bias 已重新初始化: 新範數={new_norm :.6f} (bound={bound :.3f})")
                                                output_layer_fixed =True 
                                            else :
                                                print (f"[Client {args .client_id }] 輸出層 bias 範數: {bias_norm :.6f} (正常)")
                                    if output_layer_fixed :
                                        print (f"[Client {args .client_id }]已修復輸出層 bias，確保模型可以正常學習")
                                    model ._has_global_weights =True 
                                    print (f"[Client {args .client_id }] 已載入全局權重，可以開始訓練")
                            except Exception as e :
                                print (f"[Client {args .client_id }] 載入全局權重失敗: {e }，跳過本輪訓練")
                                await asyncio .sleep (5 )
                                continue 
                    print (f"[Client {args .client_id }] 開始訓練輪次 {training_round }（已確保使用相同初始值）")
                    import sys 
                    sys .stdout .flush ()
                    print (f" 檢查數據加載器:")
                    print (f"  - 訓練數據加載器大小: {len (train_loader )}")
                    print (f"  - 驗證數據加載器大小: {len (val_loader )}")
                    print (f"  - 批次大小: {train_loader .batch_size }")
                    print (f"  - 設備: {device }")
                    import sys 
                    sys .stdout .flush ()
                    print (f" 測試數據加載器...")
                    test_batch =next (iter (train_loader ))
                    print (f"  - 測試批次數據形狀: {test_batch [0 ].shape }")
                    print (f"  - 測試批次目標形狀: {test_batch [1 ].shape }")
                    print (f"  - 數據類型: {test_batch [0 ].dtype }")
                    print (f"  - 目標類型: {test_batch [1 ].dtype }")
                    sys .stdout .flush ()
                    print (f" 檢查模型架構:")
                    print (f"  - 模型參數數量: {sum (p .numel ()for p in model .parameters ())}")
                    print (f"  - 模型設備: {next (model .parameters ()).device }")
                    print (f"  - 模型訓練模式: {model .training }")
                    sys .stdout .flush ()
                    print (f" 測試模型前向傳播...")
                    try :
                        from utils .cuda_memory_manager import safe_cuda_operation 
                        def test_forward ():
                            test_input =test_batch [0 ][:2 ].to (device )
                            with torch .no_grad ():
                                output =model (test_input )
                                return output 
                        test_output =safe_cuda_operation (test_forward )
                        print (f"  - 輸入形狀: {test_batch [0 ][:2 ].shape }")
                        print (f"  - 輸出形狀: {test_output .shape }")
                        print (f"  - 輸出範圍: [{test_output .min ().item ():.4f}, {test_output .max ().item ():.4f}]")
                    except ImportError :
                        try :
                            test_input =test_batch [0 ][:2 ].to (device )
                            with torch .no_grad ():
                                output =model (test_input )
                                test_output =output 
                            print (f"  - 輸入形狀: {test_input .shape }")
                            print (f"  - 輸出形狀: {test_output .shape }")
                            print (f"  - 輸出範圍: [{test_output .min ().item ():.4f}, {test_output .max ().item ():.4f}]")
                        except RuntimeError as e :
                            if "CUBLAS_STATUS_ALLOC_FAILED"in str (e )or "out of memory"in str (e ).lower ():
                                print (f" CUDA記憶體不足，切換到CPU: {e }")
                                device =torch .device ("cpu")
                                model =model .to (device )
                                test_input =test_batch [0 ][:1 ].to (device )
                                with torch .no_grad ():
                                    output =model (test_input )
                                    test_output =output 
                                print (f"  - 已切換到CPU設備")
                                print (f"  - 輸入形狀: {test_input .shape }")
                                print (f"  - 輸出形狀: {test_output .shape }")
                            else :
                                raise e 
                    sys .stdout .flush ()
                    model .train ()
                    sys .stdout .flush ()
                    for epoch in range (config .LOCAL_EPOCHS ):
                        print (f"[Client {args .client_id }] 開始第 {epoch +1 }/{config .LOCAL_EPOCHS } 輪本地訓練 (聯邦輪次: {training_round })")
                        import sys 
                        sys .stdout .flush ()
                        if epoch ==0 :
                            model ._weights_before_training ={name :param .data .clone ()for name ,param in model .named_parameters ()}
                        if scheduler is not None :
                            current_lr =optimizer .param_groups [0 ]['lr']
                            print (f"[Client {args .client_id }] 當前學習率: {current_lr :.6f}")
                        try :
                            from utils .cuda_memory_manager import safe_cuda_operation ,get_memory_manager 
                            memory_manager =get_memory_manager ()
                            def train_epoch ():
                                global_weights_for_fedprox =getattr (model ,'_fedprox_global_weights',None )
                                if global_weights_for_fedprox is None and hasattr (model ,'_last_global_weights'):
                                    global_weights_for_fedprox =model ._last_global_weights 
                                teacher_model =getattr (model ,"_teacher_model",None )
                                client_kd_cfg =getattr (config ,"CLIENT_KD_CONFIG",{})or {}
                                return train_one_round (model ,train_loader ,optimizer ,criterion ,device ,global_weights =global_weights_for_fedprox ,current_round =training_round ,epoch_num =epoch ,total_epochs =config .LOCAL_EPOCHS ,teacher_model =teacher_model ,kd_cfg =client_kd_cfg )
                            train_stats =safe_cuda_operation (train_epoch )
                            print (f"[Client {args .client_id }] Epoch {epoch +1 }/{config .LOCAL_EPOCHS } 完成: Loss={train_stats ['loss']:.4f}, Acc={train_stats ['acc']:.4f}")
                            if scheduler is not None :
                                scheduler .step ()
                                new_lr =optimizer .param_groups [0 ]['lr']
                                if new_lr !=current_lr :
                                    print (f"[Client {args .client_id }] 學習率已更新: {current_lr :.6f} → {new_lr :.6f}")
                            memory_manager .monitor_memory_usage (f"Epoch {epoch +1 }")
                        except ImportError :
                            try :
                                global_weights_for_fedprox =getattr (model ,'_fedprox_global_weights',None )
                                if global_weights_for_fedprox is None and hasattr (model ,'_last_global_weights'):
                                    global_weights_for_fedprox =model ._last_global_weights 
                                teacher_model =getattr (model ,"_teacher_model",None )
                                client_kd_cfg =getattr (config ,"CLIENT_KD_CONFIG",{})or {}
                                train_stats =train_one_round (model ,train_loader ,optimizer ,criterion ,device ,global_weights =global_weights_for_fedprox ,current_round =training_round ,epoch_num =epoch ,total_epochs =config .LOCAL_EPOCHS ,teacher_model =teacher_model ,kd_cfg =client_kd_cfg )
                                print (f"[Client {args .client_id }] Epoch {epoch +1 }/{config .LOCAL_EPOCHS } 完成: Loss={train_stats ['loss']:.4f}, Acc={train_stats ['acc']:.4f}")
                            except Exception as e :
                                print (f"[Client {args .client_id }] Epoch {epoch +1 } 訓練失敗: {e }")
                                import traceback 
                                traceback .print_exc ()
                                train_stats =train_one_round (model ,train_loader ,optimizer ,criterion ,device ,global_weights =global_weights_for_fedprox ,current_round =training_round ,epoch_num =epoch ,total_epochs =config .LOCAL_EPOCHS ,teacher_model =teacher_model ,kd_cfg =client_kd_cfg )
                                print (f"[Client {args .client_id }] Epoch {epoch +1 }/{config .LOCAL_EPOCHS } 完成 (回退模式): Loss={train_stats ['loss']:.4f}, Acc={train_stats ['acc']:.4f}")
                                if scheduler is not None :
                                    scheduler .step ()
                            except RuntimeError as e :
                                if "CUBLAS_STATUS_ALLOC_FAILED"in str (e )or "out of memory"in str (e ).lower ():
                                    print (f" 訓練時CUDA記憶體不足，切換到CPU: {e }")
                                    device =torch .device ("cpu")
                                    model =model .to (device )
                                    if class_weights_tensor is not None :
                                        class_weights_tensor =class_weights_tensor .to (device )
                                        criterion =build_loss (class_weights =class_weights_tensor ,label_smoothing =label_smoothing )
                                    weight_decay =float (os .environ .get ("WEIGHT_DECAY",getattr (config ,"WEIGHT_DECAY",1e-4 )))
                                    optimizer =torch .optim .Adam (model .parameters (),lr =config .LEARNING_RATE ,weight_decay =weight_decay )
                                    print (f"[Client {args .client_id }]優化器配置: lr={config .LEARNING_RATE }, weight_decay={weight_decay }")
                                    global_weights_for_fedprox =getattr (model ,'_fedprox_global_weights',None )
                                    if global_weights_for_fedprox is None and hasattr (model ,'_last_global_weights'):
                                        global_weights_for_fedprox =model ._last_global_weights 
                                    teacher_model =getattr (model ,"_teacher_model",None )
                                    client_kd_cfg =getattr (config ,"CLIENT_KD_CONFIG",{})or {}
                                    train_stats =train_one_round (model ,train_loader ,optimizer ,criterion ,device ,global_weights =global_weights_for_fedprox ,current_round =training_round ,epoch_num =epoch ,total_epochs =config .LOCAL_EPOCHS ,teacher_model =teacher_model ,kd_cfg =client_kd_cfg )
                                    print (f"[Client {args .client_id }] Epoch {epoch +1 }/{config .LOCAL_EPOCHS } 完成 (CPU模式): Loss={train_stats ['loss']:.4f}, Acc={train_stats ['acc']:.4f}")
                                else :
                                    raise e 
                        import sys 
                        sys .stdout .flush ()
                    print (f"[Client {args .client_id }] 訓練完成，準備保存 encoder / 評估 / 上傳")
                    sys .stdout .flush ()
                    try :
                        from utils .cuda_memory_manager import safe_cuda_operation 
                        def eval_model ():
                            return evaluate (model ,val_loader ,criterion ,device )
                        print (f"[Client {args .client_id }] 開始評估 (val_loader={len (val_loader )})")
                        val_stats =safe_cuda_operation (eval_model )
                        print (f"[Client {args .client_id }] 評估完成")
                        print (f"Validation: Loss={val_stats ['val_loss']:.4f}, Acc={val_stats ['val_acc']:.4f}, F1={val_stats ['f1_score']:.4f}")
                    except ImportError :
                        try :
                            val_stats =evaluate (model ,val_loader ,criterion ,device )
                            print (f"Validation: Loss={val_stats ['val_loss']:.4f}, Acc={val_stats ['val_acc']:.4f}, F1={val_stats ['f1_score']:.4f}")
                        except RuntimeError as e :
                            if "CUBLAS_STATUS_ALLOC_FAILED"in str (e )or "out of memory"in str (e ).lower ():
                                print (f" 評估時CUDA記憶體不足，切換到CPU: {e }")
                                device =torch .device ("cpu")
                                model =model .to (device )
                                val_stats =evaluate (model ,val_loader ,criterion ,device )
                                print (f"Validation (CPU): Loss={val_stats ['val_loss']:.4f}, Acc={val_stats ['val_acc']:.4f}, F1={val_stats ['f1_score']:.4f}")
                            else :
                                raise e 
                    import sys 
                    sys .stdout .flush ()
                    local_head_snapshot =_extract_head_state (model )
                    _save_local_head_state (args ,local_head_snapshot )
                    weights ={}
                    global_head_state =getattr (model ,"_global_head_state",{})
                    for name ,param in model .state_dict ().items ():
                        if name .startswith ('_teacher_model'):
                            continue 
                        tensor =param .detach ().cpu ()
                        if _is_head_layer (name ):
                            if name in global_head_state :
                                tensor =global_head_state [name ].detach ().clone ()
                        weights [name ]=tensor 
                    if hasattr (model ,'_last_global_weights'):
                        weight_changes ={}
                        for name ,current_weight in weights .items ():
                            if name in model ._last_global_weights :
                                old_weight =model ._last_global_weights [name ]
                                if isinstance (current_weight ,torch .Tensor )and isinstance (old_weight ,torch .Tensor ):
                                    if current_weight .shape ==old_weight .shape :
                                        if current_weight .dtype not in (torch .float32 ,torch .float64 ,torch .float16 ):
                                            continue 
                                        if old_weight .dtype not in (torch .float32 ,torch .float64 ,torch .float16 ):
                                            continue 
                                        weight_diff =(current_weight -old_weight ).abs ().mean ().item ()
                                        weight_norm =current_weight .norm ().item ()
                                        weight_changes [name ]={'mean_diff':weight_diff ,'norm':weight_norm ,'relative_change':weight_diff /(weight_norm +1e-8 )}
                        if weight_changes :
                            avg_change =sum (w ['mean_diff']for w in weight_changes .values ())/len (weight_changes )
                            avg_relative =sum (w ['relative_change']for w in weight_changes .values ())/len (weight_changes )
                            print (f"[Client {args .client_id }] 權重更新統計: 平均絕對變化={avg_change :.6f}, 平均相對變化={avg_relative :.6f}")
                            if avg_change <1e-6 :
                                print (f"[Client {args .client_id }] 警告：權重更新極小，可能未有效訓練")
                            sorted_changes =sorted (weight_changes .items (),key =lambda x :x [1 ]['mean_diff'],reverse =True )
                            print (f"[Client {args .client_id }] 前5個變化最大的層:")
                            for name ,stats in sorted_changes [:5 ]:
                                print (f"  - {name }: 絕對變化={stats ['mean_diff']:.6f}, 相對變化={stats ['relative_change']:.6f}")
                    model ._last_global_weights ={name :param .clone ().cpu ()for name ,param in model .state_dict ().items ()}
                    try :
                        from utils .cuda_memory_manager import get_memory_manager 
                        memory_manager =get_memory_manager ()
                        memory_manager .clear_memory (aggressive =True )
                    except ImportError :
                        if device .type =='cuda':
                            torch .cuda .empty_cache ()
                    metrics_payload ={"train_loss":float (train_stats .get ('loss',0.0 )),"train_acc":float (train_stats .get ('acc',0.0 )),"val_loss":float (val_stats .get ('val_loss',0.0 )),"val_acc":float (val_stats .get ('val_acc',0.0 )),"val_f1":float (val_stats .get ('f1_score',0.0 )),"train_samples":len (X_train ),"val_samples":len (X_val )}
                    success ,status_code ,msg =await upload_weights (session ,args .aggregator_url ,args .client_id ,weights ,len (X_train ),val_stats ,training_round ,metrics_payload =metrics_payload )
                    try :
                        train_time_ms =int (getattr (model ,'_last_train_ms',0 ))
                        joint_f1 =float (val_stats .get ('f1_score',0.0 ))
                        alpha_str =""
                        if hasattr (model ,"get_alpha"):
                            try :
                                alpha =model .get_alpha ().detach ().cpu ().numpy ().reshape (-1 ).tolist ()
                                alpha_str ="|".join ([f"{v :.4f}"for v in alpha ])
                            except Exception :
                                alpha_str =""
                        if curve_csv :
                            with open (curve_csv ,"a",encoding ="utf-8")as f :
                                f .write (f"{training_round },{train_stats .get ('loss')},{train_stats .get ('acc')},"f"{val_stats .get ('val_loss')},{val_stats .get ('val_acc')},{joint_f1 },{alpha_str },{label_flip_count },{len (X_train )},"f"{int (success )},{status_code },{'accepted'if success else 'rejected'},"f"{msg .replace (',',' ')[:200 ]},{train_time_ms }\n")
                        if 'confusion_matrix'in val_stats and val_stats ['confusion_matrix']:
                            try :
                                cm_dir =os .path .join (args .result_dir ,f"uav{args .client_id }")
                                os .makedirs (cm_dir ,exist_ok =True )
                                cm_file =os .path .join (cm_dir ,f"confusion_matrix_round_{training_round }.csv")
                                cm =np .array (val_stats ['confusion_matrix'])
                                num_classes =config .MODEL_CONFIG .get ('num_classes',5 )
                                if cm .shape !=(num_classes ,num_classes ):
                                    print (f"[Client {args .client_id }] 混淆矩陣形狀不正確: {cm .shape }, 預期: ({num_classes }, {num_classes })")
                                    full_cm =np .zeros ((num_classes ,num_classes ),dtype =int )
                                    if cm .size >0 :
                                        min_dim =min (cm .shape [0 ],num_classes )
                                        full_cm [:min_dim ,:min_dim ]=cm [:min_dim ,:min_dim ]
                                    cm =full_cm 
                                cm_df =pd .DataFrame (cm ,columns =[f'Pred_{i }'for i in range (num_classes )])
                                cm_df .index =[f'True_{i }'for i in range (num_classes )]
                                cm_df .to_csv (cm_file ,index =True )
                                print (f"[Client {args .client_id }] 混淆矩陣已保存: {cm_file } (形狀: {cm .shape })")
                            except Exception as e :
                                print (f"[Client {args .client_id }] 保存混淆矩陣失敗: {e }")
                                import traceback 
                                traceback .print_exc ()
                    except Exception as e :
                        print (f"[Client {args .client_id }] 記錄curve失敗: {e }")
                    last_attempted_round =training_round 
                    print (f" 嘗試訓練輪次 {training_round }，更新 last_attempted_round")
                    if success :
                        last_confirmed_round =training_round 
                        print (f" 服務端確認上傳成功，last_confirmed_round 推進到 {last_confirmed_round }")
                    else :
                        print (f"上傳失敗，last_confirmed_round 保持 {last_confirmed_round }，將重試")
                else :
                    if current_round >last_confirmed_round :
                        last_confirmed_round =current_round 
                        print (f" 客戶端未訓練，同步已確認輪次到 {last_confirmed_round }")
                    else :
                        print (f" 客戶端未訓練，已確認輪次保持 {last_confirmed_round }")
                    import sys 
                    sys .stdout .flush ()
                idle_ticks =0 
            except Exception as e :
                print (f"Error in client loop: {e }")
                import traceback 
                print (f"Traceback: {traceback .format_exc ()}")
                import sys 
                sys .stdout .flush ()
                idle_ticks +=1 
                if idle_ticks >10 :
                    print ("Too many errors, exiting...")
                    break 
            await asyncio .sleep (5 )
if __name__ =="__main__":
    asyncio .run (main ())
