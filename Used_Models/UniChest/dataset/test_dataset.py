import csv
import json
import logging
import os
import random
import re
import sys
from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import cv2
import pydicom
from skimage import exposure


class Openi_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,3])
        self.class_list = np.asarray(data_info.iloc[:,4:])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize(image_res, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        # img_path = self.img_path_list[index].replace('/mnt/cfs/xmzhang/DATA/ChestXray8/','/remote-home/share/medical/public/ChestXray8/')
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
        
        
        
class Chestxray14_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize(image_res, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index].replace('/mnt/petrelfs/zhangxiaoman/DATA/Chestxray/ChestXray8/','/remote-home/share/medical/public/ChestXray8/')
        
        # img_path = self.img_path_list[index].replace('/mnt/cfs/xmzhang/DATA/ChestXray8/','/remote-home/share/medical/public/ChestXray8/')
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)



class CheXpert_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        # self.class_list = np.asarray(data_info.iloc[:,[9,3,7,6,11]])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = os.path.join('/remote-home/share/xmzhang/CheXpert/',self.img_path_list[index])
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

class Padchest_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        # try:
            img_path = self.img_path_list[index].replace('/mnt/petrelfs/zhangxiaoman/DATA/Chestxray/PadChest/images/', '/remote-home/share/xmzhang/PadChest/LR_images/')
            class_label = self.class_list[index] 
            img_array = np.array(Image.open(img_path))
            img_array = (img_array/img_array.max())*255
            img = Image.fromarray(img_array.astype('uint8')).convert('RGB')   
            image = self.transform(img)
            return {
                "img_path": img_path,
                "image": image,
                "label": class_label
                }
        # except:
        #     select_index = random.randint(10000)
        #     img_path = self.img_path_list[select_index]
        #     class_label = self.class_list[select_index] 
        #     img_array = np.array(Image.open(img_path))
        #     img_array = (img_array/img_array.max())*255
        #     img = Image.fromarray(img_array.astype('uint8')).convert('RGB')   
        #     image = self.transform(img)
        #     return {
        #         "img_path": img_path,
        #         "image": image,
        #         "label": class_label
        #         }
    
    def __len__(self):
        return len(self.img_path_list)

class Vindr_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
        

class SIIMACR_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

    
class Shenzhen_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)


    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,5])
        self.class_list = np.asarray(data_info.iloc[:,6:])
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = '/remote-home/share/medical/public/MIMIC-CXR-JPG/MIMIC-CXR/small/' + self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
    
class MIMIC_Test_Dataset(Dataset):
    def __init__(self, csv_path, cls_path,image_res):
        # self.json_info = json.load(open(json_path,'r'))
        # data_info = pd.read_csv(csv_path)
        # self.img_path_list = np.asarray(data_info.iloc[:,0])
        # self.class_list = np.asarray(data_info.iloc[:,1:])#40 class for fine-grained query list
        # cls_info = pd.read_csv(cls_path)
        # self.sty_dict_info = self.csv_to_dict(sty_info)


        self.cxr_labels = ['Atelectasis','Cardiomegaly', 
                            'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                            'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                            'Pneumothorax', 'Support Devices']

        tmp = []
        tmp_img_path = []
        for idx, path in enumerate(tqdm(pd.read_csv(csv_path)["img_path"][:2000])):
            for path_ in str(path).split(";"):
                tmp_img_path.append(path_)
                tmp.append(path_.split("/")[2][1:])
        test_study_id = tmp
        self.img_path_list = tmp_img_path

        full_labels_ori = pd.read_csv(cls_path)
        new_rows = []
        for idx, study_id in tqdm(enumerate(test_study_id)):    #len(test_study_id)
            new_row = full_labels_ori[full_labels_ori["study_id"]==int(test_study_id[idx])]
            new_rows.append(new_row)
        full_labels = pd.concat(new_rows, axis=0)
        full_labels.replace(-1, 0, inplace=True)
        full_labels.fillna(0, inplace=True)
        full_labels_array = full_labels.loc[:, self.cxr_labels]
        self.subject_id_list = full_labels.loc[:, "subject_id"].tolist()
        self.study_id_list = full_labels.loc[:, "study_id"].tolist()
        self.class_list = full_labels_array.to_numpy()

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # if args.colourjitter:
        #     self.transform = transforms.Compose([                        
        #         transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        #         transforms.RandomHorizontalFlip(),

        #         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        #         transforms.RandomGrayscale(),

        #         RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
        #                                         'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
        #         transforms.ToTensor(),
        #         normalize,
        #     ])

        # else:
        self.transform = transforms.Compose([                        
            # transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            # RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
            #                                 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    

    
    # def csv_to_dict(self,sty_info):
    #     tui_list = sty_info.iloc[:,0]
    #     sty_list = sty_info.iloc[:,1]
    #     sty_dict = defaultdict(list)
    #     for idx in tqdm(range(len(tui_list))):
    #         tui_idx = tui_list[idx]
    #         sty_idx = sty_list[idx]
    #         sty_dict[tui_idx] = sty_idx
    #     return sty_dict
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        subject_id = str(self.subject_id_list[index])
        study_id = str(self.study_id_list[index])

        # index_transit = np.load("/remote-home/tianjiedai/KAD/R1_CLIP_LR/A1_DATA/small/index0626.npy")
        # new_index_json = index_transit[index]
        # entities = self.json_info[new_index_json]['entities']
        # captions = self.json_info[new_index_json]['caption']
        
        # entities = self.json_info[index]['entities']
        # captions = self.json_info[index]['caption']


        # if len(entities) != 0:
        #     caption_list = ''
        #     entity_details = ''
        #     for entity in entities:
        #         sub_caption = entity['caption']
        #         sub_entities = entity['entity']#搞错了 还不是list
        #         sub_entity_details = ''
        #         for sub_entity in sub_entities:
        #             try:
        #                 sub_entity_details += ' [ENT] ' + sub_entity['Entity'] 
        #             except:
        #                 sub_entity_details += ' [ENT] ' + sub_entity['Entity']  
        #         entity_details = entity_details + sub_entity_details + ' [SEP] '
        #         caption_list = caption_list + sub_caption + ' [SEP] '
        # else:
        #     caption_list = ''
        #     entity_details = ''
        #     for sub_caption in captions:
        #         caption_list = caption_list + sub_caption + ' [SEP] '
        #     entity_details = caption_list
        
        # img = open_jpg(img_path).convert('RGB')  
        img = Image.open(os.path.join("../CXR-MultiAgentSystem/CheXzero/data/MIMIC/mimic_cxr/images", img_path)).convert('RGB') 
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label,
            "subject_id": subject_id,
            "study_id": study_id,
            # "caption": caption_list,
            # "entity": entity_details
            }