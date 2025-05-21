from torch.utils.data import Dataset, DataLoader
from clip import tokenize,senceGraph_tokenize
from PIL import Image
import json
from transformers import BertTokenizer
from torchvision import transforms
import torch
from tqdm import tqdm
import os
from easydict import EasyDict as edict
from pycocotools import mask as mask_utils
import numpy as np
import torchvision.transforms.functional as TF
class VG_Relation(Dataset):
    def __init__(self, transform = None):
        self.transform = transform
        self.length = 5
        self.padding_num = 6
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.root_dir = "/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/lvxiaotian/Structure-CLIP/data/images_relation/"
        
        with open("/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/lvxiaotian/Structure-CLIP/data/visual_genome_relation_aug_final.json", "r") as f:
            self.dataset = json.load(f)
        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(self.root_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):    
        rel = self.dataset[index]
        # step1 
        image = Image.open(rel["image_path"]).convert('RGB')
        transform = transforms.ToTensor()
        tensor_image = transform(image)    
        """# Get the bounding box that contains the relation
        image = image.crop((rel["bbox_x"], rel["bbox_y"], rel["bbox_x"]+rel["bbox_w"], rel["bbox_y"]+rel["bbox_h"]))
        
        if self.transform is not None:
            image = self.transform(image)"""
        
        # step2 
        #caption = rel["true_caption"]
        #reversed_caption = rel["false_caption"]  

        # step3 
        # 3.1 
        triples = rel['true_triples']
        caption = ""
        for triple in triples:
            caption += triple[0] + " " + triple[1] + " " + triple[2] + ". "
        caption = caption[:-1]
        head_inputs, relation_inputs, tail_inputs, attention_mask = self.get_triple_token_infor(triples)
        #print(caption)
        # 3.2 
        reversed_triples = rel['false_triples']
        reversed_caption = ""
        for reversed_triple in reversed_triples:
            reversed_caption += reversed_triple[0] + " " + reversed_triple[1] + " " + reversed_triple[2] + ". "
        reversed_caption = reversed_caption[:-1]
        reversed_head_inputs, reversed_relation_inputs, reversed_tail_inputs, reversed_attention_mask = self.get_triple_token_infor(reversed_triples)

        """feature_segs = []
        for seg in rel["feature_seg"]:
            #print(seg)
            c = mask_utils.decode(seg)       
            feature_segs.append(c)
        if len(feature_segs) < 20:            
            for i in range(20-len(feature_segs)):
                feature_segs.append(np.zeros((14, 14), dtype=np.uint8))
        segs_array = np.stack(feature_segs, axis=0)
        segs = torch.from_numpy(segs_array)"""

        item = edict({"image_options": [tensor_image], 
                      "caption_options": [reversed_caption, caption], 
                      "relation": rel["relation_name"],
                      "head_inputs": head_inputs,
                      "relation_inputs": relation_inputs,
                      "tail_inputs": tail_inputs,
                      "attention_mask": attention_mask,
                      #"feature_segs":segs,
                      "reversed_head_inputs": reversed_head_inputs,
                      "reversed_relation_inputs": reversed_relation_inputs,
                      "reversed_tail_inputs": reversed_tail_inputs,
                      "reversed_attention_mask": reversed_attention_mask})
        return item

    def get_triple_token_infor(self, triples):
        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()
        if len(triples) > 0:
            for triple in triples:
                if len(head_word_list) < self.padding_num:
                    head_word_list.append(triple[0])
                    relation_word_list.append(triple[1])
                    tail_word_list.append(triple[2])
                

        token_type_ids = torch.zeros([1,self.padding_num], dtype=int)

        if len(head_word_list) > 0:
            attention_mask = torch.cat((torch.ones([1, len(head_word_list)], dtype=int), torch.zeros([1, self.padding_num-len(head_word_list)], dtype=int)), dim=1)
        else:
            attention_mask = torch.zeros([1, self.padding_num], dtype=int)
        

        for i in range(self.padding_num - len(head_word_list)):
            head_word_list.append('')
            relation_word_list.append('')
            tail_word_list.append('')
        
        head_inputs = self.tokenizer.batch_encode_plus(
            head_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        relation_inputs = self.tokenizer.batch_encode_plus(
            relation_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        tail_inputs = self.tokenizer.batch_encode_plus(
            tail_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        return head_inputs, relation_inputs, tail_inputs, attention_mask


class VG_Attribution(Dataset):
    def __init__(self, data_path=None, transform=None):

        self.transform = transform
        self.length = 5
        self.padding_num = 6
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if 'adjchange' in data_path:
            self.root_dir = '/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/lvxiaotian/Structure-CLIP/data/'
        else:
            self.root_dir = "/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/lvxiaotian/Structure-CLIP/data/images_attribution/"
        self.data_path = data_path
        
        with open(self.data_path, "r") as f:
            self.dataset = json.load(f)
        for item in self.dataset:
            item["image_path"] = os.path.join(self.root_dir, item["image_path"])
        if 'adjchange' not in data_path:
            self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]

    def __len__(self):
        return len(self.dataset)
    
    def get_triple_token_infor(self, triples):

        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()

        if len(triples) > 0:
            for triple in triples:
                if len(head_word_list) < self.padding_num:
                    head_word_list.append(triple[0])
                    relation_word_list.append(triple[1])
                    tail_word_list.append(triple[2])
                
        token_type_ids = torch.zeros([1,self.padding_num], dtype=int)

        if len(head_word_list) > 0:
            attention_mask = torch.cat((torch.ones([1, len(head_word_list)], dtype=int), torch.zeros([1, self.padding_num-len(head_word_list)], dtype=int)), dim=1)
        else:
            attention_mask = torch.zeros([1, self.padding_num], dtype=int)
        
        for i in range(self.padding_num - len(head_word_list)):
            head_word_list.append('')
            relation_word_list.append('')
            tail_word_list.append('')
        
        head_inputs = self.tokenizer.batch_encode_plus(
            head_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        relation_inputs = self.tokenizer.batch_encode_plus(
            relation_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        tail_inputs = self.tokenizer.batch_encode_plus(
            tail_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        return head_inputs, relation_inputs, tail_inputs, attention_mask
    
    def __getitem__(self, index):
        scene = self.dataset[index]
        # step1 
        image = Image.open(scene["image_path"]).convert('RGB')
        transform = transforms.ToTensor()
        tensor_image = transform(image)
        # Get the bounding box that contains the relation
        """if self.root_dir != '/root/data/coco_data/':
            image = image.crop((scene["bbox_x"], scene["bbox_y"], scene["bbox_x"] + scene["bbox_w"], scene["bbox_y"] + scene["bbox_h"]))"""

        """if self.transform is not None:
            image = self.transform(image)"""

        # step2 
        """true_caption = scene["true_caption"]
        false_caption = scene["false_caption"]"""
        

        # step3 
        triples = scene['true_triples']
        head_inputs, relation_inputs, tail_inputs, attention_mask = self.get_triple_token_infor(triples)
        #print(true_caption)
        reversed_triples = scene['false_triples']

        reversed_head_inputs, reversed_relation_inputs, reversed_tail_inputs, reversed_attention_mask = self.get_triple_token_infor(reversed_triples)
        
        if "train" in self.data_path:
            feature_segs = []
            for seg in scene["feature_seg"]:
                #print(seg)
                c = mask_utils.decode(seg)       
                feature_segs.append(c)
            if len(feature_segs) ==0:
                print("triples:",scene['true_caption'])
            if len(feature_segs) < 20:            
                for i in range(20-len(feature_segs)):
                    feature_segs.append(np.zeros((14, 14), dtype=np.uint8))

            segs_array = np.stack(feature_segs, axis=0)
            segs = torch.from_numpy(segs_array).reshape(20,-1)
            triples = scene['true_triples']
            true_caption = []
            object_list = []
            for triple in triples:
                
                true_caption.append(triple[0])
                true_caption.append(triple[1])
                true_caption.append(triple[2])
                true_caption.append(".")

            false_caption = []
            for reversed_triple in reversed_triples:
                false_caption.append(reversed_triple[0])
                false_caption.append(reversed_triple[1])
                false_caption.append(reversed_triple[2])
                false_caption.append(".")
            true_caption_encode,true_text_encode_lenth,true_lenth,true_triples_encode = senceGraph_tokenize(true_caption,  truncate=True)
            false_caption_encode,false_text_encode_lenth,false_lenth,false_triples_encode = senceGraph_tokenize(false_caption,  truncate=True)
            if len(true_triples_encode) != len(triples):
                print("error")
            obj1_indices = []
            rel_indices = []
            obj2_indices = []
            index1=0
            index2=0
            index3=0
            index =0
            matrix1 = [[0 for _ in range(77)] for _ in range(10)]
            matrix2 = [[0 for _ in range(77)] for _ in range(10)]
            matrix3 = [[0 for _ in range(77)] for _ in range(10)]
            #print(len(matrix1[0]))
        
            
            if len(true_triples_encode) > 1:
                for i in range(len(true_triples_encode)):
                    if true_triples_encode[i][0] not in obj1_indices:
                        obj1_indices.append(true_triples_encode[i][0])
                        
                        
                    index111 = obj1_indices.index(true_triples_encode[i][0])
                    start_index1 = true_text_encode_lenth[i*4 + 0][0]
                    end_index1 = true_text_encode_lenth[i*4 + 0][1]
                    while(start_index1 <= end_index1):
                        matrix1[index111][start_index1] = 1
                        start_index1 += 1
                    
                    if true_triples_encode[i][1] not in rel_indices:
                        rel_indices.append(true_triples_encode[i][1])
                    index222 = rel_indices.index(true_triples_encode[i][1])
                    start_index2 = true_text_encode_lenth[i*4 + 1][0]
                    end_index2 = true_text_encode_lenth[i*4 + 1][1]
                    while(start_index2 <= end_index2):
                        matrix2[index222][start_index2] = 1
                        start_index2 += 1
                    
                    if true_triples_encode[i][2] not in obj2_indices:
                        obj2_indices.append(true_triples_encode[i][2])
                    index333 = obj2_indices.index(true_triples_encode[i][2])
                    start_index3 = true_text_encode_lenth[i*4 + 2][0]
                    end_index3 = true_text_encode_lenth[i*4 + 2][1]
                    while(start_index3 <= end_index3):
                        matrix3[index333][start_index3] = 1
                        start_index3 += 1
                    
                    if true_triples_encode[i][2] in obj1_indices:
                        obj1_indices.append(true_triples_encode[i][2])
                        #print("++++++++++++++++++")
                        index444 = obj1_indices.index(true_triples_encode[i][2])
                        start_index4 = true_text_encode_lenth[i*4 + 2][0]
                        end_index4 = true_text_encode_lenth[i*4 + 2][1]
                        while(start_index4 <= end_index4):
                            matrix1[index444][start_index4] = 1
                            start_index4 += 1
                    
            elif len(true_triples_encode) == 1:
                if true_triples_encode[0][0] not in obj1_indices:
                    obj1_indices.append(true_triples_encode[0][0])
                    
                index111 = obj1_indices.index(true_triples_encode[0][0])
                start_index1 = true_text_encode_lenth[0*4 + 0][0]
                end_index1 = true_text_encode_lenth[0*4 + 0][1]
                while(start_index1 <= end_index1):
                    matrix1[index111][start_index1] = 1
                    start_index1 += 1
                
                if true_triples_encode[0][1] not in rel_indices:
                    rel_indices.append(true_triples_encode[0][1])
                index222 = rel_indices.index(true_triples_encode[0][1])
                start_index2 = true_text_encode_lenth[0*4 + 1][0]
                end_index2 = true_text_encode_lenth[0*4 + 1][1]
                while(start_index2 <= end_index2):
                    matrix2[index222][start_index2] = 1
                    start_index2 += 1
                
                if true_triples_encode[0][2] not in obj2_indices:
                    obj2_indices.append(true_triples_encode[0][2])
                index333 = obj2_indices.index(true_triples_encode[0][2])
                start_index3 = true_text_encode_lenth[0*4 + 2][0]
                end_index3 = true_text_encode_lenth[0*4 + 2][1]
                while(start_index3 <= end_index3):
                    matrix3[index333][start_index3] = 1
                    start_index3 += 1
                
                if true_triples_encode[0][2] not in obj1_indices:
                    obj1_indices.append(true_triples_encode[0][2])
                    #print("zzzzzzzzzzzzzzzzzzzzzzzzzz")
                index444 = obj1_indices.index(true_triples_encode[0][2])
                start_index4 = true_text_encode_lenth[0*4 + 2][0]
                end_index4 = true_text_encode_lenth[0*4 + 2][1]
                while(start_index4 <= end_index4):
                    matrix1[index444][start_index4] = 1
                    start_index4 += 1
            for i in range(10-len(obj1_indices)):
                obj1_indices.append([0,0,0,0,0,0,0])
            for i in range(10-len(obj2_indices)):
                obj2_indices.append([0,0,0,0,0,0,0])
            for i in range(10-len(rel_indices)):
                rel_indices.append([0,0,0,0,0,0,0])
            for i in range(10-len(true_triples_encode)):
                true_triples_encode.append([[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]])
            #print("pppppppppppppppppppppppppppp")    
            matrix1 = torch.tensor(matrix1)
            matrix2 = torch.tensor(matrix2)
            matrix3 = torch.tensor(matrix3)
            obj1_indices = torch.tensor(obj1_indices)
            obj2_indices = torch.tensor(obj2_indices)
            rel_indices = torch.tensor(rel_indices)
            true_tries_encode = torch.tensor(true_triples_encode)
            """for triple in triples:
                if triple[0] not in obj1_indices:
                    obj1_indices[triple[0]] = [index1]
                    index1+=1
                else:
                    obj1_indices[triple[0]].append(index1)
                if triple[1] not in rel_indices:
                    rel_indices[triple[1]] = [index2]
                else:
                    rel_indices[triple[1]].append(index2)
                if triple[2] not in obj2_indices:
                    obj2_indices[triple[2]] = [index3]
                    index3+=1
                else:
                    obj2_indices[triple[2]].append(index3)
                
            if '-' in obj1_indices:
                del obj1_indices['-']
            if '-' in rel_indices:
                del rel_indices['-']  
            if '-' in obj2_indices:
                del obj2_indices['-']      
            # 打印结果
            if len(obj1_indices) > 10:
                print("obj1_indices:",len(obj1_indices))
                print("triples:",scene['true_caption'])
            if len(rel_indices) > 10:
                print("rel_indices:",len(rel_indices))
                print("triples:",scene['true_caption'])
            if len(obj2_indices) > 10:
                print("obj2_indices:",len(obj2_indices))
                print("triples:",scene['true_caption'])"""
            #true_caption = scene["true_caption"]
            #false_caption = scene["false_caption"]
            #true_caption = tokenize(true_caption, truncate=True)
            #false_caption = tokenize(false_caption, truncate=True)
            item = edict({"image_options": [tensor_image], 
                        "caption_options": [false_caption_encode, true_caption_encode], 
                        #"caption_options": [false_caption, true_caption], 
                        "relation": "attribution",
                        "head_inputs": head_inputs,
                        "relation_inputs": relation_inputs,
                        "tail_inputs": tail_inputs,
                        "attention_mask": attention_mask,
                        "feature_segs":segs,
                        "triples_segs": [matrix1, matrix2, matrix3],
                        "triples_index": [obj1_indices, rel_indices, obj2_indices],
                        "triples_position":true_tries_encode,
                        "reversed_head_inputs": reversed_head_inputs,
                        "reversed_relation_inputs": reversed_relation_inputs,
                        "reversed_tail_inputs": reversed_tail_inputs,
                        "reversed_attention_mask": reversed_attention_mask})
        else:
            triples = scene['true_triples']
            true_caption = ""
            for triple in triples:
                true_caption += triple[0] + " " + triple[1] + " " + triple[2] + ". "
            true_caption = true_caption[:-1]
            false_caption = ""
            for reversed_triple in reversed_triples:
                false_caption += reversed_triple[0] + " " + reversed_triple[1] + " " + reversed_triple[2] + ". "
            false_caption = false_caption[:-1]
            #true_caption = scene['true_caption']
            #false_caption = scene['false_caption']
            item = edict({"image_options": [tensor_image], 
                        "caption_options": [false_caption, true_caption], 
                        "relation": "attribution",
                        "head_inputs": head_inputs,
                        "relation_inputs": relation_inputs,
                        "tail_inputs": tail_inputs,
                        "attention_mask": attention_mask,
                        #"feature_segs":segs,
                        "reversed_head_inputs": reversed_head_inputs,
                        "reversed_relation_inputs": reversed_relation_inputs,
                        "reversed_tail_inputs": reversed_tail_inputs,
                        "reversed_attention_mask": reversed_attention_mask})

        return item
class NegCLIPData(Dataset):
    def __init__(self, data_path=None, transform=None):

        self.transform = transform
        self.length = 5
        self.padding_num = 6
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if 'adjchange' in data_path:
            self.root_dir = '/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/lvxiaotian/Structure-CLIP/data/'
        else:
            self.root_dir = "/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/lvxiaotian/Structure-CLIP/data/images_attribution/"
        self.data_path = data_path
        
        with open(self.data_path, "r") as f:
            self.dataset = json.load(f)
        for item in self.dataset:
            item["image_path"] = os.path.join(self.root_dir, item["image_path"])
        if 'adjchange' not in data_path:
            self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]

    def __len__(self):
        return len(self.dataset)
    
    def get_triple_token_infor(self, triples):

        head_word_list = list()
        relation_word_list = list()
        tail_word_list = list()

        if len(triples) > 0:
            for triple in triples:
                if len(head_word_list) < self.padding_num:
                    head_word_list.append(triple[0])
                    relation_word_list.append(triple[1])
                    tail_word_list.append(triple[2])
                
        token_type_ids = torch.zeros([1,self.padding_num], dtype=int)

        if len(head_word_list) > 0:
            attention_mask = torch.cat((torch.ones([1, len(head_word_list)], dtype=int), torch.zeros([1, self.padding_num-len(head_word_list)], dtype=int)), dim=1)
        else:
            attention_mask = torch.zeros([1, self.padding_num], dtype=int)
        
        for i in range(self.padding_num - len(head_word_list)):
            head_word_list.append('')
            relation_word_list.append('')
            tail_word_list.append('')
        
        head_inputs = self.tokenizer.batch_encode_plus(
            head_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        relation_inputs = self.tokenizer.batch_encode_plus(
            relation_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        tail_inputs = self.tokenizer.batch_encode_plus(
            tail_word_list,
            max_length=self.length,
            add_special_tokens=True,
            padding = "max_length",
            return_tensors="pt",
            truncation=True
        )
        return head_inputs, relation_inputs, tail_inputs, attention_mask
    
    def __getitem__(self, index):
        scene = self.dataset[index]
        # step1 
        image = Image.open(scene["image_path"]).convert('RGB')
        transform = transforms.ToTensor()
        tensor_image = transform(image)
        true_caption = scene["true_caption"]
        false_caption = scene["false_caption"]
        
        
        if "train" in self.data_path:
            
            #true_caption = scene["true_caption"]
            #false_caption = scene["false_caption"]
            true_caption = tokenize(true_caption, truncate=True)
            false_caption = tokenize(false_caption, truncate=True)
            item = edict({"image_options": [tensor_image], 
                        #"caption_options": [false_caption_encode, true_caption_encode], 
                        "caption_options": [false_caption, true_caption], 
                        })
        else:
            triples = scene['true_triples']
            true_caption = ""
            for triple in triples:
                true_caption += triple[0] + " " + triple[1] + " " + triple[2] + ". "
            true_caption = true_caption[:-1]
            false_caption = ""
            for reversed_triple in reversed_triples:
                false_caption += reversed_triple[0] + " " + reversed_triple[1] + " " + reversed_triple[2] + ". "
            false_caption = false_caption[:-1]
            #true_caption = scene['true_caption']
            #false_caption = scene['false_caption']
            item = edict({"image_options": [tensor_image], 
                        "caption_options": [false_caption, true_caption], 
                        "relation": "attribution",
                        "head_inputs": head_inputs,
                        "relation_inputs": relation_inputs,
                        "tail_inputs": tail_inputs,
                        "attention_mask": attention_mask,
                        #"feature_segs":segs,
                        "reversed_head_inputs": reversed_head_inputs,
                        "reversed_relation_inputs": reversed_relation_inputs,
                        "reversed_tail_inputs": reversed_tail_inputs,
                        "reversed_attention_mask": reversed_attention_mask})

        return item


    
if __name__ == '__main__':
    data =VG_Attribution(data_path='/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/lvxiaotian/Structure-CLIP/data/visual_genome_attribution_aug_winner.json')
    train_vg_dataloader = DataLoader(data, num_workers=1, batch_size=1, shuffle=True)
    for i, batch in enumerate(tqdm(train_vg_dataloader, total=len(train_vg_dataloader))):
        print(batch["feature_segs"].shape)     