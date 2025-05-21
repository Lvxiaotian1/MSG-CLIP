import argparse
import random
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from PIL import Image
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
import sng_parser

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/coco_dataset_train.txt', type=str)
    parser.add_argument('--test_path', default='data/coco_dataset_test.txt', type=str)
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument('--manualSeed', type=int, default=120)
    parser.add_argument('--batch_size', default='128', type=int)
    parser.add_argument('--lr', default='2e-7', type=float)
    parser.add_argument('--epoch', default='10', type=int)
    parser.add_argument('--weight_decay', default='0.1', type=float)
    parser.add_argument('--knowledge_weight', default='0.01', type=float)
    parser.add_argument('--transformer_layer_num', default='4', type=int)
    parser.add_argument('--neg_loss_weight', default='1', type=float)
    

    return parser.parse_args()

def stable_match(sim):
    softmax_sim = torch.softmax(torch.from_numpy(sim), -1).numpy() + torch.softmax(torch.from_numpy(sim), 0).numpy()
    sim_shape = softmax_sim.shape
    softmax_sim = np.reshape(softmax_sim, (1, -1))
    sort_sim = np.argsort(-softmax_sim)
    
    x, y = sim_shape
    result = {}
    book = set([])
    book1 = set([])
    zzz = 0
    loss = 0
    for _, i in enumerate(sort_sim.tolist()[0]):
        if len(result.keys()) == sim_shape[0]:
            break
        a = i // y
        b = i % y# + x
        if a in book or b in book1:
            continue
        if sim[a][b]>0.5:
            result[a] = b
            book.add(a)
            book1.add(b)  
    c = 0
    for a, b in result.items():
        min1 = np.min(sim[a])
        min2 = np.min(sim[:,b])
        
        loss = loss + (1-sim[a][b]+min1+min2)/3
        zzz+=1
    if zzz>0:
        return loss/zzz,result
    else:
        return 0,result
def stable_match_f(sim):
    softmax_sim = torch.softmax(torch.from_numpy(sim), -1).numpy() + torch.softmax(torch.from_numpy(sim), 0).numpy()
    sim_shape = softmax_sim.shape
    softmax_sim = np.reshape(softmax_sim, (1, -1))
    sort_sim = np.argsort(-softmax_sim)
    

    x, y = sim_shape
    result = {}
    book = set([])
    book1 = set([])
    zzz = 0
    loss = 0
    for _, i in enumerate(sort_sim.tolist()[0]):
        if len(result.keys()) == sim_shape[0]:
            break
        a = i // y
        b = i % y# + x
        if a in book or b in book1:
            continue
        if sim[a][b]>0:
            result[a] = b
            book.add(a)
            book1.add(b)
    c = 0
    for a, b in result.items():
        min1 = np.min(sim[a])
        min2 = np.min(sim[:,b])
        loss = loss + (1-sim[a][b]+min1+min2)/3
        zzz+=1
    return loss/zzz,result
def senceGraph_loss(image_features, text_true_features, feature_segs, obj1_segs,obj1_index,rel_segs,rel_index,obj2_segs,obj2_index,triples_position,flag):
    batch_size = image_features.shape[0]
    scence_loss = 0
    for i in range(batch_size):
        img = image_features[i][1:]
        #print(img.shape)
        txt = text_true_features[i]
        segs = feature_segs[i]
        obj1s = obj1_segs[i]
        indexs = obj1_index[i]
        rels = rel_segs[i]
        rel_indexs = rel_index[i]
        obj2s = obj2_segs[i]
        obj2s_index = obj2_index[i]
        triples_positions = triples_position[i]
        #print(obj1s.shape)
        #print(segs.shape)
        row_sums = torch.sum(segs, dim=1)
        non_zero_rows = row_sums != 0
        segs_no_zero = segs[non_zero_rows]
        segs_lenth = segs_no_zero.shape[0]
        #print(segs_lenth)
        features = []
        for zzz in range(segs_lenth):
            feature =img[segs_no_zero[zzz]].norm(dim=0, keepdim=True)
            features.append(feature)
        final_features = torch.cat(features, dim=0)
        final_features = final_features/final_features.norm(dim=1, keepdim=True)
        row_sums1 = torch.sum(obj1s, dim=1)
        non_zero_rows1 = row_sums1 != 0
        obj1s_no_zero = obj1s[non_zero_rows1]
        obj1s_lenth = obj1s_no_zero.shape[0]
        
        row_sums2 = torch.sum(indexs, dim=1)
        non_zero_rows2 = row_sums1 != 0
        indexs_no_zero = indexs[non_zero_rows2]
        indexs_lenth = indexs_no_zero.shape[0]
        
        row_sums3 = torch.sum(rels, dim=1)
        non_zero_rows3 = row_sums3 != 0
        rels_no_zero = rels[non_zero_rows3]
        rels_lenth = rels_no_zero.shape[0]
        
        row_sums4 = torch.sum(rel_indexs, dim=1)
        non_zero_rows4 = row_sums4 != 0
        rel_indexs_no_zero = rel_indexs[non_zero_rows4]
        rel_indexs_lenth = rel_indexs_no_zero.shape[0]
        
        row_sums5 = torch.sum(obj2s, dim=1)
        non_zero_rows5 = row_sums5 != 0
        obj2s_no_zero = obj2s[non_zero_rows5]
        obj2s_lenth = obj2s_no_zero.shape[0]
        
        row_sums6 = torch.sum(obj2s_index, dim=1)
        non_zero_rows6 = row_sums6 != 0
        obj2s_index_no_zero = obj2s_index[non_zero_rows6]
        obj2s_index_lenth = obj2s_index_no_zero.shape[0]
        
        obj1s_features = []
        for yyy in range(obj1s_lenth):
            feature1 =txt[obj1s_no_zero[yyy]].norm(dim=0, keepdim=True)
            obj1s_features.append(feature1)
        final_obj1s_features = torch.cat(obj1s_features, dim=0)
        final_obj1s_features = final_obj1s_features/final_obj1s_features.norm(dim=1, keepdim=True)
        
        obj2s_features = []
        for yyy in range(obj2s_lenth):
            feature2 =txt[obj2s_no_zero[yyy]].norm(dim=0, keepdim=True)
            obj2s_features.append(feature2)
        final_obj2s_features = torch.cat(obj2s_features, dim=0)
        final_obj2s_features = final_obj2s_features/final_obj2s_features.norm(dim=1, keepdim=True)
        
        rels_features = []
        for yyy in range(rels_lenth):
            feature3 =txt[rels_no_zero[yyy]].norm(dim=0, keepdim=True)
            rels_features.append(feature3)
        final_rels_features = torch.cat(rels_features, dim=0)
        final_rels_features = final_rels_features/final_rels_features.norm(dim=1, keepdim=True)
        
        final_features = final_features.to('cpu')
        final_obj1s_features = final_obj1s_features.to('cpu')
        final_rels_features = final_rels_features.to('cpu')
        final_obj2s_features = final_obj2s_features.to('cpu')
        
        wwt=0.75
        
        if final_obj1s_features.shape[0] ==1:
            logit_obj1tofeatures = final_obj1s_features @ final_features.T
            max_values,max_indices = torch.max(logit_obj1tofeatures,dim = -1)
            feature_img = final_features[max_indices[0]].shape
            #print(torch.where(triples_positions == indexs_no_zero[0]))
            matching_indices =[]
            for i in range(triples_positions.shape[0]):
                for j in range(triples_positions.shape[1]):
                    if torch.equal(triples_positions[i,j,:],indexs_no_zero[0]):
                        matching_indices.append((i,j))
            relation_loss = 0
            flagzzz = 0
            for i,j in matching_indices:
                if j ==0:
                    rel = triples_position[i,1,:]
                    obj2 = triples_position[i,2,:]
                    matching_indices1 = []
                    matching_indices2 = []
                    for hhh in range(rels_lenth):
                        if torch.equal(rel,rels_no_zero[hhh]):
                            matching_indices1.append(hhh)
                            break
                    if len(matching_indices1)==0:
                        continue
                    else:
                        for hzs in range(obj2s_lenth):
                            if torch.equal(obj2,obj2s_no_zero[hzs]):
                                matching_indices2.append(hzs)
                                break
                        if len(matching_indices2)==0:
                            continue
                        else:
                            obj2_feature =final_obj2s_features[matching_indices2[0]]
                            rel_feature =final_rels_features[matching_indices1[0]]
                            relation_loss += torch.mean((obj2_feature - rel_feature - feature_img)**2)
                            flagzzz+=1
            
                    
            #indices_1 ,indices_2 = torch.where(triples_positions == indexs_no_zero[0])
            #erri = torch.where(indeces_2 !=0)
            #if erri.numel()==0:
            #    print("error")
            
            if relation_loss == 0:
                scence_loss += (1 - max_values + torch.min(logit_obj1tofeatures,dim = -1)[0])/2
            else:
                relation_loss = relation_loss/flagzzz   
                scence_loss += (wwt*((1 - max_values + torch.min(logit_obj1tofeatures,dim = -1)[0])/2)+(1-wwt)*relation_loss)
        elif final_features.shape[0] == 1:
            logitfeaaturestoobj1 = final_features @ final_obj1s_features.T
            max_values,max_indices = torch.max(logitfeaaturestoobj1,dim = -1)
            obj1 =indexs_no_zero[max_indices[0]]
            
            feature_img = final_features[0]
            
            matching_indices =[]
            for i in range(triples_positions.shape[0]):
                for j in range(triples_positions.shape[1]):
                    if torch.equal(triples_positions[i,j,:],obj1):
                        matching_indices.append((i,j))
            relation_loss = 0
            flagzzz = 0
            for i,j in matching_indices:
                if j ==0:
                    rel = triples_position[i,1,:]
                    obj2 = triples_position[i,2,:]
                    matching_indices1 = []
                    matching_indices2 = []
                    for hhh in range(rels_lenth):
                        if torch.equal(rel,rels_no_zero[hhh]):
                            matching_indices1.append(hhh)
                            break
                    if len(matching_indices1)==0:
                        continue
                    else:
                        for hzs in range(obj2s_lenth):
                            if torch.equal(obj2,obj2s_no_zero[hzs]):
                                matching_indices2.append(hzs)
                                break
                        if len(matching_indices2)==0:
                            continue
                        else:
                            obj2_feature =final_obj2s_features[matching_indices2[0]]
                            rel_feature =final_rels_features[matching_indices1[0]]
                            relation_loss += torch.mean((obj2_feature - rel_feature - feature_img)**2)
                            flagzzz+=1
                if j ==2:
                    rel = triples_position[i,1,:]
                    obj1_1 = triples_position[i,0,:]
                    matching_indices1 = []
                    matching_indices2 = []
                    for hhh in range(rels_lenth):
                        if torch.equal(rel,rels_no_zero[hhh]):
                            matching_indices1.append(hhh)
                            break
                    if len(matching_indices1)==0:
                        continue
                    else:
                        for hzs in range(obj1s_lenth):
                            if torch.equal(obj1_1,obj1s_no_zero[hzs]):
                                matching_indices2.append(hzs)
                                break
                        if len(matching_indices2)==0:
                            continue
                        else:
                            obj1_feature =final_obj1s_features[matching_indices2[0]]
                            rel_feature =final_rels_features[matching_indices1[0]]
                            relation_loss += torch.mean((-obj1_feature - rel_feature + feature_img)**2)
                            flagzzz+=1
            
            
            if relation_loss == 0:
                scence_loss += (1 - max_values + torch.min(logitfeaaturestoobj1,dim = -1)[0])/2
            else:   
                relation_loss = relation_loss/flagzzz
                scence_loss += (wwt*(((1 - max_values + torch.min(logitfeaaturestoobj1,dim = -1)[0])/2))+(1-wwt)*relation_loss)   
            
        else:
            logit_obj1tofeatures = final_obj1s_features @ final_features.T
            if flag == 0:
                los1,result = stable_match(logit_obj1tofeatures.detach().numpy())
                if len(result) == 0:
                    scence_loss += los1
                else:
                    relation_loss = 0
                    flagzzz = 0
                    for x,y in result.items():
                        obj1 =indexs_no_zero[x]
            
                        feature_img = final_features[y]
                        
                        matching_indices =[]
                        for i in range(triples_positions.shape[0]):
                            for j in range(triples_positions.shape[1]):
                                if torch.equal(triples_positions[i,j,:],obj1):
                                    matching_indices.append((i,j))
                        
                        for i,j in matching_indices:
                            if j ==0:
                                rel = triples_position[i,1,:]
                                obj2 = triples_position[i,2,:]
                                matching_indices1 = []
                                matching_indices2 = []
                                for hhh in range(rels_lenth):
                                    if torch.equal(rel,rels_no_zero[hhh]):
                                        matching_indices1.append(hhh)
                                        break
                                if len(matching_indices1)==0:
                                    continue
                                else:
                                    for hzs in range(obj2s_lenth):
                                        if torch.equal(obj2,obj2s_no_zero[hzs]):
                                            matching_indices2.append(hzs)
                                            break
                                    if len(matching_indices2)==0:
                                        continue
                                    else:
                                        obj2_feature =final_obj2s_features[matching_indices2[0]]
                                        rel_feature =final_rels_features[matching_indices1[0]]
                                        relation_loss += torch.mean((obj2_feature - rel_feature - feature_img)**2)
                                        flagzzz+=1
                            if j ==2:
                                rel = triples_position[i,1,:]
                                obj1_1 = triples_position[i,0,:]
                                matching_indices1 = []
                                matching_indices2 = []
                                for hhh in range(rels_lenth):
                                    if torch.equal(rel,rels_no_zero[hhh]):
                                        matching_indices1.append(hhh)
                                        break
                                if len(matching_indices1)==0:
                                    continue
                                else:
                                    for hzs in range(obj1s_lenth):
                                        if torch.equal(obj1_1,obj1s_no_zero[hzs]):
                                            matching_indices2.append(hzs)
                                            break
                                    if len(matching_indices2)==0:
                                        continue
                                    else:
                                        obj1_feature =final_obj1s_features[matching_indices2[0]]
                                        rel_feature =final_rels_features[matching_indices1[0]]
                                        relation_loss += torch.mean((-obj1_feature - rel_feature + feature_img)**2)
                                        flagzzz+=1
                    
                    if relation_loss == 0:
                        scence_loss += los1
                    else:
                        relation_loss = relation_loss/flagzzz
                        scence_loss += wwt*los1 +(1-wwt)*relation_loss   
            else:
                los2,result = stable_match_f(logit_obj1tofeatures.detach().numpy())
                if len(result) == 0:
                    scence_loss += los2
                else:
                    relation_loss = 0
                    flagzzz = 0
                    for x,y in result.items():
                        obj1 =indexs_no_zero[x]
            
                        feature_img = final_features[y]
                        
                        matching_indices =[]
                        for i in range(triples_positions.shape[0]):
                            for j in range(triples_positions.shape[1]):
                                if torch.equal(triples_positions[i,j,:],obj1):
                                    matching_indices.append((i,j))
                        
                        for i,j in matching_indices:
                            if j ==0:
                                rel = triples_position[i,1,:]
                                obj2 = triples_position[i,2,:]
                                matching_indices1 = []
                                matching_indices2 = []
                                for hhh in range(rels_lenth):
                                    if torch.equal(rel,rels_no_zero[hhh]):
                                        matching_indices1.append(hhh)
                                        break
                                if len(matching_indices1)==0:
                                    continue
                                else:
                                    for hzs in range(obj2s_lenth):
                                        if torch.equal(obj2,obj2s_no_zero[hzs]):
                                            matching_indices2.append(hzs)
                                            break
                                    if len(matching_indices2)==0:
                                        continue
                                    else:
                                        obj2_feature =final_obj2s_features[matching_indices2[0]]
                                        rel_feature =final_rels_features[matching_indices1[0]]
                                        relation_loss += torch.mean((obj2_feature - rel_feature - feature_img)**2)
                                        flagzzz+=1
                            if j ==2:
                                rel = triples_position[i,1,:]
                                obj1_1 = triples_position[i,0,:]
                                matching_indices1 = []
                                matching_indices2 = []
                                for hhh in range(rels_lenth):
                                    if torch.equal(rel,rels_no_zero[hhh]):
                                        matching_indices1.append(hhh)
                                        break
                                if len(matching_indices1)==0:
                                    continue
                                else:
                                    for hzs in range(obj1s_lenth):
                                        if torch.equal(obj1_1,obj1s_no_zero[hzs]):
                                            matching_indices2.append(hzs)
                                            break
                                    if len(matching_indices2)==0:
                                        continue
                                    else:
                                        obj1_feature =final_obj1s_features[matching_indices2[0]]
                                        rel_feature =final_rels_features[matching_indices1[0]]
                                        relation_loss += torch.mean((-obj1_feature - rel_feature + feature_img)**2)
                                        flagzzz+=1
                    
                    if relation_loss == 0:
                        scence_loss += los2
                    else:
                        relation_loss = relation_loss/flagzzz
                        scence_loss += wwt*los2 +(1-wwt)*relation_loss
            #print(logit_obj1tofeatures.shape)
        #print(final_obj1s_features.shape)
    return float(scence_loss/batch_size)
def set_manualSeed(args):
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

# image transform
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def image_transform(is_train = True, num_pix = 224):
    if is_train:
        return Compose([
            RandomResizedCrop(num_pix, scale=(0.9, 1.0), interpolation=BICUBIC),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        return Compose([
            Resize(num_pix, interpolation=BICUBIC),
            CenterCrop(num_pix),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


def swap_words(s, x, y):
    return y.join(part.replace(y, x) for part in s.split(x))

class scene_graph():
    def __init__(self, text):
        self.graph = sng_parser.parse(text)
        self.text = text
    def swap_words(self, s, x, y):
        return y.join(part.replace(y, x) for part in s.split(x))
    def neg_gen(self):
        neg_text = []
        for relation in self.graph['relations']:
            subject = self.graph['entities'][relation['subject']]['span']

            object = self.graph['entities'][relation['object']]['span']
            temp = self.swap_words(self.text, subject, object)
            neg_text.append(temp.lower())
        if neg_text == []:
            print(self.text)
        return neg_text[0]

def compute_logits(image_features, text_true_features, text_gen_features, logit_scale):
    logit_img2text_true = logit_scale * image_features @ text_true_features.T
    logit_text_true2img = logit_scale * text_true_features @ image_features.T
    logit_img2text_gen = logit_scale * image_features @ text_gen_features.T
    logit_text_gen2img = logit_scale * text_gen_features @ image_features.T
    device = image_features.device
    num_logits = image_features.shape[0]
    logit_diag_true = torch.diag(logit_img2text_true)
    logit_diag_gen = torch.diag(logit_img2text_gen)
    x = logit_diag_true - logit_diag_gen
    Wino_label = torch.tensor([1] * num_logits, device=device)

    return logit_diag_true, logit_diag_gen, Wino_label

class WinoLoss(nn.Module):
    def __init__(self, margin):
        super(WinoLoss, self).__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='mean')
        self.relu = nn.ReLU()
    def forward(self, image_features, text_features, logit_scale, is_hard):


        # last
        batch_size = image_features.shape[0]
        cos_img2text = torch.matmul(image_features, text_features.T) # [bs,bs]

        pos_score = torch.diag(cos_img2text) #[bs]
        img_neg_score = torch.max(cos_img2text - 10 * torch.eye(batch_size, requires_grad=False).cuda(0), dim=-1)[0] # [bs]

        cos_text2img = cos_img2text.T #[bs,bs]
        if is_hard:
            text_neg_score = torch.max(cos_text2img - 10 * torch.eye(batch_size, requires_grad=False).cuda(0), dim=-1)[0] # [bs]
        else: 
            text_neg_score = torch.mean(cos_text2img, dim=-1)
        # text_neg_score = torch.max(cos_text2img - 10 * torch.eye(batch_size, requires_grad=False).cuda(0), dim=-1)[0] # [bs]
        margin = torch.ones_like(pos_score, requires_grad=False) * 0.2  #[bs]

        loss = self.relu(img_neg_score + margin - pos_score) + self.relu(text_neg_score + margin - pos_score)
        loss = torch.mean(loss)
        return loss

class MarginLoss(nn.Module):
    def __init__(self, margin):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    def forward(self, image_features, text_true_features, text_gen_features, logit_scale):
        logit_img2text_true = logit_scale * image_features @ text_true_features.T
        logit_text_true2img = logit_scale * text_true_features @ image_features.T
        logit_img2text_gen = logit_scale * image_features @ text_gen_features.T
        logit_text_gen2img = logit_scale * text_gen_features @ image_features.T
        device = image_features.device
        num_logits = image_features.shape[0]
        logit_diag_true = torch.diag(logit_img2text_true)
        logit_diag_gen = torch.diag(logit_img2text_gen)

        clip_label = torch.arange(num_logits, device=device, dtype=torch.long)
        Wino_label = torch.tensor([1] * num_logits, device=device)
        total_loss = self.loss(logit_diag_true, logit_diag_gen, Wino_label)

        return total_loss

class MyLoss(nn.Module):
    def __init__(self, margin):
        super(MyLoss, self).__init__()
        self.margin = margin
        self.loss = nn.ReLU()
    def forward(self, image_feat, pos_GCN_emb, neg_GCN_emb, pos_text_feat, neg_text_feat):
        alpha = 1.0
        pos_GCN_emb = pos_GCN_emb / pos_GCN_emb.norm(dim=-1, keepdim=True)
        neg_GCN_emb = neg_GCN_emb / neg_GCN_emb.norm(dim=-1, keepdim=True)
        pos_text = pos_text_feat * 0.0 + alpha * pos_GCN_emb
        neg_text = neg_text_feat * 0.0 + alpha * neg_GCN_emb
        
        img2pos = image_feat @ pos_text.T
        img2neg = image_feat @ neg_text.T
        margin = 0.2
        pos_score = torch.diag(img2pos)
        neg_score = torch.diag(img2neg)
        loss = self.loss(neg_score - pos_score + margin)
        return loss

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_embed, text_embed):

        local_batch_size = image_embed.size(0)


        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(local_batch_size, device=image_embed.device)
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = self.scale * image_embed @ text_embed_all.t()
        logits_per_text = self.scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + F.cross_entropy(logits_per_text[:len(logits_per_image)], self.labels)) / 2

        # compute accuracy

        return loss


def triple2emb(head, rel, tail, clip_model):
    gamma = 1.0
    gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
    
    head_emb = clip_model.encode_text(head)
    rel_emb = clip_model.encode_text(rel)
    tail_emb = clip_model.encode_text(tail)
    # transe求emb
    score = head_emb + rel_emb - tail_emb
    # score = gamma.item() - torch.norm(score, p=1, dim=-1)
    
    return score


