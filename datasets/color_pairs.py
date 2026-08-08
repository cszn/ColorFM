from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch.nn.functional as F

class FlowDataset(Dataset):
    def __init__(self, cfg, device, SegModel=None, feature_extractor=None, full=False, seg_mode=True):
        self.cfg = cfg
        self.x_0 = cfg.x_0
        self.x_1 = cfg.x_1
        self.device = device
        self.full = full
        self.seg_mode = seg_mode
        image0 = Image.open(self.x_0).convert('RGB')
        image1 = Image.open(self.x_1).convert('RGB')
        image0 = image0.resize((512, 512))
        image1 = image1.resize((512, 512))

        # if not full:
        # if image0.width * image0.height > 1920 * 1080:
        #     aspect_ratio = image0.width / image0.height
        #     if aspect_ratio >= 16/9:
        #         image0 = image0.resize((int(1080 * aspect_ratio), 1080))
        #     else:
        #         image0 = image0.resize((1920, int(1920 / aspect_ratio)))
        # if image1.width * image1.height > 1920 * 1080:
        #     aspect_ratio = image1.width / image1.height
        #     if aspect_ratio >= 16/9:
        #         image1 = image1.resize((int(1080 * aspect_ratio), 1080))
        #     else:
        #         image1 = image1.resize((1920, int(1920 / aspect_ratio))) 
        

        x_0 = T.ToTensor()(image0).to(self.device)
        x_1 = T.ToTensor()(image1).to(self.device)
        H_0, W_0 = x_0.shape[1], x_0.shape[2]
        H_1, W_1 = x_1.shape[1], x_1.shape[2]

        if self.seg_mode:
            if SegModel is None:
                SegModel = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b5-finetuned-ade-640-640",
                    local_files_only=True,
                ).to(self.device).eval()
            if feature_extractor is None:
                feature_extractor = SegformerImageProcessor.from_pretrained(
                    "nvidia/segformer-b5-finetuned-ade-640-640",
                    local_files_only=True,
                )

            image_seg = feature_extractor(images=x_0, return_tensors="pt", do_rescale=False)
            style_seg = feature_extractor(images=x_1, return_tensors="pt", do_rescale=False)
            image_seg.to(self.device)
            style_seg.to(self.device)

            with torch.no_grad():
                image_seg_outputs = SegModel(**image_seg).logits.contiguous()
                style_seg_outputs = SegModel(**style_seg).logits.contiguous()
            image_seg = F.interpolate(image_seg_outputs, size=(H_0, W_0), mode='bilinear', align_corners=False).argmax(dim=1).unsqueeze(1)
            style_seg = F.interpolate(style_seg_outputs, size=(H_1, W_1), mode='bilinear', align_corners=False).argmax(dim=1).unsqueeze(1)
        else:
            image_seg = torch.ones((1, 1, H_0, W_0), dtype=torch.long, device=self.device)
            style_seg = torch.ones((1, 1, H_1, W_1), dtype=torch.long, device=self.device)

        image_seg, style_seg = self.process_seg_map(image_seg, style_seg, self.seg_mode)

        x_0 = torch.cat([x_0, image_seg], dim=0)
        x_1 = torch.cat([x_1, style_seg], dim=0)
        x_0 = x_0.view(4, -1).permute(1, 0)
        x_1 = x_1.view(4, -1).permute(1, 0)

        classes_num = torch.unique(image_seg)
        self.num = len(classes_num)
        self.pairs = []
        max_pairs_per_class = 0

        for class_num in classes_num:
            x_0_class = x_0[x_0[:, 3] == class_num]
            x_1_class = x_1[x_1[:, 3] == class_num]

            if int(class_num.item()) == 1 and len(x_1_class) == 0:
                x_1_class = x_1
            x_0_class = x_0_class[:, :3]
            x_1_class = x_1_class[:, :3]
            pairs_class = self.hierarchical_color_coupling_partition(x_0_class, x_1_class, 0, 3)

            if len(pairs_class) == 0:
                continue

            pairs_class = torch.cat(pairs_class, dim=0)
            max_pairs_per_class = max(max_pairs_per_class, pairs_class.shape[0])
            self.pairs.append(pairs_class)

        if len(self.pairs) == 0:
            fallback_pairs = self.random_color_pairs(x_0[:, :3], x_1[:, :3])
            if len(fallback_pairs) == 0:
                raise ValueError("No valid color pairs can be generated from the input images.")
            self.pairs = [fallback_pairs[0]]
            max_pairs_per_class = self.pairs[0].shape[0]

        for i, pair_class in enumerate(self.pairs):
            leng = pair_class.shape[0]
            if leng < max_pairs_per_class:
                repeats = (max_pairs_per_class + leng - 1) // leng
                pair_class = pair_class.repeat((repeats, 1, 1))[:max_pairs_per_class]
                self.pairs[i] = pair_class
        self.pairs = torch.cat(self.pairs, dim=0)

        min_batch_size = 4096 
        current_len = self.pairs.shape[0]

        if current_len < min_batch_size:
            repeats = (min_batch_size // current_len) + 1
            self.pairs = self.pairs.repeat((repeats, 1, 1))
            
        self.pairs = self.pairs.cpu()

        self.real_len = len(self.pairs)

    def hierarchical_color_coupling_partition(self, x0, x1, depth, max_depth):
        if depth == max_depth or min(len(x0), len(x1)) == 0:
            return self.random_color_pairs(x0, x1)

        center0 = x0.mean(dim=0)
        center1 = x1.mean(dim=0)

        x0_centered = x0 - center0
        x1_centered = x1 - center1

        oct_0 = self.octant_indices(x0_centered)
        oct_1 = self.octant_indices(x1_centered)

        pairs_list = []

        for oct_num in range(8):
            idx0 = torch.where(oct_0 == oct_num)[0]
            idx1 = torch.where(oct_1 == oct_num)[0]

            if len(idx0) == 0 or len(idx1) == 0:
                continue

            sub_pairs = self.hierarchical_color_coupling_partition(
                x0_centered[idx0] + center0,
                x1_centered[idx1] + center1,
                depth + 1,
                max_depth
            )
            pairs_list.extend(sub_pairs)

        if len(pairs_list) == 0:
            return self.random_color_pairs(x0, x1)

        return pairs_list

    @staticmethod
    def random_color_pairs(x0, x1):
        n = min(len(x0), len(x1))
        if n == 0:
            return []

        idx0 = torch.randperm(len(x0), device=x0.device)[:n]
        idx1 = torch.randperm(len(x1), device=x1.device)[:n]
        return [torch.stack([x0[idx0], x1[idx1]], dim=1)]
        
    def octant_indices(self, points):
        signs = (points >= 0).int()
        octants = signs[:, 0]*4 + signs[:, 1]*2 + signs[:, 2]
        return octants

    @staticmethod
    def process_seg_map(image_seg, style_seg, seg_mode):
        image_seg = image_seg.squeeze(0)
        style_seg = style_seg.squeeze(0)

        C, H, W = image_seg.size()

        style_classes = torch.unique(style_seg)
        image_classes = torch.unique(image_seg)

        valid_class_map = {}
        index = 2

        # pair image class to style class
        for class_label in style_classes:
            if class_label in image_classes:
                valid_class_map[class_label] = index
                index += 1
                # print(class_label, index)

        image_seg_new = torch.ones_like(image_seg)
        style_seg_new = torch.ones_like(style_seg)

        if not seg_mode:
            return image_seg_new, style_seg_new

        for old_label, new_label in valid_class_map.items():
            image_seg_new[image_seg == old_label.item()] = new_label
            style_seg_new[style_seg == old_label.item()] = new_label
        return image_seg_new, style_seg_new
        
    def __len__(self):
        base_len = len(self.pairs)
        target = 700 * 4096
        if base_len >= target:
            return base_len    
        multiplier = (target // base_len) + 1    
        return base_len * multiplier

    def __getitem__(self, idx):
        idx = idx % self.real_len
        p_0 = self.pairs[idx][0]
        p_1 = self.pairs[idx][1]

        return {
            'p_0': p_0,
            'p_1': p_1,
            'path': self.cfg.path,
        }

def get_loader(cfg, device, SegModel, feature_extractor, full=False, seg_mode=True):
    dataset = FlowDataset(cfg, device, SegModel, feature_extractor, full, seg_mode)
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    loader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        drop_last = True
    )

    return loader