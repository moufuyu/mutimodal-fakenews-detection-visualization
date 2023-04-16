import re
import ast
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
from skimage import io, transform
from torchvision import models, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import external_preprocess, text_preprocessing


class FakeNewsDataset(Dataset):
    """Fake News Dataset"""

    def __init__(self, df, image_transform, tokenizer, MAX_LEN):

        self.df = df
        self.file_names = df["file_path"].values
        self.image_transform = image_transform
        self.tokenizer_xlnet = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return self.df.shape[0]
    
    def pre_processing_XLNet(self, sent):
        # Create empty lists to store outputs
        input_ids = []
        attention_mask = []
        
        encoded_sent = self.tokenizer_xlnet.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=self.MAX_LEN,                  # Max length to truncate/pad
            padding='max_length',         # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            truncation=True
            )
        
        input_ids = encoded_sent.get('input_ids')
        attention_mask = encoded_sent.get('attention_mask')
        
        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return input_ids, attention_mask
     
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # image
        file_path = self.file_names[idx]

        image = Image.open(file_path).convert("RGB")
        
        torch_image = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])(image)
        
        image = self.image_transform(image)
        #print(image)
    
        text = self.df['title'].values[idx]
        text = text_preprocessing(str(text))
        # 外部知識
        keyword = self.df['keyword'].values[idx]
        introduction = self.df['introduction'].values[idx]
        
        # introductionを前処理
        intro_list = introduction
        extra_text = ""
        if not pd.isna(introduction):
            intro_list = external_preprocess(ast.literal_eval(str(introduction)))
            extra_text = self.tokenizer_xlnet.sep_token
            for step, intro in enumerate(intro_list):
                extra_text += intro
                if step != len(intro_list) - 1:
                    extra_text += self.tokenizer_xlnet.sep_token

        # 追加するときはextra_textを足す
        tensor_input_id, tensor_input_mask = self.pre_processing_XLNet(text + extra_text)

        label = self.df['6_way_label'].values[idx]
        label = torch.tensor(label)

        sample = {'original_image':torch_image, 'image': image, 'text':text, 'BERT_ip': [tensor_input_id, tensor_input_mask],  'label':label}

        return sample