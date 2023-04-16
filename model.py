import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetTokenizer, XLNetModel
import torchvision
from torchvision import models, transforms

# Create the Bert custom class 
class TextEncoder(nn.Module):

    def __init__(self, dropout_p=0.4, fine_tune_module=False):

        super(TextEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module

        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased',
                                                return_dict=True,
                                               )

        self.text_enc_fc1 = torch.nn.Linear(768, 500)
        self.text_enc_fc2 = torch.nn.Linear(500, 100)
        self.bn = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout(dropout_p)
        
        if fine_tune_module:
            self.fine_tune()

    def forward(self, input_ids, attention_mask):

        # Feed input to BERT
        out = self.xlnet(input_ids=input_ids,
                            attention_mask=attention_mask, output_attentions=True)
        attentions = out['attentions'][-1]
        x = out['last_hidden_state'][:,-1,:]
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.text_enc_fc1(x))
        x = torch.nn.functional.relu(self.text_enc_fc2(x))
        x = self.bn(x)
        x = self.dropout(x)

        return x, attentions
    
            
    def fine_tune(self):
        for parameter in self.xlnet.parameters():
            parameter.requires_grad = False

        layer = self.xlnet.layer[-2:]
        for parameter in layer.parameters():
            parameter.requires_grad = True

# Create the Bert custom class 
class VisionEncoder(nn.Module):
    def __init__(self, dropout_p=0.4, fine_tune_module=False):
        super(VisionEncoder, self).__init__()
        
        self.fine_tune_module = fine_tune_module

        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        
        self.vis_encoder = vgg

        self.vis_enc_fc1 = torch.nn.Linear(4096, 2000)
        self.vis_enc_fc2 = torch.nn.Linear(2000, 2000)
        self.vis_enc_fc3 = torch.nn.Linear(2000, 1000)
        self.vis_enc_fc4 = torch.nn.Linear(1000, 100)
        self.dropout = nn.Dropout(dropout_p)
        self.bn = nn.BatchNorm1d(100)

        if fine_tune_module:
            self.fine_tune()
        
        
    def forward(self, images):
        x = self.vis_encoder(images)
        x = torch.nn.functional.relu(self.vis_enc_fc1(x))
        x = torch.nn.functional.relu(self.vis_enc_fc2(x))
        x = torch.nn.functional.relu(self.vis_enc_fc3(x))
        x = torch.nn.functional.relu(self.vis_enc_fc4(x))
        x = self.bn(x)
        x = self.dropout(x)
        
        return x
    
    def fine_tune(self):
        for p in self.vis_encoder.features.parameters():
            p.requires_grad = False
        
        module = self.vis_encoder.features[-5:]
        for p in module.parameters():
            p.require_grad = True


class LanguageAndVisionConcatWithExtra(torch.nn.Module):
    def __init__(
        self,
        model_params
        
    ):
        super(LanguageAndVisionConcatWithExtra, self).__init__()
        
        self.text_encoder = TextEncoder(model_params['dropout_p'], 
                                        model_params['fine_tune_text_module'])
        self.vision_encoder = VisionEncoder(model_params['dropout_p'],
                                            model_params['fine_tune_vis_module'])
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 6)
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])

    def forward(self, text, image, label=None):

        ## Pass the text input to encoder 
        text_features, attentions = self.text_encoder(text[0], text[1])
        ## Pass the image input 
        image_features = self.vision_encoder(image)


        ## concatenating Image and text 
        combined_features = torch.cat(
            [text_features, image_features], dim=1
        )
        fused = self.dropout(combined_features)
        fused = torch.nn.functional.relu(self.fc1(combined_features))
        fused = torch.nn.functional.relu(self.fc2(fused))
        fused = torch.nn.functional.relu(self.dropout(self.fc3(fused)))
        
        prediction = torch.nn.functional.sigmoid(fused)

        return prediction, attentions