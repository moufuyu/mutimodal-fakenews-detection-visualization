from model import LanguageAndVisionConcatWithExtra
from dataset import FakeNewsDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import set_seed
from evaluate import evaluate
import gc
import os
import time
import argparse
import pandas as pd
from PIL import Image
from PIL import ImageFile
import torch
from torch import nn
import torchvision
from torchvision import models, transforms
from transformers import XLNetTokenizer, XLNetModel
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', 
                        default='/working/my-fake-detect/fakeddit/fakeddit_split_dataset/RE_LARGE_extra_all_keys_2len_all_test.csv',
                        help='Path to test datasets')
    parser.add_argument('--max_len', default=250, type=int,
                        help='Limit of tokens')
    parser.add_argument('--batch_size', default=400, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--resize_size', default=224, type=int,
                        help='Resize the input image to the given size.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')


    
    opt = parser.parse_args()
    print(opt)
    
    model_name = "best_model.pt"
    
    # For descriptive error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    # Load the BERT tokenizer
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    MAX_LEN = opt.max_len
    df_test = pd.read_csv(opt.test_data_path)
    df_test = df_test[:500]

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    transformed_dataset_test = FakeNewsDataset(df_test, image_transform, tokenizer, MAX_LEN)

    test_dataloader = DataLoader(transformed_dataset_test, batch_size=opt.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    parameter_dict_model={
        'fine_tune_text_module': False,
        'dropout_p': 0.4, 
        'fine_tune_vis_module': False
    }
    
    # モデルの宣言・設定
    model = LanguageAndVisionConcatWithExtra(parameter_dict_model)
    state = torch.load(f"/working/my-fake-detect/fakeddit/fakeddit_python_file/saved_models/{best_model}")
    model.eval()
    model.to(device)
    model.load_state_dict(state['model_state_dict'])


    set_seed(42)

    loss_fn = nn.CrossEntropyLoss()

    test_loss, test_accuracy, predictions = evaluate(model=model, loss_fn=loss_fn, val_dataloader=test_dataloader,  device=device)
    print("test_accuracy: ", test_accuracy)


if __name__ == '__main__':
    main()