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
    parser.add_argument('--train_data_path', 
                        default='/working/my-fake-detect/fakeddit/fakeddit_split_dataset/RE_LARGE_extra_all_keys_2len_all_train.csv',
                        help='Path to train datasets')
    parser.add_argument('--valid_data_path', 
                        default='/working/my-fake-detect/fakeddit/fakeddit_split_dataset/RE_LARGE_extra_all_keys_2len_all_valid.csv',
                        help='Path to valid datasets')
    parser.add_argument('--max_len', default=250, type=int,
                        help='Limit of tokens')
    parser.add_argument('--num_epochs', default=15, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=400, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--resize_size', default=224, type=int,
                        help='Resize the input image to the given size.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--dropout', default=0.4, type=float,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--use_eval', action='store_true',
                        help='Use evaluation')
    parser.add_argument('--save_best', action='store_true',
                        help='Save best model')
    parser.add_argument('--debug', action='store_true',
                            help='train few train data')
    opt = parser.parse_args()
    print(opt)
    
    # For descriptive error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    # Load the BERT tokenizer
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    MAX_LEN = opt.max_len
    
    df_train = pd.read_csv(opt.train_data_path)
    if opt.debug:
        df_train = df_train[:1000]
    df_valid= pd.read_csv(opt.valid_data_path)

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

        
    # define a callable image_transform with Compose
    # 224x224へのリサイズはspotFake+準拠
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(opt.resize_size, opt.resize_size)),
            torchvision.transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    
    # Run function `preprocessing_for_bert` on the dataset
    transformed_dataset_train = FakeNewsDataset(df_train, image_transform, tokenizer, MAX_LEN)

    transformed_dataset_val = FakeNewsDataset(df_valid, image_transform, tokenizer, MAX_LEN)

    train_dataloader = DataLoader(transformed_dataset_train, batch_size=opt.batch_size,
                            shuffle=True, num_workers=2,pin_memory=True)

    val_dataloader = DataLoader(transformed_dataset_val, batch_size=opt.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    parameter_dict_model={
        'fine_tune_text_module': True,
        'dropout_p': opt.dropout, 
        'fine_tune_vis_module': True
    }

    parameter_dict_opt={'l_r': opt.learning_rate}

    EPOCHS=opt.num_epochs

    set_seed(opt.seed)    # Set seed for reproducibility

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    # 外部知識版モデル
    final_model = LanguageAndVisionConcatWithExtra(parameter_dict_model)

    final_model = final_model.to(device) 

    # Create the optimizer
    optimizer = torch.optim.Adam(final_model.parameters(),
                      lr=parameter_dict_opt['l_r'])

    # Total number of training steps
    total_steps = len(train_dataloader) * EPOCHS

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)

    predictions = train(model=final_model, 
                        loss_fn=loss_fn, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        epochs=EPOCHS, 
                        evaluation=opt.use_eval, 
                        device=device, 
                        param_dict_model=parameter_dict_model, 
                        param_dict_opt=parameter_dict_opt,
                        save_best=opt.save_best,
                        file_path='./saved_models/best_model.pt')

    
def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, device='cpu', param_dict_model=None,
          param_dict_opt=None, save_best=False, file_path='./saved_models/best_model.pt'):
    """Train the model.
    """
    # Start training loop
    best_acc_val = 0
    print("Start training...\n")
    for epoch_i in range(epochs):
        gc.collect()
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            img_ip , text_ip, label = batch["image"], batch["BERT_ip"], batch['label']
            
            b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)

            
            imgs_ip = img_ip.to(device)
            
            b_labels = label.to(device)
            
 
            # Zero out any previously calculated gradients
            model.zero_grad()

            logits, attentions = model(text=[b_input_ids, b_attn_mask], image=imgs_ip, label=b_labels)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                
                
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        gc.collect()
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy, predictions = evaluate(model, loss_fn, val_dataloader, device)            
            
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            
            # Save the best model
            if save_best: 
                if val_accuracy > best_acc_val:
                    best_acc_val = val_accuracy
                    torch.save({
                                'epoch': epoch_i+1,
                                'model_params': param_dict_model,
                                'opt_params': param_dict_opt,
                                'model_state_dict': model.state_dict(),
                                'opt_state_dict': optimizer.state_dict(),
                                'sch_state_dict': scheduler.state_dict()
                               }, file_path)
                    
        print("\n")
        gc.collect()
    
    print("Training complete!")
    return predictions

if __name__ == '__main__':
    main()