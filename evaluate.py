import torch
import numpy as np
@torch.no_grad()
def evaluate(model, loss_fn, val_dataloader, device):
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    preds = []
    labels = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        img_ip , text_ip, label = batch["image"], batch["BERT_ip"], batch['label']
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in text_ip)
        imgs_ip = img_ip.to(device)
        b_labels = label.to(device)


        # Compute logits
        with torch.no_grad():
            logits, attentions = model(text=[b_input_ids, b_attn_mask], image=imgs_ip, label=b_labels)


        # Compute loss
        # print(type(logits), type(b_labels))
        # print(logits)
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        pred = torch.argmax(logits, dim=1).flatten()

        preds.append(pred.to('cpu').numpy())
        labels.append(b_labels.to('cpu').numpy())
        # Calculate the accuracy rate
        accuracy = (pred == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
        # print('-'*100)

    # Compute the average accuracy and loss over the validation set.
    predictions = np.concatenate(preds)
    labels = np.concatenate(labels)
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy, predictions

