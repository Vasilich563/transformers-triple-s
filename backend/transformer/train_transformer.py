import csv
from datetime import datetime
from threading import Thread
from random import shuffle
import torch
from torch.utils.data import DataLoader
from backend.define_torch_device import define_device
from bidirectional_transformer import BidirectionalTransformer
from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling


def train_step(model, optimizer, schedule, loss_function, dataloader, batches_amount, step, warmup_step, total_steps):
    running_loss = 0
    for data in dataloader:
        optimizer.zero_grad(set_to_none=True)
        x_batch = data["input_ids"]
        mask_batch = data["hugging_face_mask"]
        y_batch = data["labels"]
        sample_logits = model.train_forward(x_batch, hugging_face_mask=mask_batch)  # TODO unpack x

        batch, seq_len, vocab_size = sample_logits.shape
        sample_logits = sample_logits.view(batch * seq_len, vocab_size)
        y_batch = y_batch.view(batch * seq_len)

        loss = loss_function(input=sample_logits, target=y_batch)
        loss.backward()
        optimizer.step()
        step += 1

        running_loss += loss.detach().cpu().item()
        end_train = (step == total_steps)
        if end_train:
            running_loss = running_loss / batches_amount
            return running_loss, step, True  # loss, step, end_training
        if step % warmup_step == 0:
            schedule.step()

    running_loss = running_loss / batches_amount
    return running_loss, step, False  # loss, step, end_training


def validation_step(model, loss_function, dataloader, batches_amount):
    running_loss = 0
    for data in dataloader:
        x_batch = data["input_ids"]
        mask_batch = data["hugging_face_mask"]
        y_batch = data["labels"]
        sample_logits = model.train_forward(x_batch, hugging_face_mask=mask_batch)  # TODO unpack x

        batch, seq_len, vocab_size = sample_logits.shape
        sample_logits = sample_logits.view(batch * seq_len, vocab_size)
        y_batch = y_batch.view(batch * seq_len)

        loss = loss_function(input=sample_logits, target=y_batch)

        running_loss += loss.detach().cpu().item()

    running_loss = running_loss / batches_amount
    return running_loss


def get_batches_amount(dataset_size, batch_size):
    if int(dataset_size / batch_size) * batch_size == dataset_size:
        return int(dataset_size / batch_size)
    elif int(dataset_size / batch_size) * batch_size < dataset_size:
        return int(dataset_size / batch_size) + 1


def save_model_daemon(model, path_to_save_models, epoch):
    save_model_daemon = Thread(
        target=torch.save,
        args=(model.state_dict(), f"{path_to_save_models}/after_epoch{epoch}.pt"),
        daemon=True
    )
    save_model_daemon.start()


def train(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, schedule, loss_function,
    train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader,
    warmup_step, total_steps, save_period, path_to_save_models
):
    train_start = datetime.now()
    print("Start training")
    train_losses = []
    val_losses = []
    train_batches_amount = get_batches_amount(len(train_dataloader.dataset), train_dataloader.batch_size)
    val_batches_amount = get_batches_amount(len(val_dataloader.dataset), val_dataloader.batch_size)
    model.train()
    step = 0
    while True:
        epoch_start = datetime.now()
        train_running_loss, step, end_training = train_step(
            model, optimizer, schedule, loss_function, train_dataloader, train_batches_amount, step, warmup_step, total_steps
        )

        if step % save_period == 0:
            print(f"Epoch {step + 1}/{total_steps}")

            with torch.no_grad():
                model.eval()
                val_running_loss = validation_step(model, loss_function, val_dataloader, val_batches_amount)
                model.train()

            train_losses.append(train_running_loss)
            val_losses.append(val_running_loss)

            print(f"\tStep is ended in {datetime.now() - epoch_start}\n\tTrain loss:\t{train_running_loss}\n\tValidation loss: {val_running_loss}")
            save_model_daemon(model, path_to_save_models, step)

        if end_training:
            print(f"Time spent on train: {datetime.now() - train_start}")
            return train_losses, val_losses


def save_losses(train_losses, validation_losses, filename, save_period):
    with open(filename, 'w') as fout:
        writer = csv.DictWriter(fout, ["Эпоха", "Ошибка обучения", "Ошибка валидации"])
        writer.writeheader()
        for i in range(len(train_losses)):
            if i % save_period == 0:
                writer.writerow(
                    {"Эпоха": i + 1, "Ошибка обучения": train_losses[i], "Ошибка валидации": validation_losses[i]}
                )
            else:
                writer.writerow(
                    {"Эпоха": i + 1, "Ошибка обучения": train_losses[i], "Ошибка валидации": "chep"}
                )


def init_dataloaders(text, tokenizer, data_collator, max_length, stride, batch_size, train_part):
    tokens_ = tokenizer(
        text, truncation=True, padding="max_length", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_tensors='pt'
    )
    dataset = [
        {"input_ids": tokens_["input_ids"][i], "mask": tokens_["attention_mask"][i]} for i in range(tokens.input_ids.shape[0])
    ]
    shuffle(dataset)
    train_dataset = dataset[:int(len(dataset) * train_part)]
    val_dataset = dataset[int(len(dataset) * train_part):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    return train_loader, val_loader


def make_dataset():
    pass


if __name__ == "__main__":
    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-large")
    mlm_probability = 0.15
    mlm = DataCollatorForLanguageModeling(tokenizer, mlm_probability=mlm_probability, return_tensors='pt')
    device = define_device()
    dtype = torch.float32

    vocab_size = len(tokenizer.get_vocab())
    max_len = 256
    num_layers = 12
    d_model = 768
    num_attention_heads = 12
    d_ffn_hidden = 3072
    dropout_p = 0.1
    padding_index = tokenizer.pad_token_type_id

    triple_s_roberta = BidirectionalTransformer(
        vocab_size, max_len, num_layers, d_model, num_attention_heads, d_ffn_hidden, dropout_p, device, dtype, padding_index
    )

    batch_size = 4096
    total_steps = 1048576
    warmup_step = 49152
    weight_decay = 0.01
    eps = 1e-6
    beta1 = 0.9
    beta2 = 0.98
    peak_lr = 6e-4
    optimizer = torch.optim.Adam(
        triple_s_roberta.parameters(), lr=peak_lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay
    )
    lr_decay_schedule = torch.optim.lr_scheduler.LinearLR(optimizer)

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)

    

    text = """Meshuggah is a Swedish extreme metal band formed in Umeå in 1987. Since 2004, the band's lineup consists of founding members Jens Kidman (lead vocals) and Fredrik Thordendal (lead guitar), alongside rhythm guitarist Mårten Hagström, drummer Tomas Haake and bassist Dick Lövgren. Since its formation, the band has released nine studio albums, six EPs and eight music videos. Their latest studio album, Immutable, was released in April 2022 via Atomic Fire Records.
        Meshuggah has become known for their innovative musical style and their complex, polymetered song structures and polyrhythms. They rose to fame as a significant act in extreme underground music, became an influence for modern metal bands, and gained a cult following. The band was labelled as one of the ten most important hard rock and heavy metal bands by Rolling Stone and as the most important band in metal by Alternative Press. In the late 2000s, the band was an inspiration for the djent subgenre.
        In 2006 and 2009, Meshuggah was nominated for two Swedish Grammis Awards for their albums Catch Thirtythree and obZen, respectively. In 2018, the band was nominated for a Grammy Award for their song "Clockworks" under the "Best Metal Performance" category.[2] The band has performed in various international festivals, including Ozzfest and Download, and embarked on the obZen world tour from 2008 to 2010, and also the "Ophidian Trek".
        """
    from pprint import pprint
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=8, stride=4, return_tensors='pt', return_overflowing_tokens=True)
    print(tokens.input_ids.shape)

    dataset = [
        {"input_ids": tokens["input_ids"][i], "hugging_face_mask": tokens["attention_mask"][i]} for i in range(tokens.input_ids.shape[0])
    ]
    #pprint(dataset)
    #print(dataset)


    model = BidirectionalTransformer(len(tokenizer.get_vocab()), 8, 1, 32, 2, 14, 0, torch.device("cuda"), torch.float32, tokenizer.pad_token_type_id)



    from bidirectional_transformer import make_mask
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=mlm)
    for i, x in enumerate(loader):
        print(i, x)
        break

    for i, x in enumerate(loader):
        print(i, x)
        break

    # for x in loader:
    #     #print(make_mask(x["hugging_face_mask"], torch.float32).shape)
    #     model(
    #         x["input_ids"].to(torch.device("cuda")),
    #         hugging_face_mask=torch.tensor([
    #             [1, 1, 1, 1, 1, 0, 0, 0],
    #             [1, 1, 1, 1, 1, 1, 1, 0]
    #         ], requires_grad=False, device=torch.device("cuda"))
    #     )
        #model(x["input_ids"].to(torch.device("cuda")), hugging_face_mask=x["hugging_face_mask"])






    #dataset = []
    # for text in texts:
    #     text_tokens =
    #     dataset.append(
    #
    #     )
    # print(dataset[-1]["attention_mask"])


    # a, b = mlm.torch_mask_tokens(tokens)
    # print(a)
    # print("#" * 100)
    # print(b)




