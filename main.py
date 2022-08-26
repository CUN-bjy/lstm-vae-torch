##########################
# Autor: Junyeob Baek
# email: wnsdlqjtm@gmail.com
##########################

import secrets

import easydict
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from src.models import LSTMAE, LSTMVAE
from utils.MovingMNIST import MovingMNIST

writer = SummaryWriter()

## visualization
def imshow(past_data, title="MovingMNIST"):
    num_img = len(past_data)
    fig = fig = plt.figure(figsize=(4 * num_img, 4))

    for idx in range(1, num_img + 1):
        ax = fig.add_subplot(1, num_img + 1, idx)
        ax.imshow(past_data[idx - 1])
    plt.suptitle(title, fontsize=30)
    plt.savefig(f"{title}")
    plt.close()


def train(args, model, train_loader, test_loader):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ## interation setup
    epochs = tqdm(range(args.max_iter // len(train_loader) + 1))

    ## training
    count = 0
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        train_iterator = tqdm(
            enumerate(train_loader), total=len(train_loader), desc="training"
        )

        for i, batch_data in train_iterator:

            if count > args.max_iter:
                return model
            count += 1

            future_data, past_data = batch_data

            ## reshape
            batch_size = past_data.size(0)
            example_size = past_data.size(1)
            image_size = past_data.size(2), past_data.size(3)
            past_data = (
                past_data.view(batch_size, example_size, -1).float().to(args.device)
            )
            # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

            mloss, recon_x, info = model(past_data)

            # Backward and optimize
            optimizer.zero_grad()
            mloss.mean().backward()
            optimizer.step()

            train_iterator.set_postfix({"train_loss": float(mloss.mean())})
        writer.add_scalar("train_loss", float(mloss.mean()), epoch)

        model.eval()
        eval_loss = 0
        test_iterator = tqdm(
            enumerate(test_loader), total=len(test_loader), desc="testing"
        )

        with torch.no_grad():
            for i, batch_data in test_iterator:
                future_data, past_data = batch_data

                ## reshape
                batch_size = past_data.size(0)
                example_size = past_data.size(1)
                past_data = (
                    past_data.view(batch_size, example_size, -1).float().to(args.device)
                )
                # future_data = future_data.view(batch_size, example_size, -1).float().to(args.device)

                mloss, recon_x, info = model(past_data)

                eval_loss += mloss.mean().item()

                test_iterator.set_postfix({"eval_loss": float(mloss.mean())})

                if i == 0:
                    nhw_orig = past_data[0].view(example_size, image_size[0], -1)
                    nhw_recon = recon_x[0].view(example_size, image_size[0], -1)
                    imshow(nhw_orig.cpu(), f"orig{epoch}")
                    imshow(nhw_recon.cpu(), f"recon{epoch}")
                    # writer.add_images(f"original{i}", nchw_orig, epoch)
                    # writer.add_images(f"reconstructed{i}", nchw_recon, epoch)

        eval_loss = eval_loss / len(test_loader)
        writer.add_scalar("eval_loss", float(eval_loss), epoch)
        print("Evaluation Score : [{}]".format(eval_loss))

    return model


if __name__ == "__main__":

    # training dataset
    train_set = MovingMNIST(
        root=".data/mnist",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )

    # test dataset
    test_set = MovingMNIST(
        root=".data/mnist",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
    )

    args = easydict.EasyDict(
        {
            "batch_size": 512,
            "device": torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            "input_size": 4096,
            "hidden_size": 2048,
            "latent_size": 1024,
            "learning_rate": 0.001,
            "max_iter": 1000,
        }
    )

    batch_size = args.batch_size
    input_size = args.input_size
    hidden_size = args.hidden_size
    latent_size = args.latent_size

    # define LSTM-based VAE model
    model = LSTMVAE(input_size, hidden_size, latent_size, device=args.device)
    model.to(args.device)

    # reduce dataset size(customization for quick experiments)
    # tr_split_len, te_split_len = 9000, 1000
    # part_tr = torch.utils.data.random_split(
    #     train_set, [tr_split_len, len(train_set) - tr_split_len]
    # )[0]
    # part_te = torch.utils.data.random_split(
    #     test_set, [te_split_len, len(test_set) - te_split_len]
    # )[0]

    # convert to format of data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False
    )

    # training
    train(args, model, train_loader, test_loader)

    # save model
    id_ = secrets.token_hex(nbytes=4)
    torch.save(model.state_dict(), f"lstmvae{id_}.model")

    # load model
    model_to_load = LSTMVAE(input_size, hidden_size, latent_size, device=args.device)
    model_to_load.to(args.device)
    model_to_load.load_state_dict(torch.load(f"lstmvae{id_}.model"))
    model_to_load.eval()

    # show results
    ## past_data, future_data -> shape: (10,10)
    future_data, past_data = train_set[0]

    ## reshape
    example_size = past_data.size(0)
    image_size = past_data.size(1), past_data.size(2)
    past_data = past_data.view(example_size, -1).float().to(args.device)
    _, recon_data, info = model_to_load(past_data.unsqueeze(0))

    nhw_orig = past_data.view(example_size, image_size[0], -1).cpu()
    nhw_recon = (
        recon_data.squeeze(0)
        .view(example_size, image_size[0], -1)
        .detach()
        .cpu()
        .numpy()
    )

    imshow(nhw_orig, title=f"final_input{id_}")
    imshow(nhw_recon, title=f"final_output{id_}")
    plt.show()
