import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def train(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    epoch,
    train_loss,
    train_acc,
    mse=None,
):
    model.train()

    curr_loss = 0
    t_pred = 0
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images).squeeze()
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        curr_loss += loss.sum().item()
        _, preds = torch.max(output, 1)
        t_pred += torch.sum(preds == targets.data).item()

        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(images),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

            train_loss.append(loss.sum().item() / len(images))
            train_acc.append(preds.sum().item() / len(images))
    epoch_loss = curr_loss / len(train_loader.dataset)
    epoch_acc = t_pred / len(train_loader.dataset)

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    print(
        "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            epoch_loss,
            t_pred,
            len(train_loader.dataset),
            100.0 * t_pred / len(train_loader.dataset),
        )
    )

    return train_loss, train_acc, epoch_loss


def valid(
    model, device, test_loader, criterion, epoch, valid_loss, valid_acc, mse=None
):
    model.eval()
    test_loss = 0
    correct = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            output = model(images).squeeze()

            loss = criterion(output, targets)
            test_loss += loss.sum().item()

            probs = F.softmax(output, dim=1)
            # ImageFolder: fake=0, real=1 → score for fake class
            all_scores.extend(probs[:, 0].cpu().numpy())
            all_labels.extend((targets == 0).cpu().numpy().astype(int))

            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == targets.data)

            if batch_idx % 10 == 0:
                print(
                    "Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(images),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        loss.item(),
                    )
                )

                valid_loss.append(loss.sum().item() / len(images))
                valid_acc.append(preds.sum().item() / len(images))

    epoch_loss = test_loss / len(test_loader.dataset)
    epoch_acc = correct / len(test_loader.dataset)

    valid_loss.append(epoch_loss)
    valid_acc.append(epoch_acc.item())

    val_auc = roc_auc_score(all_labels, all_scores)

    print(
        "Valid Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), AUC: {:.4f}\n".format(
            epoch_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
            val_auc,
        )
    )

    return valid_loss, valid_acc, val_auc
