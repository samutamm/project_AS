
import time
from torch import nn
import torch

import numpy as np

import torchvision.transforms as transforms

from .tools import AverageMeter
from .Snip import SNIP

def get_dataloaders(dataset, path):
    pruning_batch_size = 128
    batch_size = 64

    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = dataset(path, train=True, download=True, transform=data_transforms)
    test_dataset = dataset(path, train=False, download=True, transform=data_transforms)
    pruning_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=pruning_batch_size, shuffle=True, num_workers=2)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size, shuffle=True, num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=batch_size, shuffle=True, num_workers=2)
    return pruning_data_loader, train_data_loader, test_data_loader

class MeanEvaluator:
    """
    Evaluates the given model k times and computes the mean score.
    """

    def __init__(self, model_class, dataset, eval_n, epochs, pruning_ratio, path = '../data/'):
        """
        :param model type torch.nn.Module :
        :param dataset type torchvision.datasets :
        :param eval_n number of evaluations :
        :param epochs number of epochs for each evaluation :
        :param pruning_ratio proportion of weight to be used,
            e.g pruning_ratio = 0.1 means that using only one tenth of weights :
        """
        self.model_class = model_class
        self.eval_n = eval_n
        self.pruning_ratio = pruning_ratio
        self.epochs = epochs
        self.pruning_data_loader, self.train_data_loader, self.test_data_loader = get_dataloaders(dataset, path)

    def evaluate_baseline(self):
        results = []
        for i in range(self.eval_n):
            model = self.model_class() # create new instance to reset the training
            _, _, accuracys = self.train_model(model, None)
            score = np.max(accuracys)
            print("iteration : {}, score : {}".format(i, score))
            results.append(score)
        return np.mean(results)

    def create_pruning_model(self):
        prune_model = self.model_class()
        snip = SNIP(prune_model)
        total_param_number = snip.get_total_param_number()
        K = int(total_param_number * self.pruning_ratio)
        snip.compute_mask(self.pruning_data_loader, K=K)
        return prune_model, snip

    def evaluate_pruned_model(self):
        results = []
        for i in range(self.eval_n):
            prune_model, snip = self.create_pruning_model() # create new instance to reset the training
            _, _, accuracys = self.train_model(prune_model, snip)
            score = np.max(accuracys)
            print("iteration : {}, score : {}".format(i, score))
            results.append(score)
        return np.mean(results)

    def train_model(self, model, snip=None, epochs=20):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        model = model.cuda()
        criterion = criterion.cuda()

        if snip:
            hooks = snip.register_masks()
            assert snip.K == snip.get_nonzero_param_number()

        train_losses = []
        test_losses = []
        accuracys = []
        # On itère sur les epochs
        for i in range(self.epochs):
            #print("=================\n=== EPOCH " + str(i + 1) + " =====\n=================\n")
            # Phase de train
            _, loss = epoch(self.train_data_loader,
                            model,
                            criterion,
                            snip_pruning=snip,
                            optimizer=optimizer,
                            PRINT_INTERVAL = -1)
            # Phase d'evaluation
            with torch.no_grad():
                acc_test, loss_test = epoch(self.test_data_loader,
                                            model,
                                            criterion,
                                            PRINT_INTERVAL = -1)

            train_losses.append(loss.avg)
            test_losses.append(loss_test.avg)
            accuracys.append(acc_test.avg)

        if snip:
            for hook in hooks:
                hook.remove()
            nonzero_params = snip.get_nonzero_param_number()
            print(nonzero_params)
            assert snip.K == nonzero_params

        return train_losses, test_losses, accuracys

def epoch(data, model, criterion, snip_pruning=None,  optimizer=None, PRINT_INTERVAL=100):
    """
    Fait une passe (appelée epoch en anglais) sur les données `data` avec le
    modèle `model`. Evalue `criterion` comme loss.
    Si `optimizer` est fourni, effectue une epoch d'apprentissage en utilisant
    l'optimiseur donné, sinon, effectue une epoch d'évaluation (pas de backward)
    du modèle.
    """

    # indique si le modele est en mode eval ou train (certaines couches se
    # comportent différemment en train et en eval)
    model.eval() if optimizer is None else model.train()

    # objets pour stocker les moyennes des metriques
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    avg_batch_time = AverageMeter()

    # on itere sur les batchs du dataset
    tic = time.time()
    for i, (input, target) in enumerate(data):

        input = input.cuda()
        target = target.cuda()

       # with torch.no_grad():

        # forward
        output = model(input)
        loss = criterion(output, target)

        # backward si on est en "train"
        if optimizer:
            #snip_pruning.prune_parameters()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calcul des metriques
        prediction = output.argmax(1)
        prec = (prediction == target).sum().cpu().item() / target.shape[0]
        batch_time = time.time() - tic
        tic = time.time()

        # mise a jour des moyennes
        avg_loss.update(loss.item())
        avg_acc.update(prec)
        avg_batch_time.update(batch_time)
        # affichage des infos
        if PRINT_INTERVAL != -1:
            if i % PRINT_INTERVAL == 0:
                print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:5.1f} ({top1.avg:5.1f})\n'.format(
                       "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
                       top1=avg_acc))

            # Affichage des infos sur l'epoch
            print('\n===============> Total time {batch_time:d}s\t'
              'Avg loss {loss.avg:.4f}\t'
              'Avg Prec {top1.avg:5.2f} %\n'.format(
               batch_time=int(avg_batch_time.sum), loss=avg_loss,
               top1=avg_acc))

    return avg_acc, avg_loss
