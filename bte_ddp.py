import os, sys, time, warnings, pickle, random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
from torchvision import models, datasets, transforms

from sklearn.model_selection import train_test_split
from components import topk_accuracy, ToRGB, BetaNet2, BetaNet3, EffNetV1, EffNetV2, MobileNetV2, ResNet50

warnings.filterwarnings("ignore")

def model_builder(model_type, model_parameters):
    if model_type == 'violin':
        model = ViolinModel(**model_parameters)
    elif model_type == 'beta2':
        model = BetaNet2(**model_parameters)
    elif model_type == 'beta3':
        model = BetaNet3(**model_parameters)
    elif model_type == 'effnetv1':
        model = EffNetV1(**model_parameters)
    elif model_type == 'effnetv2':
        model = EffNetV2(**model_parameters)
    elif model_type == 'mobilenetv2':
        model = MobileNetV2(**model_parameters)
    elif model_type == 'resnet50':
        model = ResNet50(**model_parameters)
    return model

def train_eval_ddp(rank, world_size, model_type, model_parameters, time_budget_mins, nepochs, batch_size, accumulate, train_dataset, valid_dataset, evaluate, saving):

    # Config cuda rank and model
    torch.cuda.empty_cache()
    torch.cuda.set_device(rank)
    model = model_builder(model_type, model_parameters)
    model.to(rank)
    model.cuda()
    ddp_model = DDP(model, device_ids=[rank])

    # Requiring deterministic behaviour
    cudnn.benchmark = False
    cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    SEED = 308184653
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Self assessment
    if rank == 0:
        params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        print(f'Generated {model_type} model with {params:,.0f} params.')

    # Use gradient compression to reduce communication
    ddp_model.register_comm_hook(None, default.fp16_compress_hook)

    loss_function = nn.CrossEntropyLoss(reduction='sum').to(rank)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # Init train and validation samplers and loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=6)

    if evaluate:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=False, num_workers=6)

    # Training
    results = {}
    time_remaining = time_budget_mins
    best_top1 = 0 # Trigger for model saving

    for epoch in range(nepochs):

        cumulative_train_loss = 0.0
        train_examples_seen = 0.0
        n_train_batches = len(train_loader)
        ddp_model.train(True)
        optimizer.zero_grad()
        start = time.perf_counter()

        for i, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(rank), y_train.to(rank)

            if (i + 1) % accumulate == 0 or (i + 1) == n_train_batches: # Final loop in accumulation cycle, or last batch in dataset
                z_train = ddp_model(X_train)
                loss = loss_function(z_train, y_train)
                cumulative_train_loss += loss.item()
                train_examples_seen += len(y_train)
                loss.backward() # Sync gradients between devices
                optimizer.step() # Weight update
                optimizer.zero_grad() # Zero grad

            else: # Otherwise only accumulate gradients locally to save time.
                with ddp_model.no_sync():
                    z_train = ddp_model(X_train)
                    loss = loss_function(z_train, y_train)
                    cumulative_train_loss += loss.item()
                    train_examples_seen += len(y_train)
                    loss.backward()

        # Average training loss per batch and training time
        tloss = cumulative_train_loss / train_examples_seen
        epoch_duration = time.perf_counter() - start

        # Evaluation
        if evaluate:

            cumulative_valid_loss = 0.0
            valid_examples_seen = 0.0
            top1acc = 0.0
            top5acc = 0.0
            
            ddp_model.train(False)
            with torch.no_grad():
                for X_valid, y_valid in valid_loader:
                    X_valid, y_valid = X_valid.to(rank), y_valid.to(rank)
                    z_valid = model(X_valid)

                    loss_valid = loss_function(z_valid, y_valid)
                    valid_top1acc = topk_accuracy(z_valid, y_valid, topk=1, normalize=False)
                    valid_top5acc = topk_accuracy(z_valid, y_valid, topk=5, normalize=False)

                    cumulative_valid_loss += loss_valid.item()
                    top1acc += valid_top1acc.item()
                    top5acc += valid_top5acc.item()
                    valid_examples_seen += len(y_valid)

            vloss = cumulative_valid_loss / valid_examples_seen
            top1 = (top1acc / valid_examples_seen) * 100
            top5 = (top5acc / valid_examples_seen) * 100
        
        else:

            vloss = 0
            top1 = 0
            top5 = 0

        # Gather performance from all devices and average for reporting
        transmit_data = np.array([tloss, vloss, top1, top5, epoch_duration])
        outputs = [None for _ in range(world_size)]
        dist.all_gather_object(outputs, transmit_data)
        result = np.stack(outputs).mean(axis=0)
        tloss_, vloss_, top1_, top5_, epoch_duration_ = result

        results[epoch] = {
            "tloss": tloss_,
            "vloss": vloss_,
            "top1": top1_,
            "top5": top5_,
            "time": epoch_duration_
        }

        # Learning rate scheduler reducing by factor of 10 when training loss stops reducing. Likely to overfit first.
        # Must apply same operation for all devices to ensure optimizers remain in sync.
        scheduler.step(tloss_)
        
        # If main rank, save results and report.
        if rank == 0:
            with open(f'experiment_results/{model_type}_result.pkl', 'wb') as handle:
                pickle.dump(results, handle)

            print(f'EPOCH {epoch}, TLOSS {tloss_:.3f}, VLOSS {vloss_:.3f}, TOP1 {top1_:.2f}, TOP5 {top5_:.2f}, TIME {epoch_duration_:.3f}')

            # If the best top1 accuracy has been exceeded, save the model.
            if top1_ > best_top1 and saving == 'best':
                torch.save(ddp_model.module.state_dict(), f'experiment_results/{model_type}.pt')
                best_top1 = top1_

        ## Apply time budget stopping
        # Subtract average epoch_duration from time_remaining
        time_remaining -= epoch_duration_ / 60

        # Does this rank think time has run out?
        if time_remaining <= 0:
            transmit_data = True
        else:
            transmit_data = False

        # Talk to each other
        outputs = [None for _ in range(world_size)]
        dist.all_gather_object(outputs, transmit_data)

        # If anyone thinks time has run out, we all stop.
        if any(outputs):
            # Terminate training
            break

    # If saving final model, save it.
    if rank == 0 and saving == 'final':
        torch.save(ddp_model.module.state_dict(), f'experiment_results/{model_type}.pt')

    return results


def init_process(rank, world_size, service_function, model_type, model_parameters, time_budget_mins, nepochs, batch_size, accumulate, train_dataset, valid_dataset, evaluate, saving, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    service_function(rank, world_size, model_type, model_parameters, time_budget_mins, nepochs, batch_size, accumulate, train_dataset, valid_dataset, evaluate, saving)


if __name__ == "__main__":
    '''
    Script should be passed "parameters" dictionary generated by Ax and "nepochs" defined in notebook.
    Use command "%run -i bte_ddp.py" to inherit from notebook environment:
     - model_type           # type of model to build
     - model_parameters     # parameters for building the model
     - world_size           # number of GPUs to train on
     - time_budget_mins     # training time budget
     - nepochs              # max epochs of training
     - batch_size           # the number of examples included in each forward pass. Typically set to max possible on a GPU
     - accumulate           # the number of forward passes to accumulate in each update step
     - train_dataset        # training dataset
     - valid_dataset        # validation dataset
     - evaluate             # boolean indicating whether or not to evaluate the model each epoch on a validation set.
     - saving               # save model after each epoch the model improves on the validation set if 'best' or at the end of N epochs if 'final'.
    '''

    processes = []
    mp.set_start_method("spawn", force=True)

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(
            rank, world_size, train_eval_ddp, model_type, model_parameters, time_budget_mins, nepochs, batch_size, accumulate, train_dataset, valid_dataset, evaluate, saving
            )
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    


