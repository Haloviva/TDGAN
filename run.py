#encoding:utf-8
import torch
import pandas
import json
import numpy as np
import os
import torch_geometric
from tqdm import tqdm
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.data import DataLoader as graph_DataLoader
from transformers import BertModel, XLNetModel, RobertaModel
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import *
from model import *
from utils import *


device = torch.device('cuda')

def read_dataset(datapath, datatype):

    with open(datapath+'/processed_datasets/{}_sentdata.json'.format(datatype), 'r') as f:
        train_sentdata =json.load(f)
    f.close()
    with open(datapath+'/processed_datasets/{}_table_ids.json'.format(datatype), 'r') as f:
        train_table_ids = json.load(f)
    f.close()
    with open(datapath+'/processed_datasets/{}_table_data.json'.format(datatype), 'r') as f:
        train_table_data = json.load(f)
    f.close()
    return train_sentdata, train_table_ids, train_table_data

def processor():
    args = parse_opt()

    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    # --------------------- Data loading --------------------------------------------
    if args.do_train:
        print(20 * "=", " Preparing for training", 20 * "=")
        print("\t* Loading train data...")

        train_sentdata, train_table_ids, train_table_data = read_dataset(args.data, 'train')

        #tabfact_dataset_train = TabFactDataset_with_no_edge_weight(train_sentdata, train_table_data)
        tabfact_dataset_train = EmbeddingDataset(args.data, 'train', train_table_data["table"])
        train_sampler = torch.utils.data.distributed.DistributedSampler(tabfact_dataset_train) 
        
        TabFactDataLoader_train = torch_DataLoader(tabfact_dataset_train, batch_size = args.batch, shuffle = True, collate_fn = lambda x:x)

        
        #---------------------- Model definition ----------------------------------------
        print("\t* Building model...")

        model = TDGAN(args.dim, args.dim)
        model.to(device)

        # --------------------- Preparation for training --------------------------------
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr = args.lr_default)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 50, num_training_steps = TabFactDataLoader_train.__len__()*args.epochs)


        best_score = 0.0
        start_epoch = 1

        epochs_count = []
        train_losses = []
        valid_losses = []


        # --------------------- Training epochs -----------------------------------------
        print("\n",
                20 * "=",
                "Training model on device: {}".format(device),
                20 * "=")

        patience_counter = 0
        for epoch in range(start_epoch, args.epochs+1):
            epochs_count.append(epoch)

            print("* Training epoch {}:".format(epoch))
            epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                           TabFactDataLoader_train,
                                                           optimizer,
                                                           criterion)

            train_losses.append(epoch_loss)
            print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
                  .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

            torch.save(model.state_dict(), 'models/model_ep{}.pt'.format(epoch))
    
    if args.do_example:
        print(20 * "=", " Preparing for example", 20 * "=")
        print("\t* Loading example data...")

        example_sentdata, example_table_ids, example_table_data = read_dataset(args.data, 'example') 
        '''
        tabfact_dataset_example = TabFactDataset_with_no_edge_weight(example_sentdata, example_table_data)
        TabFactDataLoader_example = torch_DataLoader(tabfact_dataset_example, batch_size = args.batch, shuffle = True, collate_fn = lambda x:x)
        '''
        EmbeddingDataLoader_example = EmbeddingDataset(args.data, 'example')
        TabFactDataLoader_example = torch_DataLoader(EmbeddingDataLoader_example, batch_size = args.batch, shuffle = True, collate_fn = lambda x:x)
            

        #---------------------- Model definition ----------------------------------------
        print("\t* Building model...")

        model = TDGAN(args.dim, args.dim)
        model.to(device)

        # --------------------- Preparation for training --------------------------------
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr = args.lr_default)
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 50, num_training_steps = TabFactDataLoader_example.__len__()*args.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 50, num_training_steps = TabFactDataLoader_example.__len__()*args.epochs)
        '''
        if args.fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level = args.fp16_opt_level)
        '''
        best_score = 0.0
        start_epoch = 1

        epochs_count = []
        train_losses = []
        valid_losses = []


        # --------------------- Training epochs -----------------------------------------
        print("\n",
                20 * "=",
                "Training model on device: {}".format(device),
                20 * "=")

        patience_counter = 0
        for epoch in range(start_epoch, args.epochs+1):
            epochs_count.append(epoch)

            print("* Training epoch {}:".format(epoch))

            epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                           TabFactDataLoader_example,
                                                           optimizer,
                                                           criterion)
            
            train_losses.append(epoch_loss)
            print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
                  .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

            torch.save(model.state_dict(), 'model_save/model_ep{}.pt'.format(epoch)) 

            
    if args.do_dev:
        print("\t* Loading dev data...")

        dev_sentdata, dev_table_ids, dev_table_data = read_dataset(args.data, 'dev')

        table_graph_dataset_dev = TableGraphDataset(args.data, 'dev', dev_table_data, dev_table_ids)
        table_graph_dataset_dev.read_num_type()
        table_graph_dev, remove_file_dev = table_graph_dataset_dev.table_graph() 
        tabfact_dataset_dev = TabFactDataset(dev_sentdata, table_graph_dev, remove_file_dev)
        TabFactDataLoader_dev = torch_DataLoader(tabfact_dataset_dev, batch_size = args.batch, shuffle = Ture, collate_fn = collate_fn)

        print("* Validation for epoch {}".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = validation(model,
                                                            TabFactDataLoader_dev,
                                                            criterion)

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n".format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        scheduler.step(epoch_accuracy)
        
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            beat_score = epoch_accuracy
            patience_counter = 0

            torch.save({
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                        os.path.join(target_dir, "best.pth.tar"))

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
                "epochs_count": epochs_count,
                "train_losses": train_losses,
                "valid_losses": valid_losses
            },
            os.path.join(target_dir, "TDGAN_{}.pth.tar".format(epoch))
        )

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            f = open("final_valid_result.txt", "w")
            f.write("epoch:"+str(epoch)+'\n'+"epoch_loss:"+str(epoch_loss)+'\n'+"epoch_accuracy:"+ str(epoch_accuracy*100))
            f.close()
 
    if args.do_test:
        print(20 * "=", " Preparing for testing", 20 * "=")
        print("\t* Loading test data...")

        test_sentdata, test_table_ids, test_table_data = read_dataset(args.data, 'test')

        table_graph_dataset_test = TableGraphDataset(args.data, 'test', test_table_data, test_table_ids)
        table_graph_dataset_test.read_num_type()
        table_graph_test, remove_file_test = table_graph_dataset_test.table_graph()
        tabfact_dataset_test = TabFactDataset(test_sentdata, table_graph_test, remove_file_test)
        TabFactDataLoader_test = torch_DataLoader(tabfact_dataset_test, batch_size = args.batch, shuffle = True, collate_fn = collate_fn)

        print("\t* Building model...")
        checkpoint = troch.load(args.target_dir)
        model = TDGAN()
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['train_loss']

        print(20 * "=",
          " Testing TDGAN model on device: {} ".format(device),
          20 * "=")
        batch_time, total_time, accuracy = test(model, 
                                                TabFactDataLoader_test)
        print("-> Average batch processing time: {:.4f}s, total_time: {:.4f}s, accuracy: {:.4f}%"
              .format(batch_time, total_time, (accuracy*100)))
    

if __name__ == "__main__":
   processor() 