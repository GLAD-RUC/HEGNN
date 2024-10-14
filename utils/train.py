import os
import time
import json
import csv

import torch

def get_edges_in_mini_batch(batch_size, num_nodes_all, edge_index):
    correction_for_batch = num_nodes_all * torch.arange(batch_size, device=edge_index.device)  # [batch_size]
    correction_for_batch = correction_for_batch.repeat_interleave(edge_index.size(1) // batch_size, dim=0).unsqueeze(0)  # [1, edge_cnt]
    correction_for_batch = correction_for_batch.repeat_interleave(2, dim=0)
    edge_index_in_mini_batch = edge_index + correction_for_batch
    return edge_index_in_mini_batch


def kernel(x, y, sigma):
    dist = torch.cdist(x, y, p=2)
    k = torch.exp(- dist / (2 * sigma * sigma))
    return k


def train_single_epoch(model, loader, optimizer, loss, sigma, weight, epoch_index, backprop, tag, sample, device='cpu'):
    if backprop:
        model.train()
    else:
        model.eval()

    result = {'loss': 0., 'counter': 0.}
    for batch_index, data in enumerate(loader):
        # All to device
        data = data.to(device)
        data = data.detach()  # All detach

        # Parse data
        batch_size = data['ptr'].size(0) - 1
        edge_index, edge_attr = data['edge_index'], data['edge_attr']
        loc_0, vel_0, loc_t = data['loc_0'], data['vel_0'], data['loc_t']
        node_feat, node_attr = data['node_feat'], data['node_attr']

        row, col = edge_index
        edge_length_0 = torch.sqrt(torch.sum((loc_0[row] - loc_0[col])**2, dim=1)).unsqueeze(1)
        edge_attr = torch.cat([edge_attr, edge_length_0], dim=1)
        
        # detach from compute graph
        loc_0, vel_0, node_attr, node_feat = loc_0.detach(), vel_0.detach(), node_attr.detach(), node_feat.detach()
        edge_attr, edge_index = edge_attr.detach(), edge_index.detach()

        optimizer.zero_grad()

        # start_time = time.time()
        if model.__class__.__name__ == 'TFNModel':
            loc_predict = model(loc=loc_0, h=node_feat, vel=vel_0, edge_index=edge_index, data_batch=data['batch'])
        elif model.__class__.__name__ == 'VNEGNN':
            loc_predict, virtual_node_loc = model(node_loc=loc_0, node_attr=None, node_feat=node_feat, edge_index=edge_index, 
                                                  virtual_node_loc=data['virtual_fibonacci'].detach(), data_batch=data['batch'], edge_attr=edge_attr)
        elif model.__class__.__name__ == 'EGNN':
            out = model(x=loc_0, h=node_feat, edge_index=edge_index, edge_fea=edge_attr, v=vel_0)
            loc_predict = out[0]
        elif model.__class__.__name__ == 'HEGNN':
            loc_predict = model(node_feat, loc_0, vel_0, edge_index, edge_attr)
        elif model.__class__.__name__ == 'GNN':
            nodes = torch.cat([loc_0, vel_0], dim=1)
            loc_predict = model(h=nodes, edge_index=edge_index, edge_fea=edge_attr)
        elif model.__class__.__name__ == 'Linear_dynamics':
            loc_predict = model(x=loc_0, v=vel_0)
        elif model.__class__.__name__ == 'RF_vel':
            vel_norm = torch.sqrt(torch.sum(vel_0 ** 2, dim=1).unsqueeze(1)).detach()
            loc_predict = model(vel_norm=vel_norm, x=loc_0, edges=edge_index, vel=vel_0, edge_attr=edge_attr)
        elif model.__class__.__name__ == 'OurDynamics':  # TFN
            loc_predict = model(loc_0, vel_0, node_attr, edge_index)
        elif model.__class__.__name__ == 'GVPNet':
            h_V = (node_feat, torch.stack([loc_0, vel_0], dim=1))  # node_s, node_v
            row, col = edge_index
            h_E = (edge_attr, (loc_0[row] - loc_0[col]).unsqueeze(1))  # edge_s, edge_v
            out = model(h_V=h_V, edge_index=edge_index, h_E=h_E, batch=data['batch'])
            loc_predict = out[1][:, 0, :]  # get coord
        elif model.__class__.__name__ == 'SchNet':
            loc_predict = model(z=node_feat, pos=loc_0, batch=data['batch'], edge_index=edge_index)
        elif model.__class__.__name__ in ['ClofNet', 'ClofNet_vel', 'clof_vel_gbf']:
            nodes = torch.sqrt(torch.sum(vel_0 ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edge_index
            loc_dist = torch.sum((loc_0[rows] - loc_0[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            n_node = torch.tensor([loc_0.size(0) // batch_size])
            loc_predict = model(nodes, loc_0.detach(), edge_index, vel_0, edge_attr, n_nodes=n_node)
        elif model.__class__.__name__ == 'MACEModel':
            loc_predict = model(loc=loc_0, h=node_feat, vel=vel_0, edge_index=edge_index, data_batch=data['batch'])
        elif model.__class__.__name__ == 'SEGNN':
            x = torch.cat([node_feat, loc_0, vel_0], dim=1)
            loc_predict = model(x=x, pos=loc_0, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr, batch=data['batch'])
        else:
            print(model.__class__.__name__)
            raise Exception('Wrong model')
        
        loss_loc = loss(loc_predict, loc_t)

        # record the loss
        result['loss'] += loss_loc.item() * batch_size
        result['counter'] += batch_size
        
        if backprop:
            loss_loc.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""

    print(f'{prefix + tag} epoch: {epoch_index}, avg loss: {result["loss"] / result["counter"] :.5f}')

    return result['loss'] / result['counter']

def train(model, loader_train, loader_valid, loader_test, optimizer, loss, sigma, weight, log_directory, log_name, early_stop=float('inf'), device='cpu', test_interval=5, sample=3, config=None):
    log_dict = {'epochs': [], 'loss': [], 'loss_train': []}
    best_log_dict = {'epoch_index': 0, 'loss_valid': 1e8, 'loss_test': 1e8, 'loss_train': 1e8}

    start =time.perf_counter()
    for epoch_index in range(1, 2500+1):
        loss_train = train_single_epoch(model, loader_train, optimizer, loss, sigma, weight, epoch_index, backprop=True, tag='train', device=device, sample=sample)
        log_dict['loss_train'].append(loss_train)

        if epoch_index % test_interval == 0:
            loss_valid = train_single_epoch(model, loader_valid, optimizer, loss, sigma, weight, epoch_index, backprop=False, tag='valid', device=device, sample=sample)
            loss_test = train_single_epoch(model, loader_test, optimizer, loss, sigma, weight, epoch_index, backprop=False, tag='test', device=device, sample=sample)
            
            log_dict['epochs'].append(epoch_index)
            log_dict['loss'].append(loss_test)
            
            if loss_valid < best_log_dict['loss_valid']:
                best_log_dict = {'epoch_index': epoch_index, 'loss_valid': loss_valid, 'loss_test': loss_test, 'loss_train': loss_train}
                name = None
                if config.dataset_name in ['5_0_0', '20_0_0', '50_0_0', '100_0_0']:
                    name = 'nbody'

                os.makedirs(f'./state_dict/{name}', exist_ok=True)
                torch.save(model.state_dict(), f'./state_dict/{name}/{model.__class__.__name__}_best_model.pth')
            print(f'*** Best Valid Loss: {best_log_dict["loss_valid"] :.5f} | Best Test Loss: {best_log_dict["loss_test"] :.5f} | Best Epoch Index: {best_log_dict["epoch_index"]}')

            if epoch_index - best_log_dict['epoch_index'] >= early_stop:
                best_log_dict['early_stop'] = epoch_index
                print(f'Early stopped! Epoch: {epoch_index}')
                break

        end = time.perf_counter() 
        time_cost = end - start
        best_log_dict['time_cost'] = time_cost
        
        json_object = json.dumps([best_log_dict, log_dict], indent=4)
        os.makedirs(log_directory, exist_ok=True)
        with open(f'{log_directory}/{log_name}', "w") as outfile:
            outfile.write(json_object)
    

    return best_log_dict, log_dict
