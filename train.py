import argparse
import configparser
import copy
import time

import numpy as np
import torch

import util
from componenets.engine import trainer
from componenets.metrics import metrics

DATASET = 'PEMS4'

# get configuration
config_file = './config/{}.conf'.format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)

parser = argparse.ArgumentParser()
parser.add_argument('--device', default=config['general']['device'], type=str)
parser.add_argument('--data', default=DATASET, help='data path', type=str, )
parser.add_argument('--adj_filename', type=str, default=config['data']['adj_filename'])
parser.add_argument('--id_filename', type=str, default=config['data']['id_filename'])
parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])
parser.add_argument('--num_nodes', type=int, default=config['data']['num_nodes'])
parser.add_argument('--lag', type=int, default=config['data']['lag'])
parser.add_argument('--horizon', type=int, default=config['data']['horizon'])

parser.add_argument('--in_dim', type=int, default=config['model']['in_dim'])
parser.add_argument('--out_dim', type=int, default=config['model']['out_dim'])
parser.add_argument('--channels', type=int, default=config['model']['channels'])
parser.add_argument('--dynamic', type=str, default=config['model']['dynamic'])
parser.add_argument('--memory_size', type=int, default=config['model']['memory_size'])

parser.add_argument('--early_stop_patience', type=int, default=config['train']['early_stop_patience'])
parser.add_argument('--learning_rate', type=float, default=config['train']['learning_rate'])
parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
parser.add_argument('--epochs', type=int, default=config['train']['epochs'])
parser.add_argument('--seed', type=int, default=config['train']['seed'])
parser.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
parser.add_argument('--save', type=str, default=config['train']['save'])
parser.add_argument('--expid', type=int, default=config['train']['expid'])
parser.add_argument('--log_step', type=int, default=config['train']['log_step'])
parser.add_argument('--mae_thresh', type=float, default=config['test']['mae_thresh'])
parser.add_argument('--mape_thresh', type=float, default=config['test']['mape_thresh'])

parser.add_argument('--column_wise', type=bool, default=False)
args = parser.parse_args()
args.dynamic = args.dynamic == 'True'
model_id = round(time.time() * 1000)


def main():
    util.init_seed(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader, test_loader, scaler = util.get_dataloader(args)

    adj_mx = util.get_adjacency_matrix(distance_df_filename=args.adj_filename,
                                       num_of_vertices=args.num_nodes, id_filename=args.id_filename)
    adj_mx = util.scaled_Laplacian(adj_mx)
    adj_mx = [adj_mx]
    supports = [torch.tensor(i.astype('float32')).to(device) for i in adj_mx]
    engine = trainer(device=device, scaler=scaler, lrate=args.learning_rate, num_nodes=args.num_nodes,
                     input_dim=args.in_dim, output_dim=args.out_dim, channels=args.channels, grad_norm=args.grad_norm,
                     dynamic=args.dynamic, lag=args.lag, horizon=args.horizon, supports=supports,
                     memory_size=args.memory_size)

    print('model initialization')
    util.print_model_parameters(engine.model, only_num=True)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    best_iteration = 0
    best_val_error = 999
    best_state_dict = None
    not_improved_count = 0
    train_per_epoch = len(train_loader)
    for i in range(1, args.epochs + 1):
        train_loss = 0
        train_kld = 0
        train_rec = 0
        t1 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            trainx = data[..., :args.in_dim].to(args.device)
            trainy = target[..., :args.out_dim].to(args.device)
            tloss, kld, rec = engine.train(trainx, trainy, i)
            train_loss += tloss
            train_rec += rec
            train_kld += kld

        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss = 0
        s1 = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            with torch.no_grad():
                testx = data[..., :args.in_dim].to(args.device)
                testy = target[..., :args.out_dim].to(args.device)
                loss = engine.eval(testx, testy)
                valid_loss += loss
        s2 = time.time()

        val_time.append(s2 - s1)
        mtrain_loss = train_loss / train_per_epoch
        mtrain_rec = train_rec / train_per_epoch
        mtrain_kld = train_kld / train_per_epoch
        mvalid_loss = valid_loss / len(val_loader)
        his_loss.append(mvalid_loss)

        for param_group in engine.optimizer.param_groups:
            lr = param_group['lr']

        log = 'Epoch: {:03d}, Prediction Loss: {:.4f}, Reconstruction Loss: {:.4f}, KLD Loss: {:.4f},' \
              ' Valid Loss: {:.4f}, LR: {:.4f},  Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_rec, mtrain_kld, mvalid_loss, lr, (t2 - t1)), flush=True)
        if mvalid_loss < best_val_error:
            best_iteration = i
            best_val_error = mvalid_loss
            best_state_dict = copy.deepcopy(engine.model.state_dict())
            not_improved_count = 0
        else:
            not_improved_count += 1

        if not_improved_count == args.early_stop_patience:
            print("Validation performance didn\'t improve for {} epochs. "
                  "Training stops.".format(args.early_stop_patience))
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Best iteration: {:.4f} secs".format(best_iteration))

    # testing
    engine.model.load_state_dict(best_state_dict)
    engine.model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data[..., :args.in_dim].to(args.device)
            label = target[..., :args.out_dim].to(args.device)
            output = engine.model(data)
            y_true.append(label)
            y_pred.append(output)

    y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

    for t in range(y_true.shape[1]):
        mae, rmse, mape, _, _ = metrics(y_pred[:, t, ...], y_true[:, t, ...], args.mae_thresh, args.mape_thresh)
        print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
            t + 1, mae, rmse, mape * 100))
    mae, rmse, mape, _, _ = metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
    print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
        mae, rmse, mape * 100))

    torch.save(best_state_dict, args.save + "_exp_best.pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
