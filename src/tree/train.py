import os
import math
import argparse
import hydra
import logging
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from tree.data import PCTreeDataset
from tree.models.common import get_linear_scheduler
from tree.models.vae_gaussian import GaussianVAE
from tree.models.vae_flow import FlowVAE
from tree.models.flow import add_spectral_norm, spectral_norm_power_iteration
from tree.evaluate import compute_all_metrics, jsd_between_point_cloud_sets

import pdb

logger = logging.getLogger()

# Datasets and loaders
logger.info('Loading datasets...')

PROJECT_PATH = os.getcwd()

def custom_collate_fn(batch):
    # Find the smallest number of points in the batch
    min_points = min(item.shape[0] for item in batch)
    
    # Truncate each point cloud to the smallest number of points
    batch = [item[:min_points] for item in batch]
    
    # Stack the batch
    batch = torch.stack(batch)
    
    return batch

# Train, validate and test
@hydra.main(version_base="1.2", config_path=os.path.join(PROJECT_PATH, "configs"), config_name='default_config')
def train(args):
    train_dset = PCTreeDataset(raw_data_path=os.path.join(PROJECT_PATH, "data", "raw", "urban_tree_dataset"),
                               split='train'
                               )
    
    train_iter = DataLoader(train_dset,
                            batch_size=args.batch_size,
                            num_workers=0,
                            shuffle=True,
                            collate_fn=custom_collate_fn
                            )

    # Model
    logger.info('Building model...')
    model = GaussianVAE(args).to(args.device)
    if args.model == 'gaussian':
        model = GaussianVAE(args).to(args.device)
    elif args.model == 'flow':
        model = FlowVAE(args).to(args.device)
    logger.info(repr(model))
    if args.spectral_norm:
        add_spectral_norm(model, logger=logger)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.weight_decay)
    
    # scheduler = get_linear_scheduler(
    #     optimizer,
    #     start_epoch=args.sched_start_epoch,
    #     end_epoch=args.sched_end_epoch,
    #     start_lr=args.lr,
    #     end_lr=args.end_lr
    # )

    # Load data
    batch = next(iter(train_iter))
    x = batch.to(args.device)

    # TODO: TEMP SOLUTION
    x = x[:, :5000, :]

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight
    loss = model.get_loss(x, kl_weight=kl_weight) #, writer=writer, it=it

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    # scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        loss.item(), orig_grad_norm, kl_weight
    )) # it,

# def validate_inspect(args):
#     z = torch.randn([args.num_samples, args.latent_dim]).to(args.device)
#     x = model.sample(z, args.sample_num_points, flexibility=args.flexibility) #, truncate_std=args.truncate_std)
#     writer.add_mesh('val/pointcloud', x, global_step=it)
#     writer.flush()
#     logger.info('[Inspect] Generating samples...')

# def test(args):
#     ref_pcs = []
#     for i, data in enumerate(val_dset):
#         if i >= args.test_size:
#             break
#         ref_pcs.append(data['pointcloud'].unsqueeze(0))
#     ref_pcs = torch.cat(ref_pcs, dim=0)

#     gen_pcs = []
#     for i in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
#         with torch.no_grad():
#             z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
#             x = model.sample(z, args.sample_num_points, flexibility=args.flexibility)
#             gen_pcs.append(x.detach().cpu())
#     gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]

#     # Denormalize point clouds, all shapes have zero mean.
#     # [WARNING]: Do NOT denormalize!
#     # ref_pcs *= val_dset.stats['std']
#     # gen_pcs *= val_dset.stats['std']

#     with torch.no_grad():
#         results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.val_batch_size)
#         results = {k:v.item() for k, v in results.items()}
#         jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
#         results['jsd'] = jsd

#     # CD related metrics
#     writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], global_step=it)
#     writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], global_step=it)
#     writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], global_step=it)
#     # EMD related metrics
#     # writer.add_scalar('test/Coverage_EMD', results['lgan_cov-EMD'], global_step=it)
#     # writer.add_scalar('test/MMD_EMD', results['lgan_mmd-EMD'], global_step=it)
#     # writer.add_scalar('test/1NN_EMD', results['1-NN-EMD-acc'], global_step=it)
#     # JSD
#     writer.add_scalar('test/JSD', results['jsd'], global_step=it)

#     # logger.info('[Test] Coverage  | CD %.6f | EMD %.6f' % (results['lgan_cov-CD'], results['lgan_cov-EMD']))
#     # logger.info('[Test] MinMatDis | CD %.6f | EMD %.6f' % (results['lgan_mmd-CD'], results['lgan_mmd-EMD']))
#     # logger.info('[Test] 1NN-Accur | CD %.6f | EMD %.6f' % (results['1-NN-CD-acc'], results['1-NN-EMD-acc']))
#     logger.info('[Test] Coverage  | CD %.6f | EMD n/a' % (results['lgan_cov-CD'], ))
#     logger.info('[Test] MinMatDis | CD %.6f | EMD n/a' % (results['lgan_mmd-CD'], ))
#     logger.info('[Test] 1NN-Accur | CD %.6f | EMD n/a' % (results['1-NN-CD-acc'], ))
#     logger.info('[Test] JsnShnDis | %.6f ' % (results['jsd']))

    # # Main loop
    # logger.info('Start training...')
    # try:
    #     it = 1
    #     while it <= args.max_iters:
    #         train(it)
    #         if it % args.val_freq == 0 or it == args.max_iters:
    #             validate_inspect(it)
    #             opt_states = {
    #                 'optimizer': optimizer.state_dict(),
    #                 'scheduler': scheduler.state_dict(),
    #             }
    #             ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
    #         if it % args.test_freq == 0 or it == args.max_iters:
    #             test(it)
    #         it += 1

    # except KeyboardInterrupt:
    #     logger.info('Terminating...')

if __name__ == "__main__":
    train()