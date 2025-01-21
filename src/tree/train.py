import logging
import os

import hydra
import torch
import torch.utils.tensorboard
from hydra.utils import to_absolute_path
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tree.data import PCTreeDataset
from tree.models.flow import add_spectral_norm, spectral_norm_power_iteration
from tree.models.vae_flow import FlowVAE
from tree.models.vae_gaussian import GaussianVAE
from tree.utils import EarlyStopper, update_hydra_config


@hydra.main(version_base="1.2", config_path=to_absolute_path("configs"), config_name="default_config")
def train(args):
    logger.info(args)

    if args.debug is True:
        logger.debug("Debug mode enabled.")
    else:
        logger.info("Debug mode disabled.")

    # Set random seed
    torch.manual_seed(args.seed)

    # Create train, val and test loaders from the dataset
    dset = PCTreeDataset(
        raw_data_path=to_absolute_path(os.path.join(*args.data_path)), device=args.device, transform=args.transform
    )

    train_iter, val_iter, test_iter = dset.get_train_val_test_loaders(
        train_ratio=args.train_split, val_ratio=args.val_split, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Create the model
    logger.info("Building model...")
    model = GaussianVAE(args).to(dset.device)
    if args.model == "gaussian":
        model = GaussianVAE(args).to(dset.device)
    elif args.model == "flow":
        model = FlowVAE(args).to(dset.device)
    logger.info("Using model: %s" % args.model)
    logger.info(repr(model))
    if args.spectral_norm:
        add_spectral_norm(model, logger=logger)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler = get_linear_scheduler(
    #     optimizer,
    #     start_epoch=args.sched_start_epoch,
    #     end_epoch=args.sched_end_epoch,
    #     start_lr=args.lr,
    #     end_lr=args.end_lr
    # )

    # Early stopping
    early_stopper = EarlyStopper(patience=args.patience, delta=args.delta)

    # Training loop
    logger.info("Start training...")
    model.train()
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = 0

        for i, batch in enumerate(train_iter):
            if args.debug and i > 1:
                break

            x = batch.to(dset.device)

            # Reset grad and model state
            optimizer.zero_grad()
            if args.spectral_norm:
                spectral_norm_power_iteration(model, n_power_iterations=1)

            # Define Kullback-Leibler weighing and compute loss
            kl_weight = args.kl_weight
            loss = model.get_loss(x, kl_weight=kl_weight)  # , writer=writer, it=it
            train_loss += loss.item() * x.size(0)

            # Backward and optimize
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            # scheduler.step()

            if i % 100 == 0:
                logger.info(
                    "[Train] Epoch %i | Iter %i | Loss %.6f | Grad %.4f | KLWeight %.4f"
                    % (epoch, i, loss.item(), orig_grad_norm, kl_weight)
                )

        train_loss /= len(train_iter.dataset)

        # Validation loop
        model.eval()
        val_loss = 0
        for j, batch in enumerate(val_iter):
            if args.debug and j > 1:
                break

            x = batch.to(dset.device)
            with torch.no_grad():
                val_loss += model.get_loss(x, kl_weight=kl_weight).item() * x.size(0)

        val_loss /= len(val_iter.dataset)

        logger.info("Epoch %i | [Train] Averaged loss %.6f | [Val] Averaged loss %.6f" % (epoch, train_loss, val_loss))

        # Early stopping if validation loss does not improve
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            logger.info("Early stopping...")

            break

    # Save the model
    logger.info("Saving model...")
    model_path = to_absolute_path(os.path.join("models", f"{args.model}_model.pth"))
    torch.save(early_stopper.best_model_state, model_path)

    # Load the best model and test it on the test set
    early_stopper.load_best_model(model)

    logger.info("Testing the model on the test set...")
    model.eval()
    test_loss = 0
    for k, batch in enumerate(test_iter):
        if args.debug and k > 1:
            break

        x = batch.to(dset.device)
        with torch.no_grad():
            test_loss += model.get_loss(x, kl_weight=kl_weight).item() * x.size(0)

    test_loss /= len(test_iter.dataset)
    logger.info("[Test] Averaged loss %.6f" % test_loss)
    logger.info("Training complete.")


# Configure the logger
logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()

if __name__ == "__main__":
    # Only create hydra outputs if debug mode is disabled
    config_path = os.path.join("configs", "default_config.yaml")
    update_hydra_config(config_path)

    train()


# def validate_inspect(args):
#     z = torch.randn([args.num_samples, args.latent_dim]).to(dset.device)
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
#             z = torch.randn([args.val_batch_size, args.latent_dim]).to(dset.device)
#             x = model.sample(z, args.sample_num_points, flexibility=args.flexibility)
#             gen_pcs.append(x.detach().cpu())
#     gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]

#     # Denormalize point clouds, all shapes have zero mean.
#     # [WARNING]: Do NOT denormalize!
#     # ref_pcs *= val_dset.stats['std']
#     # gen_pcs *= val_dset.stats['std']

#     with torch.no_grad():
#         results = compute_all_metrics(gen_pcs.to(dset.device), ref_pcs.to(dset.device), args.val_batch_size)
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
