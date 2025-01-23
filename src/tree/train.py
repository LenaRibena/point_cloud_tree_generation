import os
from datetime import datetime

import hydra
import torch
import torch.utils.tensorboard
from dotenv import load_dotenv
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import wandb
from tree.data import PCTreeDataset
from tree.models.flow import add_spectral_norm, spectral_norm_power_iteration
from tree.models.vae_flow import FlowVAE
from tree.models.vae_gaussian import GaussianVAE
from tree.utils.train_utils import EarlyStopper, update_hydra_config


@hydra.main(version_base="1.2", config_path=to_absolute_path("configs"), config_name="train")
def train(args):
    # Set random seed
    torch.manual_seed(args.seed)

    # Configure and initialize wandb
    mode = "disabled" if args.debug is True else "online"

    excluded_keys = ("hydra", "debug", "device", "num_workers", "data_path")
    if args.model == "gaussian":
        excluded_keys += ("latent_flow_depth", "latent_flow_hidden_dim")

    experiment_name = f"experiment-{datetime.now():%Y-%m-%d}-{datetime.now().strftime('%H-%M-%S')}"
    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"),
        project="tree-pc-generator",
        name=experiment_name,
        config=wandb.helper.parse_config(
            OmegaConf.to_container(args, resolve=True, throw_on_missing=True), exclude=excluded_keys
        ),
        mode=mode,
    )

    if args.debug is True:
        logger.debug("Debug mode enabled.")
    else:
        # Configure the logger
        # NOTE: If you wish to not log to stdout, use: logger.remove()
        logger.add(
            os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "pc_tree_log.log"),
            level="INFO",
            rotation="100 MB",
        )
        logger.add(os.path.join(wandb.run.dir, "pc_tree_log.log"), level="INFO")
        logger.info("Debug mode disabled.")

    logger.info(args)
    logger.info(f"Experiment name: {experiment_name}.")

    # Create train, val and test loaders from the dataset
    dset = PCTreeDataset(
        processed_data_path=to_absolute_path(os.path.join(*args.data_path)),
        device=args.device,
        transform=args.transform,
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

            x = batch.float().to(dset.device)

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

            x = batch.float().to(dset.device)
            with torch.no_grad():
                val_loss += model.get_loss(x, kl_weight=kl_weight).item() * x.size(0)

        val_loss /= len(val_iter.dataset)

        logger.info(f"[Train] Average loss: {train_loss}, Validation loss: {val_loss}")
        run.log({"Train/loss": train_loss, "Val/loss": val_loss}, step=epoch)

        # Early stopping if validation loss does not improve
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            logger.info("Early stopping...")

            break

        if epoch % 10 == 0:
            MODEL_CHECKPOINT_SAVE_PATH = to_absolute_path(
                os.path.join(args.experiment_output_dir, f"{args.model}_e-{epoch}.pth")
            )
            torch.save(early_stopper.best_model_state, MODEL_CHECKPOINT_SAVE_PATH)

    logger.info("Training complete.")

    # Save the model
    logger.info("Saving model...")
    MODEL_SAVE_PATH = to_absolute_path(os.path.join(args.experiment_output_dir, f"{args.model}_best_model.pth"))
    torch.save(early_stopper.best_model_state, MODEL_SAVE_PATH)

    artifact = wandb.Artifact(
        name="PC_tree_model",
        type="model",
        description="A model trained to generate point clouds of tree structures.",
        metadata={"Best validation loss": -early_stopper.best_score},
    )
    artifact.add_file(MODEL_SAVE_PATH)
    run.log_artifact(artifact)

    # Load the best model and test it on the test set
    early_stopper.load_best_model(model)

    logger.info("Testing the model on the test set...")
    model.eval()
    test_loss = 0
    for k, batch in enumerate(test_iter):
        if args.debug and k > 1:
            break

        x = batch.float().to(dset.device)
        with torch.no_grad():
            test_loss += model.get_loss(x, kl_weight=kl_weight).item() * x.size(0)

    test_loss /= len(test_iter.dataset)
    logger.info(f"[Test] average loss: {test_loss}")
    run.log({"test/loss": test_loss})
    logger.info("Testing complete.")

    run.finish()


if __name__ == "__main__":
    # Only create hydra outputs if debug mode is disabled
    config_path = os.path.join("configs", "train.yaml")
    debug_status = update_hydra_config(config_path)

    # Only log into wandb if debug mode is disabled
    if debug_status is False:
        load_dotenv()
        LOGIN_KEY = os.getenv("WANDB_API_KEY")
        wandb.login(key=LOGIN_KEY)

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
