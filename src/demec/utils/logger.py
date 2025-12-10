"""
Experiment logging utilities for TensorBoard and model checkpointing.
"""

import os
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    Handles TensorBoard logging and model checkpointing for training experiments.
    """
    
    def __init__(self, args, log_dir="runs", checkpoint_dir="checkpoints"):
        """
        Initialize experiment logger.
        
        Args:
            args: Argument namespace with experiment configuration
            log_dir: Base directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.args = args
        self.checkpoint_dir = checkpoint_dir
        
        # Generate experiment name if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.exp_name is None:
            graph_type = "hetero" if args.hetero else "homo"
            # Use train_tasks for experiment name
            train_tasks = getattr(args, 'train_tasks', getattr(args, 'tasks', 'unknown'))
            tasks_str = "_".join(train_tasks.split(',')[:2])  # First 2 tasks
            self.exp_name = f"{graph_type}_{args.model}_{tasks_str}_{timestamp}"
        else:
            # Use provided experiment name with timestamp
            self.exp_name = f"{args.exp_name}_{timestamp}"
        
        # Setup TensorBoard writer
        self.log_path = os.path.join(log_dir, self.exp_name)
        self.writer = SummaryWriter(self.log_path)
        
        # Track best model
        self.best_val_map = 0.0
        self.best_epoch = 0
        
        print(f"Experiment name: {self.exp_name}")
        print(f"TensorBoard logs: {self.log_path}")
    
    def log_hyperparameters(self, task_weights):
        """
        Log hyperparameters to TensorBoard.
        
        Args:
            task_weights: Dictionary of task names to weights
        """
        hparams = {
            'model': self.args.model,
            'hetero': self.args.hetero,
            'hidden_dim': self.args.hidden_dim,
            'num_layers': self.args.num_layers,
            'dropout': self.args.dropout,
            'lr': self.args.lr,
            'batch_size': self.args.batch_size,
            'train_tasks': getattr(self.args, 'train_tasks', getattr(self.args, 'tasks', 'unknown')),
            'eval_tasks': getattr(self.args, 'eval_tasks', getattr(self.args, 'tasks', 'unknown')),
            'focal_alpha': self.args.focal_alpha if self.args.focal_alpha else 0.25,
            'clip_grad_norm': self.args.clip_grad_norm if self.args.clip_grad_norm else 0.0,
        }
        
        # Add task weights
        for task, weight in task_weights.items():
            hparams[f'weight_{task}'] = weight
        
        self.hparams = hparams
    
    def log_epoch(self, epoch, train_loss, val_loss, train_task_losses, val_task_losses,
                  train_metrics, val_metrics, train_size, val_size, optimizer):
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Total training loss
            val_loss: Total validation loss
            train_task_losses: Dictionary of per-task training losses
            val_task_losses: Dictionary of per-task validation losses
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            train_size: Size of training set
            val_size: Size of validation set
            optimizer: Optimizer (for learning rate logging)
        """
        # Normalize losses
        avg_train_loss = train_loss / train_size
        avg_val_loss = val_loss / val_size
        
        # 1. Overall losses
        self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
        self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        # 2. Per-task losses
        for task_name, task_loss in train_task_losses.items():
            self.writer.add_scalar(f'TaskLoss/train_{task_name}', task_loss / train_size, epoch)
        for task_name, task_loss in val_task_losses.items():
            self.writer.add_scalar(f'TaskLoss/val_{task_name}', task_loss / val_size, epoch)
        
        # 3. Side effects metrics (primary task)
        if train_metrics:
            self.writer.add_scalar('Metrics/train_mAP', train_metrics.get('mAP', 0), epoch)
            self.writer.add_scalar('Metrics/train_P@50', train_metrics.get('P@50', 0), epoch)
            self.writer.add_scalar('Metrics/train_P@100', train_metrics.get('P@100', 0), epoch)
            self.writer.add_scalar('Metrics/train_AUROC', train_metrics.get('AUROC', 0), epoch)
        
        if val_metrics:
            self.writer.add_scalar('Metrics/val_mAP', val_metrics.get('mAP', 0), epoch)
            self.writer.add_scalar('Metrics/val_P@50', val_metrics.get('P@50', 0), epoch)
            self.writer.add_scalar('Metrics/val_P@100', val_metrics.get('P@100', 0), epoch)
            self.writer.add_scalar('Metrics/val_AUROC', val_metrics.get('AUROC', 0), epoch)
        
        # 4. Learning rate
        self.writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        return avg_train_loss, avg_val_loss
    
    def save_checkpoint(self, epoch, model, optimizer, val_map):
        """
        Save model checkpoint if it's the best so far.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            val_map: Current validation mAP
            
        Returns:
            bool: True if checkpoint was saved
        """
        if val_map > self.best_val_map:
            self.best_val_map = val_map
            self.best_epoch = epoch + 1
            
            if self.args.save_model:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.exp_name}_best.pt")
                
                # Convert args to dict to avoid pickle issues with custom classes
                args_dict = vars(self.args) if hasattr(self.args, '__dict__') else self.args
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mAP': self.best_val_map,
                    'args': args_dict,
                }, checkpoint_path)
                
                self.checkpoint_path = checkpoint_path
                return True
        
        return False
    
    def finalize(self, final_val_metrics):
        """
        Finalize logging and close writer.
        
        Args:
            final_val_metrics: Final validation metrics
        """
        # Log hyperparameters with final metrics
        final_metrics = {
            'hparam/val_mAP': self.best_val_map,
            'hparam/val_P@50': final_val_metrics.get('P@50', 0),
            'hparam/val_AUROC': final_val_metrics.get('AUROC', 0),
            'hparam/best_epoch': self.best_epoch,
        }
        self.writer.add_hparams(self.hparams, final_metrics)
        
        # Close writer
        self.writer.close()
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"Training complete!")
        print(f"Best validation mAP: {self.best_val_map:.4f} (epoch {self.best_epoch})")
        print(f"TensorBoard logs: {self.log_path}")
        if self.args.save_model and hasattr(self, 'checkpoint_path'):
            print(f"Best model saved: {self.checkpoint_path}")
        print(f"\nTo view results: tensorboard --logdir {os.path.dirname(self.log_path)}")
        print("=" * 80)


def format_metrics_string(metrics):
    """
    Format metrics dictionary into a compact string for console output.
    Handles different metric types (classification, fingerprint, regression).
    
    Args:
        metrics: Dictionary of metric names to values
        
    Returns:
        str: Formatted string
    """
    if not metrics:
        return "N/A"
    
    # Regression metrics (molprops)
    if 'MSE' in metrics:
        return (f"MSE:{metrics.get('MSE', 0):.4f} "
                f"MAE:{metrics.get('MAE', 0):.4f} "
                f"R²:{metrics.get('R2', 0):.3f}")
    
    # Fingerprint metrics (MACCS)
    elif 'Hamming' in metrics:
        return (f"Hamming:{metrics.get('Hamming', 0):.3f} "
                f"Tanimoto:{metrics.get('Tanimoto', 0):.3f} "
                f"Bit_Acc:{metrics.get('Bit_Acc', 0):.3f} "
                f"AUROC:{metrics.get('AUROC', 0):.3f}")
    
    # Classification metrics (side_effects, ATC)
    else:
        return (f"mAP:{metrics.get('mAP', 0):.3f} "
                f"P@1:{metrics.get('P@1', 0):.3f} "
                f"P@5:{metrics.get('P@5', 0):.3f} "
                f"Top1:{metrics.get('Top1_Acc', 0):.3f} "
                f"AUROC:{metrics.get('AUROC', 0):.3f}")


def format_task_losses(task_losses, dataset_size):
    """
    Format per-task losses into a string for console output.
    
    Args:
        task_losses: Dictionary of task names to losses
        dataset_size: Size of dataset for normalization
        
    Returns:
        str: Formatted string
    """
    return " | ".join([f"{task}: {loss/dataset_size:.4f}" 
                       for task, loss in task_losses.items()])
