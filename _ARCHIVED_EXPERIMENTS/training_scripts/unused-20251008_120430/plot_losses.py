# Training and Validation Loss Visualization
import torch
import matplotlib.pyplot as plt
import os
import glob

def extract_losses_from_checkpoints():
    """Extract training and validation losses from checkpoint files"""
    
    checkpoint_pattern = "models/checkpoint_epoch_*.pth"
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Skip the first 5 files as they were trained differently
    checkpoint_files = [f for f in checkpoint_files if int(f.split('_')[-1].split('.')[0]) >= 5]
    
    epochs = []
    train_losses = []
    val_losses = []
    
    print("Found {} checkpoint files".format(len(checkpoint_files)))
    
    for checkpoint_file in checkpoint_files:
        try:
            # Extract epoch number from filename
            epoch = int(checkpoint_file.split('_')[-1].split('.')[0])
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            # Extract losses - use 'loss' key for training loss
            train_loss = checkpoint.get('loss')  # This is the training loss
            val_loss = checkpoint.get('val_loss')  # This might not exist in older checkpoints
            
            if train_loss is not None:
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)  # Can be None
                
                print("Epoch {}: Train Loss = {:.4f}, Val Loss = {}".format(
                    epoch, train_loss, val_loss if val_loss else 'N/A'))
        
        except Exception as e:
            print("Error loading {}: {}".format(checkpoint_file, e))
            continue
    
    return epochs, train_losses, val_losses

def plot_training_curves():
    """Create training and validation loss visualization"""
    
    print("Extracting training data from checkpoints...")
    epochs, train_losses, val_losses = extract_losses_from_checkpoints()
    
    if not epochs:
        print("No training data found!")
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=6, label='Training Loss')
    
    # Plot validation loss if available
    val_epochs = []
    val_data = []
    for i, val_loss in enumerate(val_losses):
        if val_loss is not None:
            val_epochs.append(epochs[i])
            val_data.append(val_loss)
    
    if val_data:
        plt.plot(val_epochs, val_data, 'r-s', linewidth=2, markersize=6, label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training improvement
    plt.subplot(2, 2, 2)
    if len(train_losses) > 1:
        improvements = []
        for i in range(1, len(train_losses)):
            improvement = (train_losses[0] - train_losses[i]) / train_losses[0] * 100
            improvements.append(improvement)
        
        plt.plot(epochs[1:], improvements, 'g-^', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Improvement from Start (%)')
        plt.title('Cumulative Training Improvement')
        plt.grid(True, alpha=0.3)
    
    # Plot epoch-to-epoch changes
    plt.subplot(2, 2, 3)
    if len(train_losses) > 1:
        changes = []
        for i in range(1, len(train_losses)):
            change = (train_losses[i-1] - train_losses[i]) / train_losses[i-1] * 100
            changes.append(change)
        
        colors = ['green' if c > 0 else 'red' for c in changes]
        plt.bar(epochs[1:], changes, color=colors, alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Reduction (%)')
        plt.title('Epoch-to-Epoch Loss Changes')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Calculate statistics
    initial_loss = train_losses[0]
    final_loss = train_losses[-1]
    best_loss = min(train_losses)
    total_improvement = (initial_loss - final_loss) / initial_loss * 100
    
    stats_text = "Training Statistics:\n\n"
    stats_text += "Initial Loss: {:.4f}\n".format(initial_loss)
    stats_text += "Final Loss: {:.4f}\n".format(final_loss)
    stats_text += "Best Loss: {:.4f}\n".format(best_loss)
    stats_text += "Total Epochs: {}\n".format(len(epochs))
    stats_text += "Total Improvement: {:.1f}%\n".format(total_improvement)
    
    if val_data:
        final_val_loss = val_data[-1]
        best_val_loss = min(val_data)
        gap = abs(final_loss - final_val_loss)
        
        stats_text += "\nValidation Results:\n"
        stats_text += "Final Val Loss: {:.4f}\n".format(final_val_loss)
        stats_text += "Best Val Loss: {:.4f}\n".format(best_val_loss)
        stats_text += "Train-Val Gap: {:.4f}\n".format(gap)
        
        if gap > 0.1:
            stats_text += "Status: Potential overfitting"
        else:
            stats_text += "Status: Good generalization"
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'training_loss_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("\nTraining curves saved to: {}".format(output_path))
    
    # Show summary
    print("\nTraining Summary:")
    print("- Trained for {} epochs ({} to {})".format(len(epochs), min(epochs), max(epochs)))
    print("- Loss improved from {:.4f} to {:.4f}".format(initial_loss, final_loss))
    print("- Total improvement: {:.1f}%".format(total_improvement))
    
    if val_data:
        print("- Validation loss: {:.4f}".format(final_val_loss))
        if gap > 0.1:
            print("- Warning: Large train-validation gap ({:.4f}) - possible overfitting".format(gap))
        else:
            print("- Good generalization (gap: {:.4f})".format(gap))
    
    plt.show()

if __name__ == "__main__":
    plot_training_curves()