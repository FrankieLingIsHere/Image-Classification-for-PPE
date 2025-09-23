#!/usr/bin/env python3
"""
Plot Training and Validation Loss Curves from Checkpoints
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

def extract_losses_from_checkpoints():
    """Extract training and validation losses from all checkpoint files"""
    
    print("📊 Extracting Training History from Checkpoints")
    print("=" * 50)
    
    # Find all checkpoint files
    checkpoint_pattern = "models/checkpoint_epoch_*.pth"
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print("❌ No checkpoint files found!")
        return None, None, None
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    epochs = []
    train_losses = []
    val_losses = []
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    
    for checkpoint_file in checkpoint_files:
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            
            epoch = checkpoint.get('epoch', 0)
            train_loss = checkpoint.get('loss', 0)
            val_loss = checkpoint.get('val_loss', None)
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            
            # Handle missing validation loss (from older checkpoints)
            if val_loss is not None:
                val_losses.append(val_loss)
            else:
                val_losses.append(None)
            
            print(f"  Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss if val_loss else 'N/A'}")
            
        except Exception as e:
            print(f"  ❌ Error loading {checkpoint_file}: {e}")
    
    return epochs, train_losses, val_losses

def plot_training_curves(epochs, train_losses, val_losses, save_path="training_curves.png"):
    """Plot comprehensive training curves"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PPE Detection Model Training Analysis', fontsize=16, fontweight='bold')
    
    # 1. Main Loss Curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    
    # Only plot validation if we have the data
    val_epochs = [e for e, v in zip(epochs, val_losses) if v is not None]
    val_data = [v for v in val_losses if v is not None]
    
    if val_data:
        ax1.plot(val_epochs, val_data, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # 2. Training Loss Improvement
    ax2 = axes[0, 1]
    if len(train_losses) > 1:
        initial_loss = train_losses[0]
        improvements = [(initial_loss - loss) / initial_loss * 100 for loss in train_losses]
        ax2.plot(epochs, improvements, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Training Loss Improvement')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. Loss Reduction Rate
    ax3 = axes[1, 0]
    if len(train_losses) > 1:
        loss_reductions = []
        for i in range(1, len(train_losses)):
            reduction = (train_losses[i-1] - train_losses[i]) / train_losses[i-1] * 100
            loss_reductions.append(reduction)
        
        ax3.bar(epochs[1:], loss_reductions, alpha=0.7, color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Reduction (%)')
        ax3.set_title('Epoch-to-Epoch Loss Reduction')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    # 4. Training Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    if train_losses:
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        best_loss = min(train_losses)
        total_improvement = (initial_loss - final_loss) / initial_loss * 100
        best_improvement = (initial_loss - best_loss) / initial_loss * 100
        
        stats_text = f"""📊 Training Statistics:
        
📈 Initial Loss: {initial_loss:.4f}
📉 Final Loss: {final_loss:.4f}
🏆 Best Loss: {best_loss:.4f}
📊 Total Improvement: {total_improvement:.1f}%
🎯 Best Improvement: {best_improvement:.1f}%
🔢 Total Epochs: {len(epochs)}
        """
        
        if val_data:
            best_val_loss = min(val_data)
            final_val_loss = val_data[-1]
            stats_text += f"""
🔍 Validation Results:
📉 Final Val Loss: {final_val_loss:.4f}
🏆 Best Val Loss: {best_val_loss:.4f}
📊 Generalization Gap: {abs(final_loss - final_val_loss):.4f}"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    return fig

def analyze_training_progress():
    """Comprehensive training analysis"""
    
    print("🔍 Training Progress Analysis")
    print("=" * 50)
    
    epochs, train_losses, val_losses = extract_losses_from_checkpoints()
    
    if not epochs or not train_losses:
        print("❌ No training data found!")
        return
    
    # Print detailed analysis
    print(f"\n📈 Training Summary:")
    print(f"  Total Epochs: {len(epochs)}")
    print(f"  Epoch Range: {min(epochs)} → {max(epochs)}")
    print(f"  Initial Loss: {train_losses[0]:.4f}")
    print(f"  Final Loss: {train_losses[-1]:.4f}")
    print(f"  Best Loss: {min(train_losses):.4f}")
    
    improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    print(f"  Total Improvement: {improvement:.1f}%")
    
    # Check for validation data
    val_data = [v for v in val_losses if v is not None] if val_losses else []
    if val_data:
        print(f"\n🔍 Validation Summary:")
        print(f"  Final Val Loss: {val_data[-1]:.4f}")
        print(f"  Best Val Loss: {min(val_data):.4f}")
        
        # Check for overfitting
        gap = abs(train_losses[-1] - val_data[-1])
        print(f"  Generalization Gap: {gap:.4f}")
        
        if gap > 0.1:
            print("  ⚠️ Potential overfitting detected!")
        else:
            print("  ✅ Good generalization!")
    else:
        print("\n⚠️ No validation data found in checkpoints")
    
    # Learning rate analysis
    print(f"\n📊 Learning Progress:")
    for i in range(1, min(len(train_losses), 6)):  # Show first 5 improvements
        reduction = (train_losses[i-1] - train_losses[i]) / train_losses[i-1] * 100
        print(f"  Epoch {epochs[i-1]} → {epochs[i]}: {reduction:+.1f}% change")
    
    # Plot the curves
    print(f"\n🎨 Generating Training Curves...")
    plot_training_curves(epochs, train_losses, val_losses if val_losses else [])
    
    # Training recommendations
    print(f"\n💡 Training Recommendations:")
    
    recent_losses = train_losses[-3:] if len(train_losses) >= 3 else train_losses
    if len(recent_losses) > 1:
        recent_improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0] * 100
        
        if recent_improvement > 5:
            print("  ✅ Model is still improving well - continue training!")
        elif recent_improvement > 1:
            print("  📈 Slow improvement - consider reducing learning rate")
        else:
            print("  ⏹️ Minimal improvement - consider stopping or changing strategy")
    
    if val_data and len(val_data) > 1:
        if val_data[-1] > min(val_data):
            print("  ⚠️ Validation loss increasing - consider early stopping")
        else:
            print("  ✅ Validation loss still improving")

if __name__ == "__main__":
    analyze_training_progress()