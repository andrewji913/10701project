import matplotlib.pyplot as plt

# Data extracted from the last complete run in tmux_output.txt (epochs 0-9)
epochs = list(range(10))

train_loss = [3.034666, 2.302136, 2.118319, 2.006962, 1.926535,
              1.863751, 1.812545, 1.769883, 1.733281, 1.701518]

test_loss = [2.368976, 2.161199, 2.059809, 2.001626, 1.962736,
             1.929398, 1.905185, 1.889490, 1.872759, 1.859690]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=8)
plt.plot(epochs, test_loss, 'r-s', label='Test Loss', linewidth=2, markersize=8)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Test Loss Over Time\n(Decoder-Only RoPE Transformer, EN→FR Translation)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(epochs)

plt.tight_layout()
plt.savefig('loss_plot.png', dpi=150)
plt.savefig('loss_plot.pdf')
print("Saved loss_plot.png and loss_plot.pdf")
