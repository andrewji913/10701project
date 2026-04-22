import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

epochs = list(range(10))
train_loss = [3.034666, 2.302136, 2.118319, 2.006962, 1.926535,
              1.863751, 1.812545, 1.769883, 1.733281, 1.701518]
test_loss  = [2.368976, 2.161199, 2.059809, 2.001626, 1.962736,
              1.929398, 1.905185, 1.889490, 1.872759, 1.859690]
train_acc  = [44.7151, 53.9216, 56.4981, 58.0781, 59.2386,
              60.1490, 60.9047, 61.5280, 62.0884, 62.5591]
test_acc   = [53.5475, 56.6292, 58.1607, 59.1340, 59.8525,
              60.4007, 60.7979, 61.1077, 61.4389, 61.6710]
bleu       = [23.99, 25.24, 28.14, 28.47, 30.51,
              30.23, 30.39, 31.04, 32.16, 31.38]

# Combined 2x2 dashboard
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].plot(epochs, train_loss, 'o-', label='Train')
axes[0, 0].plot(epochs, test_loss,  's-', label='Test')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Cross-Entropy Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs, train_acc, 'o-', label='Train')
axes[0, 1].plot(epochs, test_acc,  's-', label='Test')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Token Accuracy (%)')
axes[0, 1].set_title('Next-Token Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epochs, bleu, 'o-', color='tab:green')
best_ep = bleu.index(max(bleu))
axes[1, 0].annotate(f'best: {max(bleu):.2f}', xy=(best_ep, max(bleu)),
                    xytext=(best_ep - 2, max(bleu) - 1.5),
                    arrowprops=dict(arrowstyle='->'))
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('BLEU')
axes[1, 0].set_title('Validation BLEU'); axes[1, 0].grid(True, alpha=0.3)

# Generalization gap
gap_loss = [te - tr for te, tr in zip(test_loss, train_loss)]
axes[1, 1].plot(epochs, gap_loss, 'o-', color='tab:red')
axes[1, 1].axhline(0, color='k', linewidth=0.5)
axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Test loss − Train loss')
axes[1, 1].set_title('Generalization Gap'); axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_dashboard.png', dpi=150, bbox_inches='tight')
print('saved training_dashboard.png')

# Standalone BLEU chart
plt.figure(figsize=(8, 5))
plt.plot(epochs, bleu, 'o-', color='tab:green', linewidth=2)
plt.xlabel('Epoch'); plt.ylabel('BLEU')
plt.title('EN→FR Translation BLEU over Epochs')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bleu_curve.png', dpi=150, bbox_inches='tight')
print('saved bleu_curve.png')
