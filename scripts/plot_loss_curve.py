from matplotlib import pyplot as plt
import pandas as pd


loss_file = '/home/dist/hpcai/duanjunwen/Open-Sora/loss_curve.csv'
df = pd.read_csv(loss_file)
plt.plot(df, label='train_loss')
plt.legend()
plt.show()
plt.savefig('/home/dist/hpcai/duanjunwen/Open-Sora/train_loss.png')