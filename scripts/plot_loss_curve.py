from matplotlib import pyplot as plt
import pandas as pd


loss_file = '/home/dist/hpcai/duanjunwen/Open-Sora/loss_curve/loss_curve_2024-06-12 15:44:26.393857.csv'
df = pd.read_csv(loss_file)
plt.plot(df, label='train_loss')
plt.legend()
plt.show()
plt.savefig('/home/dist/hpcai/duanjunwen/Open-Sora/loss_curve/loss_curve_2024-06-12 15:44:26.393857.png')