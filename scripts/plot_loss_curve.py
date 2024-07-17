from matplotlib import pyplot as plt
import pandas as pd

file_name = "/home/dist/hpcai/duanjunwen/Open-Sora/loss_curve/musa_loss_curve_2024-07-16 16:53:10.249999"
loss_file = f'{file_name}.csv'
df = pd.read_csv(loss_file)
plt.plot(df, label='train_loss')
plt.legend()
plt.show()
plt.savefig(f'{file_name}.png')