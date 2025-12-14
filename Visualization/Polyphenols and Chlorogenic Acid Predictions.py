import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# y_true_train = ...
# y_pred_train = ...
# y_true_val   = ...
# y_pred_val   = ...
# y_true_test  = ...
# y_pred_test  = ...

def col(arr, idx):      
    return arr[:, idx].ravel()

t_train_p = col(y_true_train, 0);  o_train_p = col(y_pred_train, 0)
t_val_p   = col(y_true_val,   0);  o_val_p   = col(y_pred_val,   0)
t_test_p  = col(y_true_test,  0);  o_test_p  = col(y_pred_test,  0)
t_all_p   = np.hstack([t_train_p,  t_val_p,  t_test_p])
o_all_p   = np.hstack([o_train_p,  o_val_p,  o_test_p])


t_train_c = col(y_true_train, 1);  o_train_c = col(y_pred_train, 1)
t_val_c   = col(y_true_val,   1);  o_val_c   = col(y_pred_val,   1)
t_test_c  = col(y_true_test,  1);  o_test_c  = col(y_pred_test,  1)
t_all_c   = np.hstack([t_train_c,  t_val_c,  t_test_c])
o_all_c   = np.hstack([o_train_c,  o_val_c,  o_test_c])

def fit_line(x, y):
    a, b = np.polyfit(x, y, 1)
    return np.polyval([a, b], x), a, b        

def plot_four(axs, groups,
              mse_pos=(0.88, 0.06),   
              mse_gap=0.005,           
              mse_font=11,              
              data_sz=14, font_sz=10):  
    """
    groups = [(target, output, subtitle, color), ...]  Total 4 groups
    """
    for ax, (t, o, sub, color) in zip(axs.ravel(), groups):

        ax.scatter(t, o, c='k', s=data_sz, label='Data')
        y_fit, a, b = fit_line(t, o)
        ax.plot(t, y_fit, color=color, lw=1.6, label='Fit')
        ax.plot(t, t,  'k--', lw=1, label='Y = T')

        r2  = r2_score(t, o)
        mse = mean_squared_error(t, o)
        ax.set_title(fr"{sub}: $R^2$ = {r2:.5f}",
                     fontsize=font_sz+1, pad=9)

        mx, my = mse_pos
        ax.text(mx,        my, "MSE:",
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=mse_font, fontweight='bold')
        ax.text(mx + mse_gap, my, f"{mse:.4f}",
                transform=ax.transAxes,
                ha='left',  va='bottom',
                fontsize=mse_font, fontweight='bold')

        ax.set_ylabel(f"Output = {a:.3f}·Target + {b:.3f}",
                      rotation=90, labelpad=8, fontsize=font_sz-1)
        ax.set_xlabel("Target", fontsize=font_sz-1)
        ax.tick_params(labelsize=font_sz-2)
        ax.legend(fontsize=font_sz-2)

fig_p, axs_p = plt.subplots(2, 2, figsize=(10, 8))
poly_groups = [
    (t_train_p, o_train_p, 'Training',   'blue'),
    (t_val_p,   o_val_p,   'Validation', 'green'),
    (t_test_p,  o_test_p,  'Test',       'red'),
    (t_all_p,   o_all_p,   'All',        'purple')
]
plot_four(axs_p, poly_groups,
          mse_pos=(0.84, 0.10), 
          mse_gap=0.004,        
          mse_font=12)
fig_p.suptitle("Polyphenols", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

fig_c, axs_c = plt.subplots(2, 2, figsize=(10, 8))
chl_groups = [
    (t_train_c, o_train_c, 'Training',   'blue'),
    (t_val_c,   o_val_c,   'Validation', 'green'),
    (t_test_c,  o_test_c,  'Test',       'red'),
    (t_all_c,   o_all_c,   'All',        'purple')
]
plot_four(axs_c, chl_groups,
          mse_pos=(0.84, 0.10),
          mse_gap=0.004,
          mse_font=12)
fig_c.suptitle("Chlorogenic Acid", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
