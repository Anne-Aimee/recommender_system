# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='blue')
    ax1.set_xlabel("users")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie, color='blue')
    ax2.set_xlabel("items")
    ax2.set_ylabel("number of ratings (sorted)")
    #ax2.set_xticks(np.arange(0, 2000, 300))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("../plots/stat_ratings")
    plt.show()
    # plt.close()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """visualize the train and test data."""
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.spy(train, precision=0.01, markersize=0.004)
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Items")
    ax1.set_title("Train")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.spy(test, precision=0.01, markersize=0.004)
    ax2.set_xlabel("Users")
    ax2.set_ylabel("Items")
    ax2.set_title("Test")
    fig.suptitle("Sparsity of Data")
    plt.tight_layout()
    plt.show()
    
def plot_simple_heatmap(data, title, xlabel, xticklabels, ylabel, yticklabels):
    "Plot a simple heatmap with a colorbar."
    f,a = plt.subplots()
    a.set_xlabel(xlabel)
    a.set_xticks(range(len(xticklabels)))
    a.set_xticklabels(xticklabels)
    a.set_ylabel(ylabel)
    a.set_yticks(range(len(yticklabels)))
    a.set_yticklabels(yticklabels)
    a.set_title(title)
    heatmap_corr = a.imshow(data)
    f.colorbar(heatmap_corr, ax=a)
    plt.show()

# Example of how to use:
# data = np.random.rand(5,7)
# plot_simple_heatmap(data, "title", "xlabel",np.arange(7), "ylabel",np.arange(5))

def plot_simple_heatmaps(data_1, data_2, fig_title, subtitle_1, subtitle_2, xlabel_shared, ylabel_shared):
    "Plot two heatmaps with colorbars as a single figure."
    f,a = plt.subplots(2,1)
    
    a[0].set_xlabel(xlabel_shared)
    a[0].set_ylabel(ylabel_shared)
    a[0].set_title(subtitle_1)
    heatmap_0 = a[0].imshow(data_1)

    a[1].set_xlabel(xlabel_shared)
    a[1].set_ylabel(ylabel_shared)
    a[1].set_title(subtitle_2)
    heatmap_1 = a[1].imshow(data_2)

    plt.tight_layout()
    
    f.colorbar(heatmap_0,ax=a[0])
    f.colorbar(heatmap_1,ax=a[1])
    
    f.suptitle(fig_title)
    plt.show()

# Example of how to use:
#data_1 = np.random.rand(200,300)
#data_2 = np.random.rand(200,300)
#plot_simple_heatmaps(data_1, data_2, 'fig_title', 'subtitle_1', 'subtitle_2', 'xlabel_shared', 'ylabel_shared')


def plot_gradient_descent_tuning_parameters_MF_SGD(num_features, lambda_user, lambda_item, gamma, gamma_dec_step_size, 
                                                   min_rmse_te):
    """ Plots the result of """
    f,a = plt.subplots(1,2)
    
    # plot first axis with  lambda_user, lambda_item, gamma, gamma_dec_step_size
    a[0].set_xlabel('iteration')
    a[0].set_ylabel(r'$\lambda_{user}, \lambda_{item}, \gamma, \delta_\gamma$')
    a[0].set_title('Evolution of\nParameters')
    line, = a[0].plot(min_rmse_te, label=r'$RMSE$')
    line, = a[0].plot(lambda_user, label=r'$\lambda_{user}$')
    line, = a[0].plot(lambda_item, label=r'$\lambda_{item}$')
    line, = a[0].plot(gamma, label=r'$\gamma$')
    line, = a[0].plot(gamma_dec_step_size, label=r'$\delta_\gamma$')
    # plot num_features out of visible range for the legend
    bottom, top = a[0].get_ylim()
    line, = a[0].plot(num_features,color='yellow',label=r'$K$')
    a[0].set_ylim(bottom, top)
    #a[0].legend(ncol=2)
    
    # plot num_features (different order of magnitude) on the secundary axis
    a0_sec = a[0].twinx()
    a0_sec.set_ylabel(r'$K$')
    a0_sec.plot(num_features,color='yellow')

    a0_sec.set_ylim(bottom,max(num_features)*1.1)

    a[1].set_xlabel('iteration')
    a[1].set_ylabel('Relative Change')
    a[1].set_title('Deviation from\ninitial condition')
    a[1].plot(np.array(min_rmse_te)/min_rmse_te[0])
    a[1].plot(np.array(lambda_user)/lambda_user[0])
    a[1].plot(np.array(lambda_item)/lambda_item[0])
    a[1].plot(np.array(gamma)/gamma[0])
    a[1].plot(np.array(gamma_dec_step_size)/gamma_dec_step_size[0])
    a[1].plot(np.array(num_features)/num_features[0],color='yellow')
    
    f.subplots_adjust(bottom=0.3, wspace=0.33)

    f.legend( loc='lower center', ncol=7)
    f.suptitle('Parameter Tuning GD')
    plt.tight_layout(rect=(0,0.1,1,0.95))

    
