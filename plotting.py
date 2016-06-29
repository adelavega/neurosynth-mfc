#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


nine_colors = [(0.89411765336990356, 0.10196078568696976, 0.10980392247438431),
(0.65845446095747107, 0.34122261685483596, 0.1707958535236471),
(1.0, 0.50591311045721465, 0.0031372549487095253),
(0.21602460800432691, 0.49487120380588606, 0.71987698697576341),
(0.30426760128900115, 0.68329106055054012, 0.29293349969620797),
(0.400002384185791, 0.4000002384185791, 0.40000002384185791), 
(0.60083047361934883, 0.30814303335021526, 0.63169552298153153),
(0.99850826852461868, 0.60846600392285513, 0.8492888871361229),
(0.99315647868549117, 0.9870049982678657, 0.19915417450315812)
 ]

def plot_polar(data, n_top=3, selection='top', overplot=False, labels=None,
               palette='husl', metric='correlation', label_size=26, threshold=None, max_val=None, return_labels=False,
               alpha_level=1, legend=False, error_bars=None, reorder=False, method='weighted'):

    n_panels = data.shape[1]

    if labels is None:

        if selection == 'top':
            labels = []
            for i in range(n_panels):
                labels.extend(data.iloc[:, i].sort_values(ascending=False) \
                    .index[:n_top])
            labels = np.unique(labels)
        elif selection == 'std':
            labels = data.T.std().sort_values(ascending=False).index[:n_top]

        data = data.loc[labels,:]
    
    else:
        data = data.loc[labels,:]

    if error_bars is not None:
        error_bars = error_bars.loc[labels,:]

    if reorder is True:
        # Use hierarchical clustering to order
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, leaves_list
        dists = pdist(data, metric=metric)
        pairs = linkage(dists, method=method)
        pairs[pairs < 0] = 0
        order = leaves_list(pairs)
        data = data.iloc[order,:]

        if error_bars is not None:
            error_bars = error_bars.iloc[order,:]

        labels = [labels[i] for i in order]


    theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
    
    ## Add first
    theta = np.concatenate([theta, [theta[0]]])
    if overplot:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
        fig.set_size_inches(10, 10)
    else:
        fig, axes = plt.subplots(n_panels, 1, sharex=False, sharey=False,
                             subplot_kw=dict(polar=True))
        fig.set_size_inches((6, 6 * n_panels))

    # from IPython.core.debugger import Tracer; Tracer()()
        
    from seaborn import color_palette
    colors = color_palette(palette, n_panels)
    for i in range(n_panels):
        if overplot:
            alpha = 0.025
        else:
            ax = axes[i]
            alpha = 0.8

        if max_val is None:
            if error_bars is not None:
                max_val = data.values.max() + error_bars.values.max() + data.values.max() * .02
            else:
                max_val = data.values.max()
        
        ax.set_ylim(data.values.min(), max_val)
        
        d = data.iloc[:,i].values
        d = np.concatenate([d, [d[0]]])
        name = data.columns[i]

        if error_bars is not None:
            e = error_bars.iloc[:,i].values
            e = np.concatenate([e, [e[0]]])
        else:
            e = None


        if error_bars is not None:
            # ax.errorbar(theta, d, yerr=e, capsize=0, color='black', elinewidth = 3, linewidth=4.5)
            ax.errorbar(theta, d, yerr=e, capsize=0, color=colors[i], elinewidth = 3, linewidth=0)
        else:
            ax.plot(theta, d, alpha=alpha_level - 0.1, color=colors[i], linewidth=8, label=name)
            ax.fill(theta, d, ec='k', alpha=alpha, color=colors[i], linewidth=8)

        ax.set_xticks(theta)
        ax.set_rlabel_position(11.12)
        ax.set_xticklabels(labels, fontsize=label_size)
        [lab.set_fontsize(22) for lab in ax.get_yticklabels()]

    
    if threshold is not None:
        theta = np.linspace(0.0, 2 * np.pi, 999, endpoint=False)
        theta = np.concatenate([theta, [theta[0]]])
        d = np.array([threshold] * 1000)
        ax.plot(theta, d, alpha=1, color='black', linewidth=2, linestyle='--')

    if legend is True:
        ax.legend(bbox_to_anchor=(1.15, 1.1))

    plt.tight_layout()

    return labels, data

def plot_clf_polar(clf, cmap=None, n_topics=60, 
    mask=None, feature_names=None, region_names=None, data_value='odds_ratio', **kwargs):
    import pandas as pd
    import seaborn as sns

    if feature_names is None:
        feature_names = clf.feature_names

    o_fi = pd.DataFrame(getattr(clf, data_value), columns=feature_names)

    # Melt feature importances, and add top_words for each feeature
    o_fi['region'] = range(1, o_fi.shape[0] + 1)
    o_fis_melt = pd.melt(o_fi, var_name='topic', value_name='importance', id_vars=['region'])
    
    if mask is not None:
        o_fis_melt = o_fis_melt[o_fis_melt.region.isin(mask)]
    
    pplot = pd.pivot_table(o_fis_melt, values='importance', index='topic', columns=['region'])

    if cmap is None:
        cmap = sns.color_palette('Set1', clf.odds_ratio.shape[0])
    if mask is not None:
        cmap = [n[0] for n in sorted(zip(np.array(cmap)[np.array(mask)-1], mask), key=lambda tup: tup[1])]

    return plot_polar(pplot, overplot=True, palette=cmap, **kwargs)


def make_thresholded_slices(regions, colors, display_mode='z', overplot=True, binarize=True, **kwargs):
    """ Plots on axial slices numerous images
    regions: Nibabel images
    colors: List of colors (rgb tuples)
    overplot: Overlay images?
    binarize: Binarize images or plot full stat maps
    """             

    from matplotlib.colors import LinearSegmentedColormap
    from nilearn import plotting as niplt
    
    if binarize:
        for reg in regions:
             reg.get_data()[reg.get_data().nonzero()] = 1
                                   
    for i, reg in enumerate(regions):
        reg_color = LinearSegmentedColormap.from_list('reg1', [colors[i], colors[i]])
        if i == 0:
            plot = niplt.plot_stat_map(reg, draw_cross=False,  display_mode=display_mode, cmap = reg_color, alpha=0.9, colorbar=False, **kwargs)
        else:
            if overplot:
                plot.add_overlay(reg, cmap = reg_color, alpha=.72)
            else:
                plt.plot_stat_map(reg, draw_cross=False,  display_mode=display_mode, cmap = reg_color, colorbar=False, **kwargs)
    
    return plot
