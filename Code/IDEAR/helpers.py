import collections

from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from statsmodels.graphics.mosaicplot import mosaic
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stats
from functools import partial
import IPython
import ipywidgets
from ipywidgets import widgets
from ipywidgets import interact, interactive,fixed
import operator
from IPython.display import Javascript, display,HTML
from ipywidgets import widgets, VBox
import seaborn as sns
from collections import OrderedDict
import yaml
import warnings
import getpass
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import collections

from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
from statsmodels.graphics.mosaicplot import mosaic
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stats
from functools import partial
import IPython
import ipywidgets
from ipywidgets import widgets
from ipywidgets import interact, interactive,fixed
import operator
from IPython.display import Javascript, display,HTML, clear_output
from ipywidgets import widgets, VBox, Output, Layout
import seaborn as sns
from collections import OrderedDict
import yaml
import copy
import warnings
import getpass
import sys

class ConfUtility():
    @staticmethod
    def parse_yaml(input_file):
        import yaml
        yaml_dict = {}
        with open (input_file,'r') as fin:
            try:
                yaml_dict = yaml.load(fin)
            except Exception as ex:
                print (ex)
        return yaml_dict

    @staticmethod
    def dict_to_htmllist(dc, include_list=None):
        dc2 = {}
        output_formatting = {'Target':'Target variable is ','CategoricalColumns':'Categorical Columns are ',
                           'NumericalColumns':'Numerical Columns are '}
        for each in dc.keys():
            if not include_list or each in include_list:
                if isinstance(dc[each],  collections.Iterable) and not isinstance(dc[each], str):
                    dc2[each] = ', \n'.join(val for val in dc[each])
                else:
                    dc2[each] = dc[each]
        html_list = "<ul>{}</ul>"
        html_list_entry = "<li>{}</li>"
        output3 = ''

        for each in set(include_list)|set(dc2.keys()):
            output3 += html_list_entry.format(output_formatting[each]+dc2[each])
        html_list = html_list.format(output3)
        return HTML(html_list)

        return df
    @staticmethod
    def chi_square_test(df, target, explanatory):

        import pandas as pd
        from scipy.stats import chi2_contingency
        import numpy as np
        import seaborn as sb
        import statistics

        # Create contingency table
        tbl = df.groupby([target, explanatory]).size().unstack(target).fillna(0)

        # Perform chi-square test
        chi2, p, dof, expected = chi2_contingency(tbl)

        # Verify that chi-square assumptions haven't been broken
        failed_assumptions = any([expect < 5 for expect in expected.flatten()])

        return p, failed_assumptions
    @staticmethod
    def bootstrap_categorical(df, target, explanatory, test, n, remove_test_violations):

        import pandas as pd
        from scipy.stats import chi2_contingency
        import numpy as np
        import seaborn as sb
        import statistics
        
        # Do some bootstrapping!
        results = []
        nrow_df = df.shape[0]

        for i in range(0,n):
            df_boot = df.sample(nrow_df, replace = True)
            result = ConfUtility.chi_square_test(df_boot, target, explanatory)
            results.append(result)

        # Tell the user whether or not any assumptions have been broken
        results = pd.DataFrame(results)

        results.columns = ['P','ViolatesAssumptions']
        n_violations = sum(results['ViolatesAssumptions'].values)
        if n_violations != 0:
            print("Warning: for variable " + explanatory + ": " + str(n_violations) + " out of " + str(n) + " trials violated assumptions.")                   
        else:
            print("For variable " + explanatory + ": " + str(n_violations) + " out of " + str(n) + " trials violated assumptions.")                               
            
        # Filter out violations if the user has requested it
        if remove_test_violations:
            results = results.loc[[not x for x in results['ViolatesAssumptions'].values]]
        
        # Extract values
        results = results['P'].values
        
        return results

    @staticmethod
    def bootstrap_many_categorical(df, target, explanatory_variables, test, n, remove_test_violations):

        import pandas as pd
        from scipy.stats import chi2_contingency
        import numpy as np
        import seaborn as sb
        import statistics
        
        # Different bootstrap tests for different variables
        results = {}
        for explanatory in explanatory_variables:
            result = ConfUtility.bootstrap_categorical(df, target, explanatory, test, n, remove_test_violations)
            results[explanatory] = result

        return results
    @staticmethod 
    def plot_categorical_bootstraps(results):
        
        import pandas as pd
        from scipy.stats import chi2_contingency
        import numpy as np
        import seaborn as sb
        import statistics
        
        # Initialize empty dataframe
        x_names = []
        y_means = []
        y_mins = []
        y_maxes = []
        x_significance = []

        # Iterate through results and calculate summary stats
        for key in results.keys():
            result = results[key]
            if result.shape[0] == 0:
                continue
            else:
                y_mean = statistics.mean(result)
                y_min = min(result)
                y_max = max(result)
                x_significant = y_max < 0.05

                x_names.append(key)
                y_means.append(y_mean)
                y_mins.append(y_min)
                y_maxes.append(y_max)
                x_significance.append(x_significant)
        data = pd.DataFrame({"Explanatory_Variable":x_names, "Y_Mean":y_means, "Y_Min":y_mins, "Y_Max":y_maxes, "Significant":x_significance})
        if data.shape[0] != 0:
            data.sort_values("Y_Max", ascending=True)
            sb.set_style("whitegrid")
            ax = sb.barplot(x="Y_Max", y="Explanatory_Variable", data=data)
    @staticmethod 
    def bootstrap_numerical(df, target, explanatory, n):

        from sklearn.feature_selection import f_classif
        import numpy as np

        # Do some bootstrapping!
        results = []
        nrow_df = df.shape[0]
        df = df.fillna(0)
        for i in range(0,n):
            df_boot = df.sample(nrow_df, replace = True)
            F, pval = f_classif(df_boot[[explanatory]], df_boot[target])
            results.append(pval[0])
        return results
    @staticmethod 
    def bootstrap_many_numerical(df, target, explanatory_variables, n):

        # Different bootstrap tests for different variables
        results = {}
        for explanatory in explanatory_variables:
            result = ConfUtility.bootstrap_numerical(df, target, explanatory, n)
            results[explanatory] = result

        return results

    @staticmethod 
    def plot_numerical_bootstraps(results, explanatory_variables):

        import seaborn as sb
        import pandas as pd

        # Results to dataframe
        data = pd.DataFrame(results)

        # Melt results
        data['FakeID'] = list(data.index)
        data = pd.melt(data, id_vars = ['FakeID'], value_vars = explanatory_variables)
        ax = sb.boxplot(x= 'variable', y = 'value', data=data)
class InteractionAnalytics():
    @staticmethod
    def rank_associations(df, conf_dict, col1, col2, col3):
        try:
            col2 = int(col2)
            col3 = int(col3)
        except:
            pass

        # Passed Variable is Numerical
        if (col1 in conf_dict['NumericalColumns']) :
            fig,(ax1,ax2) = plt.subplots(2, 1, figsize = (15, 15))
            plt.subplots_adjust(hspace = 0.5)
            if len(conf_dict['NumericalColumns'])>1:

                # Interaction with numerical variables
                df2 = df[conf_dict['NumericalColumns']]
                corrdf = df2.corr()
                corrdf = abs(corrdf)
                corrdf2 = corrdf[corrdf.index==col1].reset_index()[[each for each in corrdf.columns \
                                                      if col1 not in each]].unstack().sort_values(kind="quicksort",
                                                                                                  ascending=False).head(col2)
                corrdf2 = corrdf2.reset_index()
                corrdf2.columns = ['level0','level1','rsq']
                corrdf2.set_index('level0', inplace=True)
                corrdf2[['rsq']].plot(kind='bar', ax=ax1)
                ax1.legend().set_visible(False)
                ax1.set_xlabel('Absolute Correlation')
                ax1.set_title('Top {} Associated Numeric Variables'.format(str(col2)))
                # Interaction with categorical variables
                etasquared_dict = {}
            if len(conf_dict['CategoricalColumns']) >= 1:
                for each in conf_dict['CategoricalColumns']:
                    mod = ols('{} ~ C({})'.format(col1, each),data=df[[col1,each]],missing='drop').fit()
                    aov_table = sm.stats.anova_lm(mod, typ=1)
                    esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
                    etasquared_dict[each] = esq_sm

                topk_esq = pd.DataFrame.from_dict(etasquared_dict, orient='index').unstack().sort_values(\
                    kind = 'quicksort', ascending=False).head(col3).reset_index().set_index('level_1')
                topk_esq.columns = ['level_0', 'EtaSquared']
                topk_esq[['EtaSquared']].plot(kind='bar',ax=ax2)
                ax2.legend().set_visible(False)
                ax2.set_xlabel('Eta-squared values')
                ax2.set_title('Top {}  Associated Categoric Variables'.format(str(col2)))
        # Passed Variable is Categorical
        else:
            #Interaction with numerical variables
            fig,(ax1,ax2) = plt.subplots(2, 1, figsize = (15, 15))
            plt.subplots_adjust(hspace = 0.5)
            if len(conf_dict['NumericalColumns']) >= 1:
                etasquared_dict = {}
                for each in conf_dict['NumericalColumns']:
                    mod = ols('{} ~ C({})'.format(each, col1), data = df[[col1,each]]).fit()
                    aov_table = sm.stats.anova_lm(mod, typ=1)
                    esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
                    etasquared_dict[each] = esq_sm

                topk_esq = pd.DataFrame.from_dict(etasquared_dict, orient='index').unstack().sort_values(\
                    kind = 'quicksort', ascending=False).head(col2).reset_index().set_index('level_1')
                topk_esq.columns = ['level_0','EtaSquared']
                topk_esq[['EtaSquared']].plot(kind='bar',ax=ax1)
                ax1.legend().set_visible(False)
                ax1.set_xlabel('Eta-squared values')
                ax1.set_title('Top {} Associated Numeric Variables'.format(str(col2)))

            # Interaction with categorical variables
            cramer_dict = {}
            if len(conf_dict['CategoricalColumns'])>1:
                for each in conf_dict['CategoricalColumns']:
                    if each !=col1:
                        tbl = pd.crosstab(df[col1], df[each])
                        chisq = stats.chi2_contingency(tbl, correction=False)[0]
                        try:
                            cramer = np.sqrt(chisq/sum(tbl))
                        except:
                            cramer = np.sqrt(chisq/tbl.as_matrix().sum())
                            pass
                        cramer_dict[each] = cramer

                topk_cramer = pd.DataFrame.from_dict(cramer_dict, orient='index').unstack().sort_values(\
                    kind = 'quicksort', ascending=False).head(col3).reset_index().set_index('level_1')
                topk_cramer.columns = ['level_0','CramersV']
                topk_cramer[['CramersV']].plot(kind='bar',ax=ax2)
                ax2.legend().set_visible(False)
                ax2.set_xlabel("Cramer's V")
                ax2.set_title('Top {} Associated Categoric Variables'.format(str(col2)))

    @staticmethod
    def NoLabels(x):
        return ''

    @staticmethod
    def categorical_relations(df, col1, col2):
        if col1 != col2:
            df2 = df[(df[col1].isin(df[col1].value_counts().head(10).index.tolist()))&(df[col2].isin(df[col2].value_counts().head(10).index.tolist())) ]
            df3 = pd.crosstab(df2[col1], df2[col2])
            df3 = df3+1e-8
        else:
            df3 = pd.DataFrame(df[col1].value_counts())[:10]
        fig,ax = plt.subplots()
        fig,rects = mosaic(df3.unstack(),ax=ax, statistic=False, labelizer=InteractionAnalytics.NoLabels, label_rotation=30)
        ax.set_ylabel(col1)
        ax.set_xlabel(col2)
        ax.set_title('{} vs {}'.format(col1, col2) )

    @staticmethod
    def numerical_relations(df, col1, col2):
        from statsmodels.nonparametric.smoothers_lowess import lowess
        x = df[col2]
        y = df[col1]
        f, ax = plt.subplots(1)

        # lowess
        ax.scatter(x, y, c='g', s=6)
        lowess_results = lowess(y, x)#[:,1]
        xs = lowess_results[:, 0]
        ys = lowess_results[:, 1]
        ax.plot(xs,ys,'red',linewidth=1)

        #ols
        fit = np.polyfit(x, y, 1)
        fit1d = np.poly1d(fit)
        ax.plot(x, fit1d(x), '--b')
        ax.set_xlabel(col2)
        ax.set_ylabel(col1)
        corr = round(scipy.stats.pearsonr(x, y)[0], 6)
        ax.set_title('{} vs {}, Correlation {}'.format(col1, col2, corr))

    @staticmethod
    def numerical_correlation(df, conf_dict, col1):
        from matplotlib.pyplot import quiver, colorbar, clim,  matshow
        df2 = df[conf_dict['NumericalColumns']].corr(method=col1)
        col_names = list(df[conf_dict['NumericalColumns']].columns)
        fig,ax = plt.subplots(1, 1)
        m = ax.matshow(df2, cmap=matplotlib.pyplot.cm.coolwarm)
        ax.grid(b=False)
        fig.colorbar(m)
        ax.set_xticklabels([' '] + col_names)
        ax.set_yticklabels([' '] + col_names)

    @staticmethod
    def numerical_pca(df, conf_dict, col1, col2, col3):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        num_numeric = len(conf_dict['NumericalColumns'])
        num_pca = num_numeric
        xticklabels = ['']
        for i in range(1,num_pca+1):
            xticklabels+=['Comp'+str(i)]
            xticklabels+=['']
        df2 = df[conf_dict['NumericalColumns']]
        X = StandardScaler().fit_transform(df2.values)
        pca = PCA(n_components=num_pca)
        pca.fit(X)

        # Generate bar graph of % explained
        fig, (ax1,ax2) = plt.subplots(1, 2)
        ax1.bar(np.arange(1,(num_numeric+1),1),pca.explained_variance_ratio_ )
        ax1.set_ylabel('% Variance Explained')
        ax1.set_xticklabels(xticklabels)

        x_pca_index = int(col2) - 1
        y_pca_index = int(col3) - 1
        Y_pca = pd.DataFrame(pca.fit_transform(X))
        Y_pca_labels = []
        for i in range(1,num_pca+1):
            Y_pca_labels.append('PC'+str(i))
        Y_pca.columns = Y_pca_labels
        Y_pca[col1] = df[col1]
        colors_dict = {}
        colors_list = ['r', 'y', 'c', 'y', 'k']
        j = 0
        for i in np.unique(df[col1]):
            colors_dict[i] = colors_list[j]
            j += 1
            if j == len(colors_list):
                j = 0
        colordf = pd.DataFrame.from_dict(colors_dict, orient='index').reset_index()
        colordf.columns = [col1, 'color']
        merged_df = pd.merge(colordf,Y_pca)
        grouped_df = merged_df.groupby(col1)


        # Generate PCA scatterplot
        for name, group in grouped_df:
            ax2.scatter(
               group[Y_pca.columns[x_pca_index]], group[Y_pca.columns[y_pca_index]],label=name,
               c=group['color'],
               marker='o',
               s=6)
        ax2.set_xlabel(Y_pca.columns[x_pca_index])
        ax2.set_ylabel(Y_pca.columns[y_pca_index])
        ax2.legend(title=col1, fontsize=14)

    @staticmethod
    def numerical_pca_egv(df, conf_dict, col1, col2, col3):
        # Do the PCA.
        n_components = len(conf_dict['NumericalColumns'])
        df2 = df[conf_dict['NumericalColumns']]

        scaler = StandardScaler()
        scaler.fit(df2)
        df2 = scaler.transform(df2)
        df2 = pd.DataFrame(df2, columns = conf_dict['NumericalColumns'])

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(df2)

        # Append the principle components for each entry to the dataframe
        for i in range(0, n_components):
            df2['PC' + str(i + 1)] = reduced[:, i]

        #display(df2.head())
        if col1 not in conf_dict['NumericalColumns']:
            df.reset_index(drop=True, inplace=True)
            df2.reset_index(drop=True, inplace=True)
            df2[col1] = df[col1]

        # Show the points in terms of the first two PCs
        g = sns.lmplot(('PC' + str(col2)),
                       ('PC' + str(col3)),
                       hue=col1,
                       data=df2,
                       fit_reg=False,
                       scatter=True,
                       size=7)
        plt.show()

        # Plot a variable factor map for the first two dimensions.
        (fig, ax) = plt.subplots(figsize=(8, 8))
        for i in range(0, len(pca.components_)):
            ax.arrow(0,
                     0,  # Start the arrow at the origin
                     pca.components_[int(col2) - 1, i],  #0 for PC1
                     pca.components_[int(col3) - 1, i],  #1 for PC2
                     head_width=0.05,
                     head_length=0.08)

            plt.text(pca.components_[int(col2) - 1, i] + 0.05,
                     pca.components_[int(col3) - 1, i] + 0.05,
                     df2.columns.values[i])

        an = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
        plt.axis('equal')
        ax.set_title('Variable factor compass')
        plt.show()

        # Do a scree plot
        ind = np.arange(0, n_components)
        (fig, ax) = plt.subplots(figsize=(8, 6))
        sns.pointplot(x=ind, y=pca.explained_variance_ratio_)
        ax.set_title('Scree plot')
        ax.set_xticks(ind)
        ax.set_xticklabels(ind)
        ax.set_xlabel('Component Number')
        ax.set_ylabel('Explained Variance')
        plt.show()



    @staticmethod
    def nc_relation(df, conf_dict, col1, col2, col3=None):
        fig,ax = plt.subplots()
        f = df[[col1,col2]].boxplot(by=col2, ax=ax)
        mod = ols('{} ~ {}'.format(col1, col2), data=df[[col1, col2]]).fit()
        aov_table = sm.stats.anova_lm(mod, typ=1)
        p_val = round(aov_table['PR(>F)'][0], 6)
        status = 'Passed'
        color = 'blue'
        if p_val < 0.05:
            status = 'Rejected'
            color = 'red'
        fig.suptitle('ho {} (p_value = {})'.format( status, p_val), color=color, fontsize=10)

    @staticmethod
    def pca_3d(df, conf_dict, col1, col2,  col3=None):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from mpl_toolkits.mplot3d import Axes3D
        df2 = df[conf_dict['NumericalColumns']]
        X = StandardScaler().fit_transform(df2.values)
        pca = PCA(n_components=4)
        pca.fit(X)
        fig = plt.figure(figsize = (12, 12))
        ax = fig.gca(projection='3d')
        ax.view_init(elev=10, azim=int(col2))
        Y_pca = pd.DataFrame(pca.fit_transform(X))
        Y_pca.columns = ['PC1','PC2','PC3','PC4']
        Y_pca[col1] = df[col1]
        colors_dict = {}
        colors_list = ['r', 'y', 'c', 'y', 'k']
        j = 0
        for i in np.unique(df[col1]):
            colors_dict[i] = colors_list[j]
            j += 1
            if j == len(colors_list):
                j = 0
        colordf = pd.DataFrame.from_dict(colors_dict, orient='index').reset_index()
        colordf.columns = [col1,'color']
        merged_df = pd.merge(colordf,Y_pca)
        grouped_df = merged_df.groupby(col1)
        for name, group in grouped_df:
            ax.scatter(
               group['PC1'], group['PC2'], group['PC3'], label=name,
               c = group['color'],
               marker = 'o',
               s=6)
        ax.set_xlabel('PC1', labelpad=18)
        ax.set_ylabel('PC2', labelpad=18)
        ax.set_zlabel('PC3', labelpad=18)
        ax.legend(title=col1, fontsize=10)

    @staticmethod
    def pca_3d_new(df, conf_dict, col1, col2, col3, col4, col5):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from mpl_toolkits.mplot3d import Axes3D
        df2 = df[conf_dict['NumericalColumns']]
        X = StandardScaler().fit_transform(df2.values)
        num_numeric = len(conf_dict['NumericalColumns'])
        pca = PCA(n_components=num_numeric)
        pca.fit(X)
        fig = plt.figure(figsize = (15, 20))
        ax = fig.gca(projection='3d')
        ax.view_init(elev=10, azim=int(col5))
        Y_pca = pd.DataFrame(pca.fit_transform(X))
        Y_pca_names = []
        for i in range(1, num_numeric+1):
            Y_pca_names.append('PC'+str(i))
        Y_pca.columns = Y_pca_names
        Y_pca[col1] = df[col1]
        colors_dict = {}
        colors_list = ['r', 'y', 'c', 'y', 'k']
        j = 0
        for i in np.unique(df[col1]):
            colors_dict[i] = colors_list[j]
            j += 1
            if j == len(colors_list):
                j = 0
        colordf = pd.DataFrame.from_dict(colors_dict, orient='index').reset_index()
        colordf.columns = [col1,'color']
        merged_df = pd.merge(colordf,Y_pca)
        grouped_df = merged_df.groupby(col1)
        for name, group in grouped_df:
            ax.scatter(
               group[Y_pca_names[int(col2)-1]], group[Y_pca_names[int(col3)-1]], group[Y_pca_names[int(col4)-1]], label=name,
               c = group['color'],
               marker = 'o',
               s=6)
        ax.set_xlabel(Y_pca_names[int(col2)-1], labelpad=18)
        ax.set_ylabel(Y_pca_names[int(col3)-1], labelpad=18)
        ax.set_zlabel(Y_pca_names[int(col4)-1], labelpad=18)
        ax.legend(title=col1, fontsize=10)

    @staticmethod
    def nnc_relation(df, conf_dict, col1, col2, col3):
        import itertools
        markers = ['x', 'o', '^']
        color = itertools.cycle(['r', 'y', 'c', 'y', 'k'])
        groups = df[[col1, col2, col3]].groupby(col3)

        # Plot
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.margins(0.05)

        for (name, group), marker in zip(groups, itertools.cycle(markers)):
            ax.plot(group[col1], group[col2], marker='o', linestyle='', ms=4, label=name)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.legend(numpoints=1, loc='best', title=col3)
    @staticmethod
    def bootstrap_all_categorical(df, target, conf_dict, n, remove_test_violations = False):
        
        # Grab categorical columns
        cols_list = conf_dict['CategoricalColumns']
        cols_list = [x for x in cols_list if x != target]
        
        # Perform bootstrapping
        results = ConfUtility.bootstrap_many_categorical(df, target, cols_list, ConfUtility.chi_square_test, n, remove_test_violations)
            
        # Plot results
        ConfUtility.plot_categorical_bootstraps(results)
    @staticmethod
    def bootstrap_all_numerical(df, conf_dict, target, n):
        
        # Grab numerical columns
        cols_list = conf_dict['NumericalColumns']
        cols_list = [x for x in cols_list if x != target]
        
        # Perform bootstrapping
        results = ConfUtility.bootstrap_many_numerical(df, target, cols_list, n = n)
        
        # Plot results
        ConfUtility.plot_numerical_bootstraps(results, cols_list)        
class TargetAnalytics():
    ReportedVariables = []
    @staticmethod
    def custom_barplot(df, col1=''):
        f, (ax0,ax1) = plt.subplots(1, 2)
        df[col1].value_counts().plot(ax=ax0, kind='bar')
        ax0.set_title('Bar Plot of {}'.format(col1))
        df[col1].value_counts().plot(ax=ax1, kind='pie')
        ax1.set_title('Pie Chart of {}'.format(col1))

class NumericAnalytics():
    @staticmethod
    def shapiro_test(x):
        p_val = round(stats.shapiro(x)[1],6)
        status = 'passed'
        color = 'blue'
        if p_val < 0.05:
            status = 'failed'
            color = 'red'
        return status, color, p_val

    @staticmethod
    def custom_barplot(df, col1=''):
        fig, axes = plt.subplots(2,2, figsize=(15, 10))
        axes = axes.reshape(-1)
        df[col1].plot(ax=axes[0], kind='hist')
        axes[0].set_title('Histogram of {}'.format(col1))
        df[col1].plot(ax=axes[1], kind='kde')
        axes[1].set_title('Density Plot of {}'.format(col1))
        ax3 = plt.subplot(223)
        stats.probplot(df[col1], plot=plt)
        axes[2].set_title('QQ Plot of {}'.format(col1))
        df[col1].plot(ax=axes[3], kind='box')
        axes[3].set_title('Box Plot of {}'.format(col1))
        status, color, p_val = NumericAnalytics.shapiro_test(df[col1])
        fig.suptitle('Normality test for {} {} (p_value = {})'.format(col1, status, round(p_val, 6)), color=color, fontsize=12)
        plt.show()

class CategoricAnalytics():
    @staticmethod
    def custom_barplot(df, col1=''):
        f, (ax0,ax1) = plt.subplots(1,2)
        df[col1].value_counts().nlargest(10).plot(ax=ax0, kind='bar')
        ax0.set_xlabel(col1)
        ax0.set_title('Bar chart of {}'.format(col1))
        df[col1].value_counts().nlargest(10).plot(ax=ax1, kind='pie')
        ax1.set_title('Pie chart of {}'.format(col1))