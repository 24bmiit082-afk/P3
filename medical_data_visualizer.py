import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from medical_examination.csv and assign it to the df variable.
df = pd.read_csv('medical_examination.csv')

# 2. Add an overweight column to the data
# Calculate BMI = weight(kg) / height(m)^2
# If BMI > 25, overweight = 1, else = 0
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalize cholesterol and glucose columns
# If value = 1, set to 0; if value > 1, set to 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    """Draw categorical plot showing counts of good/bad outcomes"""
    
    # 4-5. Create a DataFrame for the cat plot using pd.melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )
    
    # 6. Group and reformat the data to split by cardio
    # Count the occurrences of each value
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7. Create the categorical plot using seaborn
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).figure
    
    return fig


def draw_heat_map():
    """Draw heatmap of correlation matrix"""
    
    # 8. Clean the data in the df_heat variable
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # 9. Calculate the correlation matrix
    corr = df_heat.corr()
    
    # 10. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # 11. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 12. Plot the correlation matrix using seaborn heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    return fig