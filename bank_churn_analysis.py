"""
Banking Customer Churn & Transaction Analysis
Author: Patrick Emad Seleem Samwael
Tools: Python, NumPy, Pandas, Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads banking dataset, handles missing values, and removes outliers using IQR."""
    df = pd.read_csv(filepath)
    print(f"Initial Dataset Shape: {df.shape}")

    df['Balance'] = df['Balance'].fillna(df['Balance'].median())
    df['CreditScore'] = df['CreditScore'].fillna(df['CreditScore'].median())

    q1 = np.percentile(df['CreditScore'], 25)
    q3 = np.percentile(df['CreditScore'], 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    cleaned_df = df[(df['CreditScore'] >= lower_bound) & (df['CreditScore'] <= upper_bound)].copy()
    print(f"Cleaned Dataset Shape: {cleaned_df.shape}")
    return cleaned_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates age segments and balance-to-salary ratios for deeper analysis."""
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[17, 30, 45, 60, 100],
        labels=['18-30', '31-45', '46-60', '60+']
    )
    df['BalanceSalaryRatio'] = np.round(df['Balance'] / (df['EstimatedSalary'] + 1), 2)
    return df


def analyze_and_visualize(df: pd.DataFrame) -> None:
    """Computes key churn metrics and generates a multi-panel Matplotlib dashboard."""
    overall_churn = df['Churned'].mean() * 100
    print(f"\nOverall Customer Churn Rate: {overall_churn:.2f}%")

    churn_by_age = df.groupby('AgeGroup', observed=False)['Churned'].mean() * 100
    churn_by_activity = df.groupby('IsActiveMember')['Churned'].mean() * 100
    churn_by_products = df.groupby('NumOfProducts')['Churned'].mean() * 100

    print("\nChurn Rate by Age Group (%):")
    print(churn_by_age.round(2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Banking Customer Churn & Transaction Analysis Dashboard', fontsize=16, fontweight='bold')

    axes[0, 0].bar(churn_by_age.index.astype(str), churn_by_age.values, color='#1b365d', edgecolor='black')
    axes[0, 0].set_title('Churn Rate by Age Group (%)', fontweight='bold')
    axes[0, 0].set_xlabel('Age Group')
    axes[0, 0].set_ylabel('Churn Rate (%)')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.6)

    retained_bal = df[df['Churned'] == 0]['Balance']
    churned_bal = df[df['Churned'] == 1]['Balance']
    axes[0, 1].hist([retained_bal, churned_bal], bins=20, label=['Retained (0)', 'Churned (1)'],
                    color=['#2b7bba', '#d95f02'], edgecolor='black', alpha=0.8)
    axes[0, 1].set_title('Account Balance Distribution by Churn Status', fontweight='bold')
    axes[0, 1].set_xlabel('Account Balance ($)')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.6)

    activity_labels = ['Inactive Member', 'Active Member']
    axes[1, 0].bar(activity_labels, churn_by_activity.values, color=['#c0392b', '#27ae60'], width=0.5, edgecolor='black')
    axes[1, 0].set_title('Impact of Account Activity on Churn (%)', fontweight='bold')
    axes[1, 0].set_ylabel('Churn Rate (%)')
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.6)

    axes[1, 1].plot(churn_by_products.index.astype(str), churn_by_products.values,
                    marker='o', linewidth=2.5, markersize=8, color='#8e44ad')
    axes[1, 1].set_title('Churn Rate by Number of Bank Products', fontweight='bold')
    axes[1, 1].set_xlabel('Number of Products Held')
    axes[1, 1].set_ylabel('Churn Rate (%)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('churn_analysis_dashboard.png', dpi=300)
    print("\nDashboard saved successfully as 'churn_analysis_dashboard.png'")


if __name__ == '__main__':
    dataset_path = 'banking_customer_data.csv'
    clean_data = load_and_clean_data(dataset_path)
    enriched_data = engineer_features(clean_data)
    analyze_and_visualize(enriched_data)
