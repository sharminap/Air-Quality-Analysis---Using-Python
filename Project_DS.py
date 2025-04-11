import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"D:\Data Science with py project\Air_Quality_Measures.csv")
print(df.head)
print(df.shape)
print(df.dtypes)
print(df.info())
print("Statical Functions: \n",df.describe().round(2))
print("Missing values: \n",df.isnull().sum())

duplicated_rows = df[df.duplicated()]
print(f"Noumber of duplicated rows: {duplicated_rows.shape[0]}")

#obj1.
top_states = df.groupby("StateName")["Value"].mean().nlargest(10).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(data=top_states, x="Value", y="StateName", hue="StateName")
plt.title("Top 10 Most Polluted States (Average Pollution Levels)", fontsize=16, fontweight='bold')
plt.xlabel("State Name", fontsize=14)
plt.ylabel("Average Pollution Level", fontsize=14)
plt.xticks(rotation=45)
plt.show()

# obj2.
yearly_trend = df.groupby("ReportYear")["Value"].mean().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(yearly_trend["ReportYear"], yearly_trend["Value"], marker='o', linestyle='-', color='b', label="Air Quality")
plt.title("Year-wise Trend of Air Quality", fontsize=16, fontweight='bold')
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Air Quality Value", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

#obj3.
top_countries = df.groupby("CountyName")["Value"].mean().nlargest(10).index
df_top = df[df["CountyName"].isin(top_countries)]
yearly_country = df_top.groupby(["CountyName", "ReportYear"])["Value"].mean().reset_index()
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_country, x="ReportYear", y="Value", hue="CountyName", linewidth=2)
plt.title("Year-wise Air Quality Trend for Top 10 Countries", fontsize=16, fontweight="bold")
plt.xlabel("Report Year", fontsize=14)
plt.ylabel("Average Pollution Level", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="CountyName", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


#obj4.
selected_year = 2010
country_name = "United States" 
df.columns = df.columns.str.strip()
year_data = df[df["ReportYear"] == selected_year]
state_pollution = (
    year_data.groupby("StateName")["Value"].sum().reset_index()
    .sort_values(by="Value", ascending=False).head(10)
)
plt.figure(figsize=(8, 8))
plt.pie(state_pollution["Value"],
        labels=state_pollution["StateName"],
        autopct="%1.1f%%",
        startangle=140)

plt.axis("equal")
plt.title(f"Top 10 States by Pollution in {selected_year} for {country_name}", 
          fontsize=16, fontweight="bold")
plt.show()


#obj 5
import warnings
warnings.filterwarnings("ignore")
pivot_df = df.pivot_table(index=["ReportYear"], 
                          columns="MeasureName", 
                          values="Value", 
                          aggfunc="mean")
pivot_df = pivot_df.dropna(how='all')
corr_matrix = pivot_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Correlation Heatmap of Air Quality Measures")
plt.tight_layout()
plt.show()

#obj 6:
df_clean = df.dropna(subset=['StateName', 'Value'])
top_states = df_clean.groupby('StateName')['Value'].mean().nlargest(10).index
df_top = df_clean[df_clean['StateName'].isin(top_states)]
plt.figure(figsize=(14, 6))
palette = sns.color_palette("Set2", n_colors=10)
sns.boxplot(data=df_top, x='StateName', y='Value', palette=palette, width=0.5)  # Adjust box width here
plt.yscale('log')  
plt.title('Box Plot of Pollutant Values (Log Scale) in Top 10 States', fontsize=14)
plt.xlabel('State', fontsize=12)
plt.ylabel('Pollutant Value (log scale)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#obj 7:
df_filtered = df[df["MeasureName"].str.contains(
    "Number of days with maximum 8-hour average ozone concentration over the National Ambient Air Quality Standard|"
    "Percent of days with PM2.5 levels over the National Ambient Air Quality Standard \(NAAQS\)",
    case=False,
    na=False
)]
plt.figure(figsize=(12, 6))
sns.histplot(data=df_filtered, x="Value", hue="MeasureName", bins=30, kde=True, palette="Set2", edgecolor="black")

plt.title("Distribution of Ozone and PM2.5 Days Exceeding NAAQS")
plt.xlabel("Pollution Value")
plt.ylabel("Frequency")
plt.legend(title="Measure Name", loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()

#obj 8:hypothesis
from statsmodels.stats.weightstats import ztest
import pandas as pd
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value", "ReportYear"])
sample_year = 2010
values_year = df[df["ReportYear"] == sample_year]["Value"]
values_all = df["Value"]
if len(values_year) > 0 and values_year.std() > 0:
    z_score, p_value = ztest(values_year, value=values_all.mean())
    print(f"Z-score: {z_score:.3f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Statistically significant difference (Reject H₀)")
    else:
        print("No statistically significant difference (Fail to reject H₀)")
else:
    print("Not enough valid data or zero variance to perform Z-test.")
