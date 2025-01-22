import pandas as pd

# Path to the Excel file
tabular_data_path = "release_midas.xlsx"

# Load the Excel file
try:
    df = pd.read_excel(tabular_data_path)
except Exception as e:
    print(f"Error reading the Excel file: {e}")
    exit()

# Check if the necessary column exists
if "midas_file_name" not in df.columns:
    print("The column 'midas_file_name' is not present in the Excel file.")
    exit()

# Split the data into three parts
total_rows = len(df)
split1 = total_rows // 3
split2 = 2 * total_rows // 3

df1 = df.iloc[:split1]  # 1.05GB
df2 = df.iloc[:split2]  # 2.16GB
df3 = df  # 3.37GB (Full dataset)

# Save the splits to new Excel files
df1.to_excel("data_1.xlsx", index=False)
df2.to_excel("data_2.xlsx", index=False)
df3.to_excel("data_3.xlsx", index=False)
