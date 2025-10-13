import pandas as pd
import random

# Read the dataset 
data = pd.read_csv("Health & Lifestyle Dataset.csv", sep="\t")

# Show first five rows to confirm the data is loaded correctly
print(" Preview of the dataset (first 5 rows):")
print(data.head())

#  Count the number of features 
num_features = len(data.columns) - 1
print(f"\n Number of features in the dataset: {num_features}")

#  Generate a random chromosome
chromosome = [random.choice([0, 1]) for _ in range(num_features)]
print("\n Generated Chromosome:")
print(chromosome)

#  Save the chromosome into a text file
with open("chromosome-output.txt", "w") as f:
    f.write("Number of features: " + str(num_features) + "\n")
    f.write("Chromosome:\n")
    f.write(str(chromosome))

print("\n Chromosome has been saved successfully as 'chromosome-output.txt' ")