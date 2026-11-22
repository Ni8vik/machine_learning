import numpy as np
import pandas as pd

#1_three sacs  (its the name of the homework I'm not naming it random )

bag_1 = np.array(["red", "red","red","red","red",
                  "red","red","red","red","red"])

bag_2 = np.array(["blue", "blue","blue","blue","blue",
                  "red","red","red","red","red"])

bag_3 = np.array(["blue", "blue","blue","blue","blue",
                  "blue","blue","blue","blue","red"])


def entropy(data):
      sigma = 0
      unique_data_counts = np.unique_counts(data).counts
      prob = unique_data_counts/ len(data)
      return -sum(prob * np.log2(prob))


def gini(data):
      unique_data_counts = np.unique_counts(data).counts
      prob = unique_data_counts / len(data)
      return 1 - sum(prob**2)   

print(f"entropy of bag1 : {entropy(bag_1)}")
print(f"entropy of bag2 : {entropy(bag_2)}")
print(f"entropy of bag3 : {entropy(bag_3)}")
print(f"gini of bag2(contains 5/5 pearl) : {gini(bag_2)}")
print("_"*80)

#2_light box

data = [
    ["Yorkshire Terrier", 3.5, 9.5, "Small Dog"],
    ["Persian", 4.5, 5, "Cat"],
    ["Siamese", 3, 7, "Cat"],
    ["Garfield", 6, 4, "Cat"],
    ["Chartreux", 5, 6, "Cat"],
    ["Beagle", 4, 11, "Small Dog"],
    ["Papillon", 4.5, 10, "Small Dog"]
]

# Create the DataFrame
df = pd.DataFrame(data, columns=["animal", "weight", "ear", "size"])
      
gain_ear_size1 = gini(df["ear"]) - gini((df["ear"] < 8.25).values)
print(f"gini gain of ear size < 8.25 :{gain_ear_size1}")

gain_ear_size2 = gini(df["ear"]) - gini((df["ear"] < 7).values)
print(f"gini gain of ear size < 7 :{gain_ear_size2}")

gain_weight = gini(df["weight"]) - gini((df["weight"] < 3.75).values)
print(f"gini gain of weight < 3.75 :{gain_weight}")

options = ["ear size< 8.25", "ear size <7", "weight< 3.75"]
gains = [gain_ear_size1, gain_ear_size2, gain_weight]
max_index = gains.index(max(gains))

print("="*80)
print(f"the best option to pick between the three gini gains is '{options[max_index]}' with gini gain of {gains[max_index]}")
