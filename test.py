import csv
with open('./Odata/label_dict.csv', 'r') as file:
  reader = csv.reader(file)
  label_dict = {}
  for index, line in enumerate(reader):
    label_dict[int(index)] = line[0]
print(label_dict)