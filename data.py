import os
import csv


def load_csv(file):
  dirname = os.path.dirname(file)
  images_path = []
  with open(file) as f:
    parsed = csv.reader(f, delimiter=",", quotechar="'")
    for row in parsed:
      images_path.append(os.path.join(dirname, row[0]))
  return images_path
