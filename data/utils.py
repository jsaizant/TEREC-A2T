import textwrap
import random

def inspect_relation(relationType, dataset, num=10):
  """A function to display dataset examples given a relation type.

  Args:
    relationType (str): A relation label from the dataset.
    dataset (dataset): A dataset of BinaryFeatures.
    num (int): The number of output examples.
  
  Returns:
    A string displaying relation examples from the dataset.
  """
  
# create a function to show examples of given relation type
    result = [datum for datum in dataset if relationType == datum.label]
    for i in random.sample(range(0, 100), num): 
        datum = result[i]
        print(f"{''.join(datum.X)} - {datum.label} - {''.join(datum.Y)} ({datum.inst_type}):")
        print(textwrap.fill(datum.context, 80))
        print()
