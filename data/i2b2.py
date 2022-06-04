# Import data libraries
import os
import re
import xml.etree.ElementTree as ET
from lxml import etree
from typing import List
from a2t.data import Dataset
from a2t.tasks import BinaryTask, BinaryFeatures
from a2t.tasks import RelationClassificationTask, RelationClassificationFeatures

class i2b2TemporalRelationDataset(Dataset):
  """A class to handle the i2b2 2012 Temporal Relations dataset.

  Args:
    inputPath (str): The directory location of the dataset.
    labels (list): The list of relation labels used in the dataset.
  
  Returns:
    A list of Binary Features containing a pair of entities, 
    the context, the entity and the relation types. For example:

    [BinaryFeatures(
      X="admitted",
      Y="fever",
      context="admitted here with fever and neutropenia.",
      inst_type="OCCURRENCE:PROBLEM",
      label="BEFORE_OVERLAP"
      )]
    
    Returned values are always strings.
  """
  def __init__(self, inputPath: str, labels: List[str], *args, **kwargs) -> None:
    super().__init__(labels=labels, *args, **kwargs)

    # Load document instances
    instances = self.loadi2b2(inputPath)
    for example in instances:
      entities = example['graph']['entities']
      if len(example['graph']['relations']) > 0:
        context = ' '.join(example['tokens'])
        for rel in example['graph']['relations']:
          # Collect X entity information 
          X_type = entities[rel[0]][2]
          X = entities[rel[0]][0]
          # Collect Y entity information
          Y_type = entities[rel[1]][2]
          Y = entities[rel[1]][0]
          rel_type = rel[2]
          self.append(BinaryFeatures(
              X=X,
              Y=Y,
              context=context,
              inst_type=f"{X_type}:{Y_type}",
              label=rel_type
              ))
    print("Done")

  def loadXML(self, filePath):
    """A function to parse XML files.

    Args:
      filePath (str): The XML file location.

    Returns:
      A parsed tree from the XML file.
    """
    # Set parser with error recovery because default XMLParser does not work
    parser = etree.XMLParser(recover=True)
    # Create XML tree from file
    tree = ET.parse(filePath, parser = parser)
    return tree

  def loadi2b2(self, dirPath):
    """A function to obtain specific information from the i2b2 2012 Temporal Relations dataset.

    Args:
      dirPath (str): The directory location of the dataset.

    Returns:
      A list of dictionaries, each mapping keys to a relation of the corpus,
      and all the information relative to it. For example:

      [{"docID": "21",
        "sent_id": "21.3",
        "tokens": ["admitted", "here", "with", "fever", "and", "neutropenia", "."],
        "graph": {
          "entities": [[0, 1, "EVENT", "OCCURRENCE", 1.0], [3, 4, "EVENT", "PROBLEM"], 
          "relation": [[0, 1, "BEFORE_OVERLAP"]]
        }}]

      In the "entities" key list, Start/ End offsets are relative to the token 
      position in the tokenized sentence list. In the "relation" key list, the 
      Head/ Tail positions are relative to the entity position in the "entities" key list.

    """
    # Create list to store dictionaries with relation instances
    instanceList = []
    # Create a list of file names in the input directory
    fileList = os.listdir(dirPath)
    # Iterate over input directory files
    print("Processing files...")
    for f in fileList:
      # Create file path
      filePath = dirPath + f
      # Open and load XML file
      docm = self.loadXML(filePath)
      # Get document ID
      docID = f.replace(".xml","")
      # ROOT contains TEXT (Index 0) and TAGS (Index 1) nodes
      root = docm.getroot()
      text = root[0].text
      tags = root[1]
      # Get entities and relations separately
      entityList = []
      tlinkList = []
      for entity in tags:
        if entity.tag != "TLINK":
          entityList.append({"id":entity.attrib["id"], 
                             "start":entity.attrib["start"], 
                             "end":entity.attrib["end"], 
                             "text":entity.attrib["text"],
                             "tag":entity.tag, 
                             "type":entity.attrib["type"]})
        if entity.tag == "TLINK":
          #if "SECTIME" not in entity.attrib["id"]:
            tlinkList.append({"id":entity.attrib["id"], 
                              "fromID":entity.attrib["fromID"], 
                              "fromText":entity.attrib["fromText"],
                              "toText":entity.attrib["toText"],
                              "toID":entity.attrib["toID"],
                              "type":entity.attrib["type"]})
      # Get sentences
      sentList = []
      for sent in re.finditer("[^\r\n]+", text):
        sentList.append({"text":sent.group(0),
                         "start":sent.start(),
                         "end":sent.end()})
      # Iterate over relations
      for rel in tlinkList:
        # Get relation ID
        relID = f"{docID}-{rel['id']}"
        # Get head and tail sentence positions
        entHeadTail = []
        for entity in entityList:
          if rel['fromID'] == entity['id']:
            entHeadTail.append(int(entity['start']))
            headList = [entity["text"], entity["tag"], entity["type"]]
          elif rel['toID'] == entity['id']:
            entHeadTail.append(int(entity['end']))
            tailList = [entity["text"], entity["tag"], entity["type"]]
          else:
            continue
        sentHeadTail = []
        for index, sent in enumerate(sentList):
          if sent["start"] <= min(entHeadTail) <= sent["end"]:
            sentHeadTail.append(index)
          elif sent["start"] <= max(entHeadTail) <= sent["end"]:
            sentHeadTail.append(index)
          else:
            continue
        # Set context threshold
        #print(f"{max(sentHeadTail)+1} - {min(sentHeadTail)}= {max(sentHeadTail)+1 - min(sentHeadTail)} ")
        if max(sentHeadTail)+1 - min(sentHeadTail) >= 10:
          pass
        else:
          # Get context
          context = []
          for sent in sentList[min(sentHeadTail):max(sentHeadTail)+1]:
            context.append(sent["text"])
          sentence = " ".join(context)
          tokens = sentence.split()
          # Get entities
          entities = []
          for ent in [headList, tailList]:
            entities.append([ent[0],
                            ent[1],
                            ent[2],
                            1.0])
          # Get relations
          relations = [[0, 1, rel["type"]]]

          #print(relID, entHeadTail, sentHeadTail, headList[0],"-", tailList[0],":", sentence)
      
          instanceList.append({ # Each 'instance' dictionary contains only one relation and two entities
              'docID' : docID, # Document ID where the sentence comes from
              'relID': relID, # Sentence ID 
              'tokens': tokens, # List with the tokenized sentence
              'graph': {
                  'entities': entities, # For each entity: [Start Token Position, End Token Position, Entity Type, Entity Subtype, 1.0]
                  'relations': relations, # For each relation: [Head Appearance Position, Tail Appearance Position, Relation Type]
                  }})
    print("Creating Binary Features...")
    return instanceList
