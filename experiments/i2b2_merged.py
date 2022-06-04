%pip install a2t

# Import data libraries
from a2t.base import EntailmentClassifier
from a2t.tasks import RelationClassificationTask, RelationClassificationFeatures
from a2t.tasks import BinaryTask, BinaryFeatures

# Set relation types
labels = ["NO-Relation",
          "BEFORE", 
          "AFTER", 
          "OVERLAP"]

# Set test path
testPath = ""

# Load i2b2 dataset
dataset = i2b2TemporalRelationDataset(testPath, labels)

# Set relation verbalization templates
templates = {
    "BEFORE": [
               "{X} happens before {Y}.",
               "{Y} follows {X}.", 
               "{X} precedes {Y}.",
               "{X} was dispensed before {Y}.", #SPECIFIC  X = treatment/drug Y = date/symptom/sign/problem
               "{X} was carried out before {Y}", #SPECIFIC  X = test/analysis Y = date/discharge/problem
               "{X} ended on {Y}",
               "{X} was discountinued on {Y}",
               "{X} ended prior to {Y}",
               "{X} shows evidence of {Y}",
               "{X} started before {Y}",
               "{X} happens while {Y}",
               "{X} when {Y}" #SPECIFIC X = treatment/problem Y = sign/problem/date
    ], 
    "AFTER": [
              "{X} happens after {Y}.",
              "{X} follows {Y}.", 
              "{Y} precedes {X}.",
              "{X} after presenting {Y}", #SPECIFIC X = test/analysis Y = evidential/problem/sign
              "{X} was carried out after {Y}" #SPECIFIC X = test/analysis Y = test/occurrence/treatment/problem
              "{X} started when {Y}",
              "{X} proceeds with {Y}", #SPECIFIC X = clinical_dept Y = occurrence/evidential 
              "{Y} takes place after {X}" #SPECIFIC Y = treatment X = duration
    ], 
    "OVERLAP": [
                "{X} is part of {Y}", 
                "{X} happens while {Y}",
                "{X} shows evidence of {Y}",
                "No {X} when {Y}", #SPECIFIC X = problem Y = problem
                "For the treatment {X} and {Y} were combined." #SPECIFIC X = treatment Y = treatment
                "{X} is {Y}",
                "{X} when {Y}", #SPECIFIC X = problem/symptom/occurrence Y = date
                "{X} at the time of {Y}" #SPECIFIC X = problem/occurrence Y = date
                "{X} happens within {Y}",
                "{X} was used in {Y}",
                "{X} was used for {Y}",
                "{X} while being in {Y}" #SPECIFIC X = treatment/evidential Y = date/clinical_dept
    ]
}

# Load Textual Entailment model
nlp = EntailmentClassifier(
    'roberta-large-mnli',
    use_cuda=True
)

# Specify the task
task = RelationClassificationTask(
    name="i2b2 Temporal Relation Classification task",
    required_variables=["X", "Y"],
    additional_variables=["inst_type"],
    labels=labels,
    templates=templates,
    negative_label_id=0,
    multi_label=True,
    features_class=BinaryFeatures
)

# Run task
testPreds, testCoefs = nlp(task=task, 
                           features=dataset, 
                           return_labels=True, 
                           return_confidences=True, 
                           return_raw_output=True)

# Evaluate metrics
task.compute_metrics(dataset.labels, testCoefs, "default")
