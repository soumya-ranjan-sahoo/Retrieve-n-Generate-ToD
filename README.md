# Retrieval-Augmented-Generative-ToDs

## This work is carried out as part of the MSc thesis at the faculty of Mathematics and Computer Science, Saarland University and Fraunhofer IAIS.

### Installation
```
conda create -n RnG-ToD python=3.7
pip install -r requirements.txt
```

## Pre-processing
### IR (indexing):
Generate indices (pkl) files by running:
```
python generate_denseidx.py
```

### Training
In multiple GPU run the following command:
```bash
python -m torch.distributed.launch train.py
```
For single GPU run:
```
python train.py
```

Training parameters:
- To be added

### Evaluation
```
python eval.py --generate runs/gpt2/incar --dataset incar --generation_params_file config/gpt2/generation_params.json --eval_dataset test  --output_file outputs/baseline_test.json
```
The program will automatically pick the best model from the directory for the evaluation.

### Refactored structure of each dialogue
```
{
      "task": "asian_oriental",
      "id": 5,
      "history": [
         "i want somewhere that serves traditional food .",
         "there are no traditional restaurants in the city . can i help you with a different type of cuisine ?",
         "how abou asian oriental food",
         "there are 5 restaurants meeting your criteria . what area and price range did you have in mind ?",
         "what is the address and phone number ?",
         "api_call asian_oriental dontcare dontcare",
         "<silence>",
         "the dojo_noodle_bar serves asian_oriental food . they are located at 40210_millers_yard_city_centre and their phone number is 01223_363471 .",
         "thank you . good bye ."
      ],
      "response": "good bye",
      "ref_ents": [],
      "kg": [
         {
            "name": "dojo_noodle_bar",
            "address": "40210_millers_yard_city_centre",
            "area": "centre",
            "food": "asian_oriental",
            "phone": "01223_363471",
            "pricerange": "cheap",
            "postcode": "cb21rq"
         },
         {
            "name": "yippee_noodle_bar",
            "address": "40428_king_street_city_centre",
            "area": "centre",
            "food": "asian_oriental",
            "phone": "01223_518111",
            "pricerange": "moderate",
            "postcode": "cb11lh"
         },
         {
            "name": "j_restaurant",
            "address": "86_regent_street_city_centre",
            "area": "centre",
            "food": "asian_oriental",
            "phone": "01223_307581",
            "pricerange": "cheap",
            "postcode": "cb21dp"
         },
         {
            "name": "saigon_city",
            "address": "169_high_street_chesterton_chesterton",
            "area": "north",
            "food": "asian_oriental",
            "phone": "01223_356555",
            "pricerange": "expensive",
            "postcode": "cb41nl"
         },
         {
            "name": "kymmoy",
            "address": "52_mill_road_city_centre",
            "area": "centre",
            "food": "asian_oriental",
            "phone": "01223_311911",
            "pricerange": "expensive",
            "postcode": "cb12as"
         }
      ]
   }
   
```
###  Interactive IR
```
python ir_indexing.py --interactive
```

