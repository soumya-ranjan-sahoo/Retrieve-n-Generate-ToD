# Retrieval-Augmented-Generative-ToDs

## This work is carried out as part of the MSc thesis at the faculty of Mathematics and Computer Science, Saarland University and Fraunhofer IAIS.

### Installation
```
conda create -n RnG-ToD python=3.7
pip install -r requirements.txt
```

## Pre-processing
### IR (indexing):
Generate indices (.pkl files) by running:
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


### Evaluation
```
python eval.py --generate runs/gpt2/incar --dataset incar --generation_params_file config/gpt2/generation_params.json --eval_dataset test  --output_file outputs/baseline_test.json
```
The program will automatically pick the best model from the directory for the evaluation.

### Refactored structure of each dialogue
```
 {
      "task": "hotel",
      "id": 1,
      "history": [
         "i need to find lodgings on the north side of town .",
         "we have 13 available lodging areas in the north . any preferences ?",
         "some place in the north and it does not need to have free parking"
      ],
      "response": "there are 2 locations that match your criteria . the alpha-milton_guest_house with a 3_star rating and the avalon with a 4_star rating . do either of these sound good ?",
      "ref_ents": [
         "3_star",
         "4_star",
         "alpha-milton_guest_house",
         "avalon"
      ],
      "kg": [
         {
            "name": "arbury_lodge_guesthouse",
            "address": "82_arbury_road",
            "area": "north",
            "phone": "01223364319",
            "postcode": "cb42je",
            "pricerange": "moderate",
            "stars": "4_star",
            "type": "guesthouse",
            "choice": "13",
            "ref": "2asa82vj"
         },
         {
            "name": "kirkwood_house",
            "address": "172_chesterton_road",
            "area": "north",
            "phone": "01223306283",
            "postcode": "cb41da",
            "pricerange": "moderate",
            "stars": "4_star",
            "type": "guesthouse",
            "choice": "13",
            "ref": "2asa82vj"
         },
         {
            "name": "worth_house",
            "address": "152_chesterton_road",
            "area": "north",
            "phone": "01223316074",
            "postcode": "cb41da",
            "pricerange": "cheap",
            "stars": "4_star",
            "type": "guesthouse",
            "choice": "13",
            "ref": "2asa82vj"
         }]}
```


