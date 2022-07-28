# kg-structure-aware-dialogues

### Installation
```
conda create -n kgconv python=3.7
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

###  Interactive IR
```
python ir_indexing.py --interactive
```

