# Molformer Training
## Baseline
Model got best validation score at epoch 3 but training loss kept falling so it is suspected that the model had to high learning rates which messed with the underlying pretrained weights. Starting lr were encoder_lr: 3.0e-5 and readout_lr: 1.0e-3

## lr strategy
As they are suspected to have started to high a series of lower weights will be tested 
### encoder_lr:
* 1.0e-5
* 3.3e-6
* 1.0e-6
* freeze_encoder=true

### readout_lr:
Wille be changed after encoder_lr has been tested and tested upon the best from there
* 1.0e-3
* 3.0e-4
* 1.0e-4

## Dropout strategy
As molformer delivers a 768-dimensional vector as output on a relatively small dataset around 3400 molecules into a high dimension mlp dim = 512. The model will have memorized the molcules quite quickly so the fropout needs to be quite high.
### dropout:
* 0.15
* 0.2 (same as in paper)
* 0.25
* 0.3

## Weight decay
using 1.0e-2

## Batch size
As many datapoints are NaN batch size needs to be quite large to combat this

### batch_size
* 8
 * 16
 * 32
 * 64

 ## Training runs
 ### Commands
 #### Encoder lr
 ##### lr = 1.0e-5
 python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-5 train.weight_decay=1.0e-2 readout.dropout=0.2 train.epochs=20 <br>
 Best epoch: 2 <br>
  "macro_ST-RAE": 0.7734841552559514 <br>

##### lr = 3.3e-6
python finetune.py model=molformer task=regression data=../data model.encoder_lr=3.3e-6 train.weight_decay=1.0e-2 readout.dropout=0.2 train.epochs=20<br>
Best epoch: 2    "macro_ST-RAE": 0.758453977963288

##### lr = 1.0e-6
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 train.weight_decay=1.0e-2 readout.dropout=0.2 train.epochs=20 <br>
Best epoch: 2 "macro_ST-RAE":  0.7655404340139761

##### lr = 0
python finetune.py model=molformer task=regression data=../data model.freeze_encoder=true train.weight_decay=1.0e-2 readout.dropout=0.2 train.epochs=20
Best epoch: 2 "macro_ST-RAE": 0.7750686894901696 


#### Readout lr
##### readout_lr=3.0e-4
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=3.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.2 train.epochs=20

Best epoch: 2  "macro_ST-RAE": 0.7574886756867926

##### readout_lr=1.0e-4
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=1.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.2 train.epochs=20
 
Best epoch: 6 "macro_ST-RAE": 0.7562107518885743

##### readout_lr=3.3e-5
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=3.3e-5 train.weight_decay=1.0e-2 readout.dropout=0.35 train.batch_size=4 train.epochs=20

epoch: 4 "macro_ST-RAE": 0.736232425456012
##### readout_lr=3.0e-6
 finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=3.0e-6 train.weight_decay=1.0e-2 readout.dropout=0.35 train.batch_size=4 train.epochs=20

 epoch": 18


##### readout_lr=1e-5
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=3.3e-5 train.weight_decay=1.0e-2 readout.dropout=0.35 train.batch_size=4 train.epochs=20

epoch: 8    "macro_ST-RAE": 0.7312589158974012

#### Dropout

##### Dropout = 0.25
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=1.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.25 train.epochs=20

Best epoch: 6 "macro_ST-RAE": 0.7542814686217199

##### Dropout = 0.3
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=1.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.3 train.epochs=20
Best epoch: 6 "macro_ST-RAE": 0.0.7521796393886069

##### Dropout = 0.35
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=1.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.35 train.epochs=20
Best epoch: 6 "macro_ST-RAE":  0.7502439557408142

##### Dropout = 0.4
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=3.3e-5 train.weight_decay=1.0e-2 readout.dropout=0.4 train.batch_size=4 train.epochs=20

#### batch size
##### batch size = 16
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=1.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.35 train.batch_size=16 train.epochs=20

Best epoch: 4  "macro_ST-RAE": 0.7490154663760749

##### batch size = 64
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=1.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.35 train.batch_size=64 train.epochs=20 <br>
best epoch = 8   macro_ST-RAE": 0.7521845989208678

##### batch size = 8
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=1.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.35 train.batch_size=8 train.epochs=20

best epoch = 8  "macro_ST-RAE": 0.7485273921450158

##### batch size = 4
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=1.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.35 train.batch_size=4 train.epochs=20

best epoch = 3  "macro_ST-RAE" 0.7445728805999874


##### batch size = 2
python finetune.py model=molformer task=regression data=../data model.encoder_lr=1.0e-6 model.readout_lr=1.0e-4 train.weight_decay=1.0e-2 readout.dropout=0.35 train.batch_size=2 train.epochs=20

best epoch = 3  macro_ST-RAE": 0.745825096925893}