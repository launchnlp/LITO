This repository provides the code and datasets for our paper **Enhanced Language Model Truthfulness
with Learnable Intervention and Uncertainty Expression**. In this work we introduce LITO, a context-aware intervention method that improves truthfulness in language models by selecting the most accurate response among multiple model-generated responses or refusing to respond if none is correct.
## Installation
In this the root folder of this repo, run the following commands to set things up.
```
conda env create -f environment.yaml
conda activate lito
pip install git+https://github.com/davidbau/baukit
mkdir results
git clone https://github.com/sylinrl/TruthfulQA.git
````

After cloning the TruthfulQA project, please replace its presents.py file located at: `TruthfulQA/truthfulqa/presents.py` with the updated [presents.py](https://github.com/launchnlp/LITO/tree/main/scripts/presets.py).

## results Directory Structure
There should be 4 directories inside the *results* directory one for each dataset. 
```
mkdir -p results/NQ
mkdir -p results/TriviaQA
mkdir -p results/SciQ
mkdir -p results/TQA 
````
Then, in each of these dataset directories, there would be one directory per model. Models names are: llama2_chat_7B, llama2_chat_13B, vicuna_7B, gpt2_large, and gpt2_xl.    

## Collect ITI Features
For each of the datasets, collect their corresponding ITI features for probe training. For example, to train linear probes for Llama2_Chat_7B model on the NQ dataset, run:
```
python collect_iti_activations.py llama2_chat_7B --dataset_name nq
```
This will provide the activations for each dataset and each model in order to find directions uisng the iti method. 


## Collect Hidden States for training and evaluating LITO
For each model, each dataset, each mode (train/test), and each of the 5 different intensity values (alpha), the following script needs to be ran:

```
python collect_hidden_states.py gpt2_xl --dataset_name trivia_qa --alpha 15 --mode train
```

## LITO Training and Evaluation

To train and then test LITO after collecting the responses, their corresponding hidden states, and confidence values, you can train/test LITO by running the following command: (here we train LITO on the responses generated by Llama2_Chat_7B model for the NQ task)

```
python classification.classifier_rnn_binary_main llama2_chat_7B --dataset_name nq --mode train
```

Please explore other hyper-parameters in each of the above scripts. 

## Contact
In case of any issues or questions, please send an email to ```farimaf (at) umich (dot) edu```.
