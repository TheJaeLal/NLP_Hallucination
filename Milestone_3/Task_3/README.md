Knowledge Grounded Response Generation Code (Milestone 3)
========================================================

Contains the task 3 code from milestone 3.

### Models
Three models as shown in the report were used : 
T5_response_gen_WoW.ipynb: Jupyter Notebook for Initial Response Generator Model 
T5_response_edit_FaithDial.ipynb: Jupyter Notebook for FaithFul Response Editor Model 
T5_response_gen_FaithDial.ipynb: Jupyter Notebook for Direct Response Generator Model 

### Saved Responses
All files are pipe separated csv and responses are from FaithDial Test set

T5_gen_WoW.txt: Responses of Initial Response Generator Model
T5_response_edit_FaithDial.ipynb: Responses of FaithFul Response Editor Model 
T5_response_gen_FaithDial.ipynb: Responses of Direct Response Generator Model 

### Evaluation
metrics.ipynb contains code for calculating BLEU, BERT and ROUGE scores.

task1_eval.py is used to obtain hallucination critic's results on responses.
```python task1_eval.py T5_gen_WoW.txt```

task1_infer_2.py -> Module for performing inference on task1 model (hallucinatoin critic) used within the notebook

