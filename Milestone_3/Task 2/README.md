**Task 1 - BEGIN-VRM multi-class multi-label classification model**

Contains the code for the models reported in the evaluation table of the paper. The contents should be self-explanatory given the filenames and the information from the table in the paper.

The file only_begin.ipynb contains the final model to classify BEGIN labels given knowledge and the user prompt.

The file only_vrm_prompt.ipynb contains final model to classify the response to its VRM speech acts given the knowledge and the user prompt.

The file only_vrm_history.ipynb contains final model to classify the response to its VRM speech acts given the knowledge and the entire conversation history.

The file combine_begin_vrm.ipynb contains final model to classify the response to its BEGIN classes and VRM speech acts given the knowledge and the entire conversation history.

These notebooks are designed to be run in Google Colab but can be run on a python notebook friendly environment. Pip installations might differ but the code should work fine.
