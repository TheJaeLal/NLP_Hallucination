Hallucination Critic Code
=========================

Contains the code for the models reported in the evaluation table for the hallucination. The contents should be
self-explanatory given the filenames and the information from the table in the paper.

The file `Task-1-Siamese.ipynb` contains our final task 1 model.

The file `Task-1-Augmented.ipynb` contains the experiment with the augmented training data.

These notebooks are designed to be run in Google Colab. You can change the root of the project with the
`PROJECT_ROOT` variable. It's expected that the `PROJECT_ROOT` directory exists, and that it has a subdirectory called
`data`. With this setup, the code should run as expected.
