Dataset can be downloaded here:
images:
http://ufldl.stanford.edu/housenumbers/train.tar.gz
http://ufldl.stanford.edu/housenumbers/test.tar.gz
To make the code work do the following:
- download test and train sets from above links
- put images in working directory, in folder train/test
- run script preprocessing_data_colab
- run script model
Description of code:
All images are filtered (those which have width less than 80 and height less than 40 are removed). Also, false annotations are filtered.
All images are resized to 32x32. For easier uploadin to Google Colab, all images are saved to .npy file.
Classificator is simple Convolutional Neural Net. Performances are shown using confusion matrix.

Note: Jupyter Notebook file can be used for Google Colab training.