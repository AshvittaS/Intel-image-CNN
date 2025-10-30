# Intel Image Classification using CNN

This project is a Convolutional Neural Network (CNN) model built with TensorFlow and Keras to classify images from the Intel Image Classification dataset. The model is trained to classify images into six categories: buildings, forest, glacier, mountain, sea, and street.

The project is deployed as a web application using Gradio and is hosted on Hugging Face Spaces.

## Live Demo

You can try out the live demo here: [Intel Image Classifier on Hugging Face Spaces](https://ashvitta07-intel-image-classification.hf.space/?__theme=system&deep_link=LhD6MZKpHhg)

## Dataset

The model is trained on the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Kaggle. This dataset contains around 25k images of size 150x150 pixels, distributed under 6 categories.

- **Training data**: ~14,000 images
- **Testing data**: ~3,000 images
- **Classes**: 'mountain', 'street', 'glacier', 'buildings', 'sea', 'forest'

## Model Architecture

The CNN model is built using `tf.keras.Sequential` and has the following layers:

1.  `Conv2D` layer with 32 filters, a kernel size of (3, 3), and ReLU activation.
2.  `MaxPooling2D` layer with a pool size of (2, 2).
3.  `Conv2D` layer with 32 filters, a kernel size of (3, 3), and ReLU activation.
4.  `MaxPooling2D` layer with a pool size of (2, 2).
5.  `Flatten` layer to convert the 2D feature maps to a 1D vector.
6.  `Dense` layer with 128 units and ReLU activation.
7.  `Dense` layer with 6 units (one for each class) and softmax activation.

The model is compiled with the `adam` optimizer and `sparse_categorical_crossentropy` loss function.

## Model Performance

The model was trained for 20 epochs and achieved a test accuracy of approximately 74%.

### Training History

The training and validation accuracy and loss over 20 epochs are shown below. The plots indicate that the model learns well, though there are signs of overfitting as the validation loss starts to increase while the training loss decreases.

*(The notebook contains plots for `train_acc vs val_acc` and `train_loss vs val_loss`)*

### Confusion Matrix

A confusion matrix was generated to evaluate the model's performance on the test set. It helps to see how many images were correctly and incorrectly classified for each class.

*(The notebook contains a heatmap of the confusion matrix)*

## How to run locally

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Intel-image-CNN.git
    cd Intel-image-CNN
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Gradio app:
    ```bash
    python app.py
    ```
    The application will be running on your local server.

## File Descriptions

-   `app.py`: The main file for the Gradio web application.
-   `intel_model.h5`: The saved, trained Keras model.
-   `class_names.pkl`: A pickled file containing the list of class names.
-   `notebook5363696015.ipynb`: The Jupyter notebook with the complete code for data loading, model creation, training, and evaluation.
-   `requirements.txt`: A list of Python packages required to run the application.
-   `README.md`: This file.