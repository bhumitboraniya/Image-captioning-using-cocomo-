# Image-captioning-using-cocomo 

In this project i have used 2 different datasets
1. Flicker 2. COCOMA

both dataset contains images with their captions. images contains with the unique id and that id maps with the captions in the dataset.
Flicker dataset is small because that contains small amount of images that’s why model that is trained on flicker dataset is not giving much more accuracy that’s why i need to switch into the COCOMA dataset that contains large amount of images with thier captions.

My project idea is to use Encoder- decoder layer. Encoder through we can extract features via CNN layers and at decoder part we can use LSTM model to predict captions based on the features.
In this project firstly i have store all the captions in the list and doing pre-processing of the captions like converting all the captions into lowercase and removing extra spaces and adding starting and ending sequence because we have to predict caption with the LSTM.

After that as we know that machine can not understand the string text. because our captions is in string format so that we will do Vectorization to convert our string captions into numbers. so that machine can easily understand the captions. after that we will find similarity between the vactorization through embedding. 
embeddings are getting similarity of the numbers and make numbers closer by closer.

After that we will pre process all the images that are in the dataset. we will do augmentaion and do resizing the image because all the imgaes will have different size of resolutions that can be difficult to find features though CNN so that we will make same size of all the images. and give image list to VGG16 model to getting feature map of the images.

Our encoder part is done. Now at decoder layer extracted features will pass to the LSTM model with the Captions embeddings and LSTM will geneate captions. based on their 3 nodes. forgot node, input node, output node.


project-directory/ |-- minor-project.ipynb # Main project notebook |-- /input |-- coco-2017-dataset |-- annotations | |-- captions_train2017.json |-- train2017 |-- /output # Stores processed data or model outputs

## **Features**
1. **Text Preprocessing**:
   - Normalizes text by converting to lowercase.
   - Removes punctuation and extra spaces.
   - Adds special tokens such as `[start]` and `[end]` for sequence modeling.
   - Utilizes regular expressions (regex) for efficient text cleaning.

2. **Image-Text Pairing**:
   - Extracts image-caption pairs from the COCO 2017 dataset.
   - Processes annotations stored in JSON format to create a structured dataset.

3. **Machine Learning Pipeline**:
   - Loads pre-trained TensorFlow/Keras models for feature extraction or training.
   - Includes visualization tools for better understanding data and model performance.

4. **Dependencies and Libraries**:
   - TensorFlow/Keras for deep learning.
   - Pandas and NumPy for data manipulation.
   - Matplotlib for visualization.
   - Regex for text cleaning.

---

## **Setup Instructions**

### **Prerequisites**
- Python 3.8 or higher.
- Libraries: Install the required libraries using:
  ```bash
  pip install tensorflow pandas numpy matplotlib


## **Directory Structure**

project-directory/
|-- minor-project.ipynb  # Main project notebook
|-- /input
    |-- coco-2017-dataset
        |-- annotations
        |   |-- captions_train2017.json
        |-- train2017
|-- /output  # Stores processed data or model outputs



## **Future Scope**
Extend the pipeline to include advanced text embeddings (e.g., BERT, GPT).
Experiment with generative models like Transformers for improved captions.
Enhance preprocessing for multilingual datasets
