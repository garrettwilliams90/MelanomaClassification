# Identifying Melanoma in Images using Deep Learning

Australia has the highest rate of melanoma per capita among all countries. This results in over deaths that could be avoided if detected early on. This project aims to identify melanoma in images of skin lesions using neural networks, more specifically deep learning tools. The dataset was generated by the [International Skin Imaging Collaboration (ISIC)](https://challenge2020.isic-archive.com/) and includes over 38,000 images. My final model was a pretrained VGG16 neural network with an accuracy of 94% and a recall of 84%.

![Melanoma?](https://raw.githubusercontent.com/garrettwilliams90/MelanomaClassification/main/Images/sunscreen-question-mark.jpeg) <br>
*Image by [Tatiana Kim](https://www.istockphoto.com/portfolio/TatianaKim?mediatype=photography) via [aad.org](https://www.aad.org/public/everyday-care/sun-protection/sunscreen-patients)*

## Business Understanding

Skin cancer is one of the most common types of cancer in the world. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. According to the [World Cancer Research Fund](https://www.wcrf.org/dietandcancer/skin-cancer-statistics/), Australia has the highest rate of Melanoma per capita. [Cancer Australia](https://www.canceraustralia.gov.au/cancer-types/melanoma/statistics) estimates close to 17,000 new cases of Melanoma have been diagnosed in 2021, resulting in over 1,000 deaths this year. 

Unlike other cancers though, skin cancer can be visibly seen. By using image classification tools, my work aims to accurately predict if a skin lesion is malignant or benign. By identifying if the skin has Melanoma early on, lives can be saved. My hope is that you, the Australian Department of Health, would use your resources in conjuction with my model to develop and market an app for the benefit of your citizens. Using their mobile phones, they can easily take a picture and recognize whether they need to go to a dermatologist for further diagnosis.

## Data Understanding

The data was created by the [Society for Imaging Informatics in Medicine (SIIM)](https://siim.org/) and [International Skin Imaging Collaboration (ISIC)](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main). SIIM is the leading healthcare organization for informatics in medical imaging who's mission is to advance medical imaging informatics through education, research, and innovation in a multi-disciplinary community. ISIC is an international effort to improve melanoma diagnosis. The ISIC Archive contains the largest publicly available collection of quality-controlled dermoscopic images of skin lesion. I was able to use the data through the [SIIM-ISIC Melanoma Classification competition](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview) on Kaggle.

The following is the citation of the original dataset under CC BY-NC 4.0:

> The ISIC 2020 Challenge Dataset https://doi.org/10.34970/2020-ds01 (c) by ISDIS, 2020
> 
> Creative Commons Attribution-Non Commercial 4.0 International License.
> 
> The dataset was generated by the International Skin Imaging Collaboration (ISIC) and images are from the following sources: Hospital Clínic de Barcelona, Medical University of Vienna, Memorial Sloan Kettering Cancer Center, Melanoma Institute Australia, Sydney Melanoma Diagnostic Centre, University of Queensland, and the University of Athens Medical School.
> 
> You should have received a copy of the license along with this work.
> 
> If not, see https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt.

One limitation of the dataset is the size of images. Since the images had different sizes and were too large for my code to run, I used resized images, found [here](https://www.kaggle.com/cdeotte/jpeg-melanoma-512x512). Another major limitation from this dataset is how imbalanced the target is. 98% of the over 33,000 images were classified as benign. To combat this imbalance, I added malignant images from the 2019 SIIM-ISIC Melanoma Classification competition, found [here](https://www.kaggle.com/cdeotte/jpeg-isic2019-512x512). I also added malignant images that weren't used in the 2019 or 2020 competitions, found [here](https://www.kaggle.com/cdeotte/malignant-v2-512x512).

The final dataset after combining the 3 had over 38,000 images. As a baseline understanding. 85.1% of the validation images are benign and 14.9% of the validation images are malignant. This means that the models accuracy would be 85% if it always predicted 'Benign'. All code was run using Kaggle and can be found [here](https://www.kaggle.com/garrettwilliams90/code).

![Benign Skin Lesions](https://raw.githubusercontent.com/garrettwilliams90/MelanomaClassification/main/Images/Examples-of-benign-skin-lesions.png)

![Malignant Skin Lesions](https://raw.githubusercontent.com/garrettwilliams90/MelanomaClassification/main/Images/Examples-of-malignant-skin-lesions.png)

## Modeling

I ran 6 different models, fitting to the training set and evaluating on the validation set:
- Simple Baseline Model
- Convolutional Neural Network
- Convolutional Neural Network with Dropout Layers
- Pretrained VGG16
- Pretrained ResNet50
- Pretrained InceptionResNetV2

The key metrics I focused on were Accuracy, Recall, and AUC-ROC. I focuesed on Recall because having False Negatives are more costly than having False Positives. False Negatives is having a malignant skin lesion but predicting that it's benign. This would result in people thinking they're healthy when they arent, which could result in lives being lost.

## Evaluation

The final model will be the VGG16 because it had the highest accuracy and recall score. Again, recall is the second evaluation metric because a low score would mean our model is predicting a skin lesion is benign when it's actually malignant. This implies you're healthy when you actually aren't and need to seek medical assistance. Now, I'll take our VGG16 model and evaluate it on the testing set, which I held out. This is so I can truly evaluate our model on unseen images.

![Confusion Matrix](https://raw.githubusercontent.com/garrettwilliams90/MelanomaClassification/main/Images/final-model-confusion-matrix.png)

![Metrics](https://raw.githubusercontent.com/garrettwilliams90/MelanomaClassification/main/Images/final-model-metrics.png)

My model accurately predicts the diagnosis of a skin lesion 93.8% of the time and incorrectly labels the lesion as benign on 16.3% of the time. As a reminder, my baseline understanding had an accuracy of 85.1% and a recall score of 0.

## Conclusion

The Australian Department of Health can develop and market a mobile app for the public that uses my model to classify if a person has melanoma or not. This would result in quicker reactions to seek out professionally-trained Dermatologists, which could save lives. 

In the future, if I had more time to work on this project, I'd use tools to combat the class imbalance, like `SMOTE` and `class_weight`, or even undersampling the benign class. This last option would also result in more memory on Kaggle, allowing the model to learn from the enitre training set (not just a third). Another way to free up more memory is to resize the images to something smaller, like 64x64, but this could mean the model isn't learning as much. Lastly, with the ultimate goal of improving the model, I'd try image augmentation, and increaing the epochs and patience to allow for the model to run longer. 

## Repository Navigation
All notebooks were run using Kaggle and can be found [here](https://www.kaggle.com/garrettwilliams90/code). <br>
My final notebook specifically can be found [here on Kaggle](https://www.kaggle.com/garrettwilliams90/melanoma-classification-final-notebook) or [here on Github](https://github.com/garrettwilliams90/MelanomaClassification/blob/main/melanoma-classification-final-notebook.ipynb) <br>
Orignal Data can be found [here on Kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview) or [here on the ISIC Archives](https://challenge2020.isic-archive.com/) <br>
The 3 datasets that were reformatted by Chris Deotte can be found [here](https://www.kaggle.com/cdeotte/jpeg-melanoma-512x512), [here](https://www.kaggle.com/cdeotte/jpeg-isic2019-512x512), and [here](https://www.kaggle.com/cdeotte/malignant-v2-512x512).
```
├── Images
├── Notebooks
│       ├── Model_Evaluations
│               ├── evaluate-1-baseline-and-2-first-cnn-model.ipynb
│               ├── evaluate-3-second-cnn-model-and-4-vgg16-model.ipynb
│               └── evaluate-5-resnet50-and-6-inceptionresnet-model.ipynb
│       ├── Model_Iterations
│               ├── baseline-model.ipynb
│               ├── first-cnn-model.ipynb
│               ├── inceptionresnetv2-model.ipynb
│               ├── resnet50-model.ipynb
│               ├── second-cnn-model.ipynb
│               └── vgg16-model.ipynb
│       └── eda-and-data-preprocessing.ipynb
├── README.md
├── final_model.h5
├── melanoma-classification-summary.ipynb
└── requirements.txt
```
