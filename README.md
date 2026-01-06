## User Interface
![Xception Model UI](UI/Xception_Ui2.jpeg)

## Deployment Demo
https://github.com/user-attachments/assets/1387e108-e032-4c76-8544-34be2f9b9d5b

## 🧠 Brain MRI PDF Report (Sample) :
![Xception AI Report](./UI/XceptionAI-Report.jpeg)


## Project Important Links :
________________________________

**Deployed Model Streamlit Link** :https://mri-xception-8cqqjkblxc5saqretu2cqx.streamlit.app/

**Dataset Link** : https://zenodo.org/records/12735692

**Trained Model Link** : :https://huggingface.co/HemantMishraDeepak/newXception/resolve/main/xception_brain_tumor_weights.weights.h5

# 🧠 MRI  Detection Model:
__________________________________
    This  Model is built using Deep Learning and Transfer Learning, 
    leveraging the power of the Xception Architecture, 
    (best for medical images fine details)   which is Pre-Trained 
     on large-scale image Datasets. Fine-tuning is applied 
    to adapt the model specifically for medical MRI Imaging.
    
### **🔬 About Model**
______________________________

      This project presents an AI-based MRI Brain Tumor Detection System 
      which is trained over 6000 mri images , having 4 classes 
      designed to assist in the identification and classification 
      of brain tumors from MRI scan images.
     
- **Detects brain tumor presence from MRI scans.**
- **Classifies tumor types including.**
 
      1. Glioma Tumor
      2. Meningioma Tumor
      3. Pituitary Tumor
      4. No Tumor
  
# Model Performance Metrics :
__________________________________
     The model is evaluated across Training, Validation,
     and Independent Test datasets.
     
### Training & Validation Performance


  | **Metric**              | **Value** |
|-------------------------|-----------|
| **Training Accuracy**   | **99.40%** |
| **Training Loss**       | **0.0270** |
| **Validation Accuracy** | **96.93%** |
| **Validation Loss**     | **0.0986** |



### Training , Validation Learning Curves :
__________________________________________


    Accuracy
    1.00 |                                  __________________ Training (0.99)
    0.98 |                          _______/─────────────── Test (0.98)
    0.96 |                    _____/──────── Validation (0.97)
    0.94 |              _____/
    0.92 |         _____/
    0.90 |    _____/
    0.88 |___/
         +------------------------------------------------------------------
          1    4    7   10   13   16   19
                        Epochs


  ### Training , Validation Loss Curves : 
_____________________________________
    
    Loss 
    0.60 |
    0.55 | \
    0.50 |  \
    0.45 |\  \
    0.40 | \  \
    0.35 |  \  \
    0.30 |   \  \
    0.25 |    \  \
    0.20 |     \  \
    0.10 |       \  \___________ Validation ≈0.10
    0.03 |        \__________________________ Train ≈0.03
    0.05 |
         +------------------------------------------------
         1   3   5   7   9   11  13  15  17  19
                              Epochs
                              
### Classification Report Validation  Data

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Glioma       | 0.98      | 0.99   | 0.98     | 264     |
| Meningioma   | 0.94      | 0.98   | 0.96     | 267     |
| No Tumor     | 0.98      | 0.92   | 0.95     | 319     |
| Pituitary    | 0.97      | 0.98   | 0.98     | 291     |
| **Accuracy** |           |        | **0.97** | **1141**|


## ROC Curve

                TPR (Y)
            1.0 ┤│──────────────────────────────────────────---
                ┤│                                         --
                ┤│                                      -- 
                ┤│                                  -- 
                ┤│                              -- 
            0.6 ┤│                          -- 
                ┤│                      -- 
                ┤│                  --                  
                ┤│              --                                            
            0.2 ┤│          --                      
                ┤│    -- 
            0.0 ┼└──────--─────────────────────────────────────
               0.0       0.2      0.4      0.6      0.8      1.0
                           FPR (X)



                  Glioma (AUC≈0.999)
                  Meningioma (AUC≈0.999)
                  No Tumor (AUC≈1.000)
                  Pituitary (AUC≈1.000)

- The ROC curve shows that the model achieves. 
- a very high true positive rates extremely
- low false positive rates, indicating excellent tumor
- discrimination capability.

                         
 ### Confusion Matrix  of Validation Data 
  _______________________________________


                          P R E D I C T E D
    ______________________________________________________________________
            |      Glioma |  Meningioma |   No Tumor |  Pituitary
    ______________________________________________________________________
    Actual  | Glioma      |     261     |      2     |      0     |    1
    ______________________________________________________________________
    Actual  | Meningioma  |      0      |     262    |      5     |    0
    ______________________________________________________________________
    Actual  | No Tumor    |      6      |     10     |    295    |    8
    ______________________________________________________________________
    Actual  | Pituitary   |      0      |      4     |      1     |   286
    ______________________________________________________________________

### Validation Confusion Matrix Simlification : 
 ________________________________________

    Actual ↓         Predicted ↓
    ------------    ----------------------
    Glioma        →   261 correct, 3 wrong
    Meningioma    →   262 correct, 5 wrong
    No Tumor      →   295 correct, 24 wrong
    Pituitary     →   286 correct, 5 wrong

    
# Test Performance
__________________________


## Test Set Performance (Unseen Data)
_________________________________________

### - **Test Accuracy** : **98.93%**
###  - **Test Precision**: **98.93%**
### - **Test Recall**   : **98.93%**

## Test Set Performance Classification Report 

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Glioma       | 1.0000    | 0.9967 | 0.9983   | 300     |
| Meningioma   | 0.9935    | 0.9967 | 0.9951   | 306     |
| No Tumor     | 1.0000    | 1.0000 | 1.0000   | 405     |
| Pituitary    | 0.9967    | 0.9967 | 0.9967   | 300     |

### TEST CONFUSION MATRIX

                             Predicted
     _______________┌──────────┬────────────┬──────────┬────────────┐
     Actual         │ Glioma   │ Meningioma │ No Tumor │ Pituitary  │
     ───────────────┼──────────┼────────────┼──────────┼────────────┤
     Glioma         │   299    │     1      │    0     │     0      │
     Meningioma     │    0     │   305      │    0     │     1      │
     No Tumor       │    0     │     0      │   405    │     0      │
     Pituitary      │    0     │     1      │    0     │   299      │
     ───────────────┴──────────┴────────────┴──────────┴────────────┘

### Test  Confusion Matrix Simlification : 
________________________________________

    Actual ↓              Predicted ↓
    --------------    ----------------------
    Glioma        →   299 correct, 1 wrong
    Meningioma    →   305 correct, 1 wrong
    No Tumor      →   405 correct, 0 wrong
    Pituitary     →   299 correct, 1 wrong


### Recall Value – Medical Interpretation (MRI Brain Tumor Detection)
_______________________________________________________________________

    This model demonstrates strong recall performance
    across all MRI brain tumor classes, with Glioma recall of 99.67%,
    Meningioma recall of 99.67%, No Tumor recall of 100.00%,
    and Pituitary tumor recall of 99.67%.

    The consistently high recall for tumor-positive classes,
    particularly Pituitary (99.67%), Meningioma (99.67%),
    and Glioma (99.67%), indicates that the model is highly sensitive
    toward detecting abnormal brain structures in MRI scans.

    In medical screening and radiological diagnostics,
    Recall is critically important, as missing a brain tumor
    (false negative) can delay treatment and significantly
    impact patient outcomes.

    Although an ideal Recall value is close to 100%,
    achieving perfect Recall in MRI-based tumor detection
    is challenging due to tumor morphology variations,
    imaging noise, inter-patient diversity, and dataset limitations.
    Despite these challenges, the proposed model achieves
    near-perfect recall across all classes on the test dataset.

Overall, the achieved Recall values suggest that the model is
well-suited for assistive clinical screening,
where prioritizing tumor detection over false alarms
is essential.

# Our Model Architecture 🫆
_____________________________

         ______________________           _________________________         ________________________
        |   Input MRI Image   |          |                         |       |                       |
        |   299 x 299 x 3     |--------> |    Data Augmentation    |------>|  Rescaling (1 / 255)  |
        |_____________________|          |_________________________|       |_______________________| 
                                                   |
                              _____________________v______________________________
                             |                                                     |
                             |                    Xception Network                 |
                             |     Pretrained on ImageNet (include_top = False)    |
                             |     Depthwise Separable Convolutions                |
                             |_____________________________________________________|
                                                   |
                                                   v
           ____________________________     ________________________     ______________________
          |  Global Average Pooling    |-->|      Dropout (0.3)     | -->|  Dense (128 units)  |
          |____________________________|   |________________________|    | Activation : ReLU   |
                                                                         |_____________________|             
                                                                                 |
                                                                                 v
                                                                    ______________________
                                                                   |    Dropout (0.25)    |
                                                                   |______________________|
                                                                           |
                                                                           v
                                                _________________________________________________
                                               |               Softmax Output Layer              |
                                               |     Classes : Glioma | Meningioma | No Tumor    |              
                                               |                   Pituitary                     |
                                               |_________________________________________________|
                                               

## Training Strategy (EfficientNet-B3) :

     ┌────────────────────────────────┐      ┌────────────────────────────────┐
     │   Load Xception Backbone       │────▶ │   Load ImageNet Weights        │
     │   (include_top = False)        │      │   (Pretrained Initialization)  │
     └───────────────┬────────────────┘      └───────────────┬────────────────┘
                     │                                       │
                     └──────────────┬────────────────────────┘
                                    ▼
        ┌────────────────────────────────┐       ┌────────────────────────────────┐
        │   Freeze Base Model            │────▶ |    Configure Optimizer          │
        │   (Feature Extraction Mode)    │       │   (Adamax, Initial LR)         │
        └───────────────┬────────────────┘       └───────────────┬────────────────┘
                         │                                       │
                         └──────────────┬────────────────────────┘
                                        ▼
              ┌────────────────────────────────┐      ┌────────────────────────────────┐
              │   Train Classifier Head        │────▶ │   Monitor Validation Metrics  │
              │   (GAP + Dense + Dropout)      │      │   (Accuracy / Loss)            │
              └───────────────┬────────────────┘      └───────────────┬────────────────┘
                              │                                       │
                              └──────────────┬────────────────────────┘
                                             ▼
              ┌────────────────────────────────────────────────────────────────┐
              │              Fine-Tuning (Unfreeze Top Layers)                 │
              │             Reduced Learning Rate for Stability                │
              └───────────────────────────────┬────────────────────────────────┘
                                              ▼
               ┌────────────────────────────────────────────────────────────────┐
               │                   Apply Training Callbacks                     │
               │               • ModelCheckpoint (Save Best Model)              │
               │               • EarlyStopping (Overfitting Control)            │
               │               • ReduceLROnPlateau (Adaptive LR)                │
               └───────────────────────────────┬────────────────────────────────┘
                                               ▼
              ┌────────────────────────────────────────────────────────────────┐
              │                  Best Xception Model Saved                     │
              └────────────────────────────────────────────────────────────────┘

 
                          
# Installation & Setup :
___________________________

> Follow the steps below to Run this project locally and test the Pneumonia Detection Model.

1. Clone this Repository:
   
       git clone https:https://github.com/HemantMishra2003/MRI-Xception.git
   
3. Install Python Dependencies:

       pip install -r requirements.txt
   
3. To Run the Streamlit app and test the model:
   
               streamlit run app.py
   
5. Test image for Prediction:

        Use  my given link Dataset to get Test image or  use my Assets folder
        my Assets folder to get all 4 classes  Mri  image to make prediction.
   
# How can you Contribute 🙂
____________________________

If you would like to contribute to this Project, please follow these steps:

1. Fork the repository.
2. Create a New Branch for your feature or fix.
3. Make your changes with clear and meaningful commits.
4. Submit a Pull Request describing your changes.
   
        Suggestions for improvements, bug fixes, documentation enhancements,
                    and feature ideas are always welcome.😊

# Model Comparison: EfficientNet-B3 vs Xception
_____________________________________________________________________

> Initially, the model was trained using EfficientNet-B3.
> where it achieved an overall test accuracy of approximately 87%.
> Although EfficientNet-B3 is a strong and lightweight architecture. 
> its performance was limited in capturing the fine-grained structural.
> variations present in MRI brain tumor images.

> To improve performance, the model was later retrained using the Xception architecture,
> which resulted in a significantly higher test accuracy of approximately 98%.
 
##  Why did Xception perform better?

    The improvement in accuracy can be attributed 
    to the architectural differences between the two models:

- Xception uses depthwise separable convolutions.
- which allow the model to learn spatial features. 
- and channel-wise features independently.
- This is particularly effective for MRI scans. 
- where subtle texture and boundary differences are critical.

- Brain tumor classes often share similar visual patterns, 
- and Xception is better at capturing fine-grained tumor 
- morphology compared to EfficientNet-B3.

- EfficientNet-B3 is optimized for parameter efficiency.
- and general-purpose image classification, 
- whereas Xception provides stronger feature disentanglement,
- making it more suitable for medical imaging tasks.

- The improved feature extraction capability of Xception led to better.
- class separation, resulting in higher accuracy and more reliable predictions.

       THAT  TARINED FILE IS AVAILABLE IN THIS REPO AND THEIR MATRICS TOO
        

# Thanks Section 🙏
 _____________________
   
         Thanks to my super senior Vishwas Mishra(Rolls Royce), who keeps motivating me
                   and continuously helps me to improve my skills.” ❤️

     
      
