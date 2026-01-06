## User Interface
![Xception Model UI](UI/Xception_Ui2.jpeg)

## Deployment Demo
https://github.com/user-attachments/assets/1387e108-e032-4c76-8544-34be2f9b9d5b
## 🧠 Brain MRI PDF Report (Sample) :

## Project Important Links :
________________________________

**Deployed Model Streamlit Link** :https://mri-efficientnet-b3-grsdkpgyhre5wldydrhpcv.streamlit.app/

**Dataset Link** : https://zenodo.org/records/12735692

**Trained Model Link** : :https://huggingface.co/HemantMishraDeepak/newXception/resolve/main/xception_brain_tumor_weights.weights.h5

# 🧠 MRI  Detection Model:
__________________________________
    This  Model is built using Deep Learning and Transfer Learning, 
    leveraging the power of the EfficientNet-B3  Architecture, 
    which is Pre-Trained on large-scale image Datasets. Fine-tuning is applied 
    to adapt the model specifically for medical MRI Imaging.
    
### **🔬 About Model**
______________________________
     This project presents an AI-based MRI Brain Tumor Detection System 
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
| **Training Accuracy**   | **93.06%** |
| **Training Loss**       | **0.1745** |
| **Validation Accuracy** | **86.75%** |
| **Validation Loss**     | **0.3912** |




### Training , Validation Learning Curves :
__________________________________________

       Accuracy
    0.98 |
    0.96 |
    0.94 |                              
    0.92 |                        ____/--------------------------- Training
    0.90 |                  _____/
    0.88 |            _____/---------------------------------Test
    0.86 |       ____/------------------------- Validation
    0.84 |  ____/
    0.82 |_/
         +--------------------------------
           1   4   7  10  13  16  19
                   Epochs


  ### Training , Validation Loss Curves : 
_____________________________________
    
    Loss
    0.50 |\
    0.45 | \
    0.40 |  \___________ Validation
    0.35 |
    0.30 |
    0.25 |
    0.20 |        __________ Train
    0.15 |_______/
    0.10 |
    0.05 |
         +-----------------------------------------
         1   3   5   7   9   11  13  15  17  19
                          Epochs

                         
   ### Confusion Matrix  of Test Data
_______________________________________

                                          P R E D I C T E D
    ________________________________________________________________________________________
            |            |     Glioma     |   Meningioma   |    No Tumor    | Pituitary  |
    _________________________________________________________________________________________
    Actual  | Glioma     |      244       |       43       |        4       |    9       |
    _________________________________________________________________________________________
    Actual  | Meningioma |       32       |      227       |       21       |    26      |
    _________________________________________________________________________________________
    Actual  | No Tumor   |        0       |        9       |      393       |    3       |
    _________________________________________________________________________________________
    Actual  | Pituitary  |        3       |       12       |        2       |    283     |
    _________________________________________________________________________________________


## Test Set Performance (Unseen Data)
_________________________________________

### - **Test Accuracy:** **87.98%**

#### Classification Report (Test Data)


| Class        | Precision | Recall  | F1-Score | Support |
|--------------|-----------|---------|----------|---------|
| Glioma       | 0.8746    | 0.8133  | 0.8428   | 300     |
| Meningioma   | 0.7801    | 0.7418  | 0.7605   | 306     |
| No Tumor     | 0.9357    | 0.9704  | 0.9527   | 405     |
| Pituitary    | 0.8816    | 0.9433  | 0.9114   | 300     |



### Recall Value – Medical Interpretation (MRI Brain Tumor Detection)
_______________________________________________________________________

      This model demonstrates strong recall performance 
      across all MRI brain tumor classes,with Glioma recall of 81.33%,
      with Glioma recall of 81.33%, Meningioma recall of 74.18%.


      No Tumor recall of 97.04%, and Pituitary tumor recall of 94.33%.
      The high recall for Tumor-Positive classes, particularly Pituitary(94.33%)
      and Glioma (81.33%), indicates that the model is highly sensitive
      toward detecting abnormal brain structures in MRI scans.


      In medical screening and radiological diagnostics,
      Recall is critically important,as missing a brain tumor (false negative) 
      can delay treatment and significantly impact patient outcomes.


      Although an ideal Recall value is close to 100%,
      achieving perfect Recall in MRI-based tumor detection is challenging due to
      tumor morphology variations, imaging noise, inter-patient diversity,
      and practical dataset limitations.


       Overall, the achieved Recall values suggest that the model is
       well-suited for assistive clinical screening,
       where prioritizing tumor detection over false alarms is essential.

