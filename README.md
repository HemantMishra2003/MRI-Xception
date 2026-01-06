## User Interface
![Xception Model UI](UI/Xception_Ui2.jpeg)

## Deployment Demo
https://github.com/user-attachments/assets/1387e108-e032-4c76-8544-34be2f9b9d5b

## 🧠 Brain MRI PDF Report (Sample) :
![Xception AI Report](./UI/XceptionAI-Report.jpeg)


## Project Important Links :
________________________________

**Deployed Model Streamlit Link** :https://mri-efficientnet-b3-grsdkpgyhre5wldydrhpcv.streamlit.app/

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
         +-------------------------------------------------
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
0.15 |      \  \
0.10 |       \  \___________ Validation ≈0.10
0.03 |        \__________________________ Train ≈0.03
0.05 |
     +------------------------------------------------
       1   3   5   7   9   11  13  15  17  19

                          Epochs

                         
   ### Confusion Matrix  of Test Data
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



## Test Set Performance (Unseen Data)
_________________________________________

### - **Test Accuracy:** **98.93%**
##  - **Test Precision:** **98.93%**
Test Recall   : 98.93%

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

