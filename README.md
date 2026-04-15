# Hand Hygiene Gel Classifier 🧼

Proper hand hygiene is critical! The goal of our group project is to use AI and computer vision to automatically look at images of hands and calculate exactly how much sanitizing gel is on them. 

## 📋 How We Built It
1. **Labeling Data:** We manually traced the hands and the gel in our images using a tool called Label Studio.
2. **Training the Brain:** We used an AI model called YOLO (Instance Segmentation) and trained it in Google Colab to recognize the difference between "hand pixels" and "gel pixels".
3. **Grading the Hygiene:** The code compares the amount of gel to the size of the hand. It then gives the hand a hygiene grade:
   * **None:** Less than 5% gel coverage
   * **Low:** 5% to 14.9% gel coverage
   * **Medium:** 15% to 29.9% gel coverage
   * **High:** Over 30% gel coverage

## 🚀 How to Run Our Code
If you want to run this project on your own computer, make sure you download our required tools first! Open your terminal and run:
`pip install -r requirements.txt`

## 👥 Project Creators
* Zaineb 
* Eric
* Aishah
