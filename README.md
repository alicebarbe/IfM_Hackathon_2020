# IfM_Hackathon_2020

### Two-part artificial intelligence + machine learning solution for quality control and inspection of produces

 - Set up camera system facing the produce to be inspected. This could be pointing to the produce as it passes on a conveyor belt. The system is able to locate produce within the image and can operate with many products in view. Therefore many camera locations will function well
 - Install dependencies and download the YOLO trained weights.
 - The pretrained YOLO  model will pick out images of specific product and identify them in their broad categories.
 - Each image output from the YOLO model is passed to an SVM  machine learning algorithm that will determine whether the product is bad or not. To increase the quality of the results, this model should be trained using the data output by the YOLO model. A user interface might be created for this in the future.
 - The end output, a pass/fail indicator on whether the product is damaged or not, could be read or connected to other modules in various forms: indicator lights, spreadsheets, display panels, etc.
