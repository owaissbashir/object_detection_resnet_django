from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

def handle_uploaded_file(f):
    with open('img.jpg','wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

# Create your views here.
def home(request):
    return render(request,'home.html')

def imageprocess(request):
    form=ImageUploadForm(request.POST,request.FILES)
    print(request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])
        model=ResNet50(weights='imagenet')
        image=Image.open('img.jpg')
        image=image.resize((224,224))
        x = img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])
        a=decode_predictions(preds, top=3)[0]
        print(a[0][1],'djdjdj')


    return render(request,'result.html',context={'item':a[0][1]})