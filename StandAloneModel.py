import torch
from PIL import Image
from torchvision import transforms

# holds all the artist on which the model was trained
classes = [
"Pablo Picasso",
     "Rene Magritte",
         "Joan Miro",
   "William Turner",
    "Leo"
]


model = torch.load('images_model_9.pth') # trained model taken from the main file

image_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image)
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    print(classes[predicted.item()])

    
    #local images given for prediction
classify(model, image_transforms, 'C:\\Users\\kaila\\PycharmProjects\\ImageClassification\\testImage\\WilliamTurner1.jpg', classes)
classify(model, image_transforms, "C:\\Users\\kaila\\PycharmProjects\\ImageClassification\\testImage\\Joan1.jpg", classes)
classify(model, image_transforms, 'C:\\Users\\kaila\\PycharmProjects\\ImageClassification\\testImage\\WilliamTurner1.jpg', classes)
classify(model, image_transforms, 'C:\\Users\\kaila\\PycharmProjects\\ImageClassification\\testImage\\PabloPicasso1.jpg', classes)
classify(model, image_transforms, 'C:\\Users\\kaila\\PycharmProjects\\ImageClassification\\testImage\\Leo.jpg.jpg', classes)

