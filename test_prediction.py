from integration.prediction import predict_xray

image_path = "data/chest_xray/images/00009486_001.png"

result = predict_xray(image_path)

print(result)