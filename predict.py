"""Command-line prediction for one image."""
import argparse, numpy as np, tensorflow as tf
from PIL import Image
CLASSES = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--image",required=True)
    parser.add_argument("--model",default="outputs/air_pollution_cnn.keras")
    args=parser.parse_args()
    model=tf.keras.models.load_model(args.model)
    image=Image.open(args.image).convert("RGB").resize((128,128))
    arr=np.asarray(image,dtype=np.float32)/255.0
    scores=model.predict(np.expand_dims(arr,0),verbose=0)[0]
    idx=int(np.argmax(scores))
    print("Visual class:",CLASSES[idx])
    print("CNN confidence: {:.1f}%".format(scores[idx]*100))
    print("This is a visual estimate, not a measured AQI.")
if __name__=="__main__": main()
