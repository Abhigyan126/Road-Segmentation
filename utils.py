import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class VideoStitcherOpenCV:
    @staticmethod
    def predict(image):
        generator_ = tf.keras.models.load_model('GAN_Sat_image_grey_300.h5')
        combined_image = tf.cast(img_to_array(image), tf.float32)

        image = combined_image

        image = tf.image.rgb_to_grayscale(tf.image.resize(image,(256,256)))/255

        predicted = generator_.predict(tf.expand_dims(image, axis=0))[0]

        plt.figure(figsize=(10, 8))

        plt.subplot(1, 3, 1)
        plt.imshow(image[:,:,0], cmap='gray')  # Show grayscale image
        plt.title("Satellite Image")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predicted[:,:,0], cmap='gray')  # Show grayscale predicted image
        plt.title("Predicted Image")
        plt.axis('off')

        plt.show()
        return predicted

    def process_video(self, video_path, limit_frames, skip_frames):
        cap = cv2.VideoCapture(video_path)

        imgs = []

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
   
            frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
            imgs.append(frame)
            frame_count += 1

            if len(imgs) >= limit_frames:
                break

        cap.release()

        stitcher = cv2.Stitcher_create()

        if stitcher is None:
            print("Error: Failed to create stitcher object")
        else:
            status, output = stitcher.stitch(imgs)

            if status != cv2.Stitcher_OK:
                print("Stitching isn't successful")
            else:
                output_file_path = "img.png"  
                output = self.predict(output)  # Corrected here
                cv2.imwrite(output_file_path, output)
                print(f'Stitched panorama saved as {output_file_path}')
