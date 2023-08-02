import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers
import random

mp_holistic = mp.solutions.holistic # Holistic Keypoints : Pose Tubuh dan Tangan
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils # Utilitas menggambar

# Actions that we try to detect
actions = np.array(['Akan', 'Anda', 'Apa', 'Atau', 'Baca', 'Bagaimana', 'Bahwa', 'Beberapa', 'Besar',
    'Bisa', 'Buah', 'Dan', 'Dari', 'Dengan', 'Dia', 'Haus', 'Ingin', 'Ini', 'Itu',
    'Jadi', 'Juga', 'Kami', 'Kata', 'Kecil', 'Kumpul', 'Labuh', 'Lain', 'Laku',
    'Lapar', 'Main', 'Makan', 'Masing', 'Mereka', 'Milik', 'Minum', 'Oleh', 'Pada',
    'Rumah', 'Satu', 'Saya', 'Sebagai', 'Tambah', 'Tangan', 'Tetapi', 'Tidak', 'Tiga',
    'Udara', 'Untuk', 'Waktu', 'Yang'])

glob_bbox = None
glob_dist = None

def find_body_centroid(landmarks, main_body):
    if landmarks: ### jika landmark ditemukan
        x_bodies = []
        y_bodies = []
        z_bodies = []
        for i in main_body:
            x_bodies.append(landmarks.landmark[i].x)
            y_bodies.append(landmarks.landmark[i].y)
            z_bodies.append(landmarks.landmark[i].z)
        glob_bbox = [x_bodies, y_bodies, z_bodies]
        return np.average(x_bodies), np.average(y_bodies), np.average(z_bodies)
    else: ### jika landmark tidak ditemukan
        return 0, 0, 0
    
def euclidean(a, b):
    sum_sq = np.sum(np.square(a - b)) ## sumasi dari kedua titik
    euclidean = np.sqrt(sum_sq) ## akar kuadrat dari sumasi

    return euclidean

def pixel_match(coordinates, w, h): # rescaling keypoints menyesuaikan resolusi frame
    x = int(coordinates[0] * w)
    y = int(coordinates[1] * h)
    coordinates[0] = x
    coordinates[1] = y

    return coordinates

def landmarks_data(landmarks, data, key):
    radius = data["radius"]
    px_radius = data["px_radius"]
    centroid = data["centroid"]
    coordinates = []
    centroid_x = centroid[0]
    centroid_y = centroid[1]
    original_h = data["image"].shape[0]
    original_w = data["image"].shape[1]
    px_centroid_x, px_centroid_y = pixel_match([centroid_x, centroid_y], original_w, original_h)
    x = 0
    if landmarks:
        for i in landmarks.landmark:
            horizontal = px_centroid_x - px_radius # selisih kanan/kiri
            vertical = px_centroid_y - px_radius # selisih atas/bawah

            px_x, px_y = pixel_match([i.x, i.y], original_w, original_h)
            
            if px_x >= 0: 
                normalized_px_x = abs(horizontal - px_x) 
            else:  
                normalized_px_x = -abs(horizontal - px_x) 
            if px_y >= 0: 
                normalized_px_y = abs(vertical - px_y) 
            else: 
                normalized_px_y = -abs(vertical - px_y)

            # data["img_pixel"] = np.zeros([10*px_radius,10*px_radius,3],dtype=np.uint8)
            # data["img_pixel"].fill(155) # or img[:] = 155
            
            data["img_pixel"] = data["image"]
            data["img_pixel"] = cv2.circle(data["img_pixel"], (px_x,px_y), radius=5, color=(66,66,245), thickness=3)
            data["img_pixel"] = cv2.circle(data["img_pixel"], (int(centroid_x), int(centroid_y)), radius=5, color=(255,255,255), thickness=5)
            # data["img_pixel"] = cv2.circle(data["img_pixel"], (normalized_px_x,normalized_px_y), radius=5, color=(62,184,64), thickness=3)
            data["img_pixel"] = cv2.circle(data["img_pixel"], (int(px_centroid_x),int(px_centroid_y)), radius=5, color=(255,251,28), thickness=5)
            #cv2.imwrite(os.path.join(data["data_save_path"])+'/'+str(data["frame_index"])+'_real_.jpg', data["img_pixel"])
            # from matplotlib import pyplot as plt
            # plt.imshow(data["img_pixel"], interpolation='nearest')
            # plt.show()


            # agar skala tidak melebihi 1.92
            i.x = normalized_px_x / (2 * px_radius) #px_radius nilainya sekitar +-500pixel
            i.y = normalized_px_y / (2 * px_radius)
            coordinates.append(i.x)
            coordinates.append(i.y)
            coordinates.append(i.z)
            x += 1
    else: ### kalo landmarks tidak ditemukan kosongkan saja semua keypoints di tubuh/tangan
        if key == "hand":
            vertex_num = 21
        if key == "pose":
            vertex_num = 33
        for i in range(0, vertex_num):
            for i in range(0, 3):
                coordinates.append(0)

    return coordinates, landmarks


def draw_landmarks(image, params, key):
    mp_drawing = params["mp_drawing"]

    if key == 'ori':
        mp_holistic = mp.solutions.holistic
        mp_drawing.draw_landmarks(image, params["pose_landmarks"], mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(0,0,128), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(0,191,255), thickness=2, circle_radius=4)
                              )
        
    elif key == 'nor':
        mp_holistic = mp.solutions.holistic
        mp_drawing.draw_landmarks(image, params["normalized"], mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(250,128,114), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(139,0,0), thickness=2, circle_radius=2)
                              )
       
    return image

def final_extract(params):
    result = params["result"]
    left_hand_landmarks = result.left_hand_landmarks
    right_hand_landmarks = result.right_hand_landmarks
    pose_landmarks = params["pose_landmarks"]
    shoulders_centroid = params["shoulders_centroid"]
    hips_centroid = params["hips_centroid"]
    image = params["image"]
    im_h = image.shape[0]
    im_w = image.shape[1]
    # centroid bahu yang direkalulasi piksel resolusi
    point_a = pixel_match(shoulders_centroid.copy(), im_w, im_h)
    # centroid pinggang yang direkalulasi piksel resolusi
    point_b = pixel_match(hips_centroid.copy(), im_w, im_h)
   
    point_a = np.array(point_a)
    point_b = np.array(point_b)
    # print(point_a[0],point_b[0])
    params["image"] = cv2.line(params["image"], (int(point_a[0]), int(point_a[1])), (int(point_b[0]), int(point_b[1])), color=(255,255,255), thickness=3)
    params["image"] = cv2.line(params["image"], (int(shoulders_centroid[0]), int(shoulders_centroid[1])), (int(hips_centroid[0]), int(hips_centroid[1])), color=(255,255,255), thickness=3)

    # radius/jarak piksel sesuai dengan rekalkulasi piksel centroid
    px_radius = int(euclidean(point_a, point_b))
    params["px_radius"] = px_radius


    left_hand_coordinates, lh_landmarks = landmarks_data(left_hand_landmarks, params, "hand")
    right_hand_coordinates, rh_landmarks = landmarks_data(
        right_hand_landmarks, params, "hand")
    pose_coordinates, p_landmarks = landmarks_data(pose_landmarks, params, "pose")
    coordinates_collection = left_hand_coordinates + \
        right_hand_coordinates + pose_coordinates

    params["normalized"] = p_landmarks
    # draw_landmarks(params["image"],params,"nor")

    return coordinates_collection

def find_centroid(landmarks, data):
    indices_a = [11, 12]
    indices_b = [23, 24]
    centroid_a = np.array(find_body_centroid(landmarks, indices_a))
    centroid_b = np.array(find_body_centroid(landmarks, indices_b))
    return centroid_a, centroid_b

def keypoints_check(data):
    # Data
    centroid_indices = [0, 11, 12, 23, 24]

    ### global bounding box berfungsi untuk menjadi perwakilan titik vektor centroid badan
    if data["pose_landmarks"] is None and glob_bbox is None: 
        ### kalau pose tidak ditemukan dan bounding box belum ada
        data["centroid"] = [0, 0, 0]
    elif data["pose_landmarks"] is None and glob_bbox is not None: 
        ### kalau pose tidak ditemukan dan bonding box sudah dibentuk maka yang jadi centroid adalah bounding box frame terakhir
        data["centroid"] = glob_bbox
    else: #### kalau pose ditemukan
        data["centroid"] = find_body_centroid(data["pose_landmarks"], centroid_indices)

    if data["pose_landmarks"] is None and glob_dist is not None:
        ### kalau pose tidak ditemukan dan jarak kamera sudah terisi
        ### digunakan untuk tetap mendapatkan nilai radius walaupun pose tidak terdeteksi. namun wajah atau tangan masih terdeteksi.
        # data["radius"] = glob_dist ## digunakan jika ada bantuan deteksi pose oleh tangan atau wajah
        data["shoulders_centroid"] = [0, 0, 0]
        data["hips_centroid"] = [0, 0, 0]
    else: ### kalau pose ditemukan
        centroid_a, centroid_b = find_centroid(data["pose_landmarks"], data) ## dapatkan centroid bahu dan centroid pinggang
        data["radius"] = euclidean(centroid_a, centroid_b) # radius jarak objek terhadap bounding box
        data["shoulders_centroid"] = centroid_a ## centroid bahu
        data["hips_centroid"] = centroid_b ## centroid pinggang
    return data    


def normalize(holistic, mp_holistic, image):
    mp_drawing = mp.solutions.drawing_utils
    result = holistic.process(image)
    original = []
    
    params = {"pose_landmarks": result.pose_landmarks,
              "image": image, 
              "result": result, 
              "mp_drawing": mp_drawing, 
              }
    
    if result.left_hand_landmarks:
        for res in result.left_hand_landmarks.landmark:
            original.append(res.x)
            original.append(res.y)
            original.append(res.z)
    else:
        for i in range(0, 21):
            for i in range(0, 3):
                original.append(0)
    
    if result.right_hand_landmarks:
        for res in result.right_hand_landmarks.landmark:
            original.append(res.x)
            original.append(res.y)
            original.append(res.z)
    else:
        for i in range(0, 21):
            for i in range(0, 3):
                original.append(0)

    if result.pose_landmarks:
        for res in result.pose_landmarks.landmark:
            original.append(res.x)
            original.append(res.y)
            original.append(res.z)
    else:
        for i in range(0, 33):
            for i in range(0, 3):
                original.append(0)
        
    params = keypoints_check(params)
    draw_landmarks(image,params,"ori")
    # image.flags.writeable = True   
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    coordinates = final_extract(params)
  

    return coordinates

def create_transformer_model(sequence_length, num_features, num_classes, num_layers, hidden_units, num_heads, dropout_rate):
    # Input Layer
    inputs = layers.Input(shape=(sequence_length, num_features))
    x = inputs

    # Positional Encoding Layer
    positional_encoding = layers.Embedding(input_dim=sequence_length, output_dim=num_features)(tf.range(sequence_length))
    x += positional_encoding

    # Encoder Layers
    for _ in range(num_layers):
        # Multi-Head Attention
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_features)(x, x)
        attention = layers.Dropout(rate=dropout_rate)(attention)
        attention = layers.LayerNormalization(epsilon=1e-6)(attention + x)

        # Feed Forward Neural Network
        ffn = layers.Dense(units=hidden_units, activation='relu')(attention)
        ffn = layers.Dropout(rate=dropout_rate)(ffn)
        ffn = layers.LayerNormalization(epsilon=1e-6)(ffn + attention)

        x = ffn

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Output Layer
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)

    # Create and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def init_model():
    init_array = np.zeros((120,225))
    model = create_transformer_model(init_array.shape[0],init_array.shape[1],50,2,225,4,0.2)
    model.load_weights('Tr_FullData_Layer_50_Epoch_3.h5')
    return model

def shuffle_soal(array):
    random_soal = random.choice(array)
    return random_soal

def run_detection (model,array):
    sequence = []
    sentence = []
    predictions = []
    soal = []
    threshold = 0.5
    sequence_length = 121
    soal.append(shuffle_soal(array))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.5) as holistic:
        
        for frame_num in range(sequence_length):

            # Read feed
                        ret, frame = cap.read()

                        # Make detections
                        coordinates = normalize(holistic, mp_holistic, frame)

                        # NEW Apply wait logic
                        if frame_num == 0: 
                            cv2.rectangle(frame, (0,0), (800, 40), (255,255,255), -1)
                            text = 'BERSIAP'
                            x, y = 250, 250
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1
                            font_thickness = 4
                            outline_thickness = 8
                            text_color = (0, 0, 0)  # Black color
                            outline_color = (255, 255, 255)  # White color

                            cv2.rectangle(frame, (0,0), (800, 40), (255,255,255), -1)
                            cv2.putText(frame, 'Soal :' ,(200,30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)
                            cv2.putText(frame, ''.join(soal[-1:]), (300,30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)

                            # Draw the text outline
                            (text_width, text_height) = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                            cv2.putText(frame, text, (x, y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)

                            # Draw the main text
                            cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', cv2.resize(frame, (800, 600)))
                            cv2.waitKey(2500)
                        else: 
                            cv2.rectangle(frame, (0,0), (800, 40), (255,255,255), -1)
                            cv2.putText(frame, 'Soal :' ,(200,30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)
                            cv2.putText(frame, ''.join(soal[-1:]), (300,30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)
                            sequence.append(coordinates)
                            cv2.imshow('OpenCV Feed', cv2.resize(frame, (800, 600)))
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
        cap.release()
        cv2.destroyAllWindows()
        # Open a new window after the main loop
        #Predict and Detect
        if len(sequence) == 120:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
                
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

        new_window = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Create a white image
        if sentence[-1:] == soal[-1:]:
            cv2.putText(new_window, 'Evaluasi : Benar' ,(250,270), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

        elif sentence[-1:] != soal[-1:]:
            cv2.putText(new_window, 'Evaluasi : Salah' ,(250,270), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(new_window, 'Soal :' ,(250,230), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(new_window, 'Prediksi :' ,(250,250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(new_window, ''.join(soal[-1:]), (325,230), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(new_window, ''.join(sentence), (375,250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)
        cv2.imshow('New Window', cv2.resize(new_window, (800, 600)))
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()


def run_endless_detection(model,array):
    sequence = []
    sentence = []
    predictions = []
    soal = []
    threshold = 0.5
    soal.append(shuffle_soal(array))
    frame_counter = 0  # Initialize the frame counter variable

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            
            # Show to screen
            coordinates = normalize(holistic, mp_holistic, frame) 
            sequence.append(coordinates)
            sequence = sequence[-120:]
            
            #Predict and Detect
            if len(sequence) == 120:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 1: 
                    sentence = sentence[-1:]

            cv2.rectangle(frame, (0,0), (800, 40), (255,255,255), -1)

            cv2.putText(frame, 'Prediksi :' ,(185,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(frame, ''.join(sentence), (305,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1, cv2.LINE_AA)
            
            
            cv2.imshow('OpenCV Feed', cv2.resize(frame, (800, 600)))

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()