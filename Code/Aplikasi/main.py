import tkinter as tk
import customtkinter
import cv2
import numpy as np
import mediapipe as mp
import video as vid

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_widget_scaling(1.25)

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("LearnSIBI")
        self.geometry("800x600")
        self.model = vid.init_model()
        self.mp_holistic = mp.solutions.holistic # Holistic Keypoints : Pose Tubuh dan Tangan
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing = mp.solutions.drawing_utils # Utilitas menggambar
        self.actions = (['Akan', 'Anda', 'Apa', 'Atau', 'Baca', 'Bagaimana', 'Bahwa', 'Beberapa','Besar',
                    'Bisa','Buah','Dan' ,'Dari' ,'Dengan','Dia','Haus', 'Ingin', 'Ini' ,'Itu',
                     'Jadi', 'Juga' ,'Kami' ,'Kata' ,'Kecil', 'Kumpul' ,'Labuh', 'Lain' ,'Laku',
                     'Lapar', 'Main', 'Makan', 'Masing', 'Mereka', 'Milik', 'Minum', 'Oleh', 'Pada',
                     'Rumah', 'Satu', 'Saya', 'Sebagai', 'Tambah', 'Tangan' ,'Tetapi', 'Tidak', 'Tiga',
                     'Udara', 'Untuk', 'Waktu', 'Yang'])
        self.selected_actions_1 = np.array(['Makan'])
        self.selected_actions_2 = np.array(['Buah', 'Dan', 'Dari', 'Dengan', 'Dia', 'Haus', 'Ingin', 'Ini', 'Itu','Jadi'])
        self.selected_actions_3 = np.array(['Juga', 'Kami', 'Kata', 'Kecil', 'Kumpul', 'Labuh', 'Lain', 'Laku', 'Lapar', 'Main'])
        self.selected_actions_4 = np.array(['Makan', 'Masing', 'Mereka', 'Milik', 'Minum', 'Oleh', 'Pada','Rumah', 'Satu', 'Saya'])
        self.selected_actions_5 = np.array(['Sebagai', 'Tambah', 'Tangan', 'Tetapi', 'Tidak', 'Tiga','Udara', 'Untuk', 'Waktu', 'Yang'])
        
        # configure grid layout (4x4)
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.button_frame =  customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.button_frame.grid(row=2, column=1, rowspan=4, sticky="nsew")
        self.button_frame.pack(pady=10)
        
        self.handle_MainMenu()
        
    def handle_MainMenu(self):
        self.clear_buttons()
        self.create_title('Main Menu')
        self.create_buttons('Evaluasi',self.handle_Evaluasi)
        self.create_buttons('Belajar',self.handle_Belajar)
        self.create_buttons('Exit',self.handle_Exit)

    def handle_Exit(self):
        self.destroy()


    def handle_Evaluasi(self):
        self.clear_buttons()
        self.create_title('Modul Evaluasi')
        self.create_buttons('Soal 1-10',commands=lambda: self.handle_detection(1))
        self.create_buttons('Soal 11-20',commands=lambda: self.handle_detection(2))
        self.create_buttons('Soal 21-30',commands=lambda: self.handle_detection(3))
        self.create_buttons('Soal 31-40',commands=lambda: self.handle_detection(4))
        self.create_buttons('Soal 41-50',commands=lambda: self.handle_detection(5))
        self.create_buttons('Semua Soal',commands=lambda: self.handle_detection(6))
        self.create_buttons('Latihan',commands=lambda: self.handle_detection(0))
        self.create_buttons('Main Menu',self.handle_MainMenu)


    def handle_Belajar(self):
        self.clear_buttons()
        self.create_title('Modul Belajar')
        self.create_belajar_buttons()
        self.create_buttons('Main Menu',self.handle_MainMenu)

    def handle_detection(self,options):
        # print(self.model)
        if options == 1:
            vid.run_detection(self.model,self.selected_actions_1)
        if options == 2:
            vid.run_detection(self.model,self.selected_actions_2)
        if options == 3:
            vid.run_detection(self.model,self.selected_actions_3)
        if options == 4:
            vid.run_detection(self.model,self.selected_actions_4)
        if options == 5:
            vid.run_detection(self.model,self.selected_actions_5)
        if options == 6:
            vid.run_detection(self.model,self.actions)
        if options == 0:
            vid.run_endless_detection(self.model,self.actions)  

    def create_belajar_buttons(self):

        num_buttons = 50
        max_buttons_per_row = 1
        num_rows = (num_buttons + max_buttons_per_row - 1) // max_buttons_per_row
        scroll_frame = customtkinter.CTkScrollableFrame(self.button_frame)
        scroll_frame.pack()
        for row in range(num_rows):
            button_frame = customtkinter.CTkFrame(scroll_frame)
            button_frame.pack()

            for col in range(max_buttons_per_row):
                index = row * max_buttons_per_row + col
                if index < num_buttons:
                    button_text = str(self.actions[index])
                    button_command = lambda index=index: self.handle_video(f'./videos/'+str(self.actions[index])+'.mp4', window_width=1280, window_height=720)
                    self.create_buttons_in_frame(button_frame, button_text, button_command)

    def create_buttons_in_frame(self, frame, name, command):

        button = customtkinter.CTkButton(frame, text=name, font=customtkinter.CTkFont(size=20, weight="bold"), command=command)
        button.pack(side=tk.LEFT, padx=10, pady=10)


    def create_buttons(self,name,commands):
        button = customtkinter.CTkButton(self.button_frame, text=name,font=customtkinter.CTkFont(size=20, weight="bold"), command=commands)
        button.pack(padx=20, pady=10)

    def create_title(self,name):
        text = customtkinter.CTkLabel(self.button_frame, text=name,font=customtkinter.CTkFont(size=30, weight="bold"))
        text.pack(pady=10)

    def handle_video(self,file_path, window_width=800, window_height=600):
        # Load the video file
        video = cv2.VideoCapture(file_path)

        # Check if video file is opened successfully
        if not video.isOpened():
            print("Error opening video file")
            return

        # Get the frames per second (fps) of the video
        fps = video.get(cv2.CAP_PROP_FPS)

        # Set the desired window width and height
        cv2.namedWindow('Video Player', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video Player', window_width, window_height)

        while True:
            # Read the current frame from the video
            ret, frame = video.read()

            if not ret:
                # End of video
                print("Video playback finished")
                break
            # Display the frame in the window named 'Video Player'
            cv2.imshow('Video Player', frame)
            # Check if the 'q' key is pressed
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

        # Release the video object and close all windows
        video.release()
        cv2.destroyAllWindows()

        # Close the window if it's still open
        if cv2.getWindowProperty('Video Player', cv2.WND_PROP_VISIBLE) > 0:
            cv2.waitKey(1)
            cv2.destroyWindow('Video Player')
    
    def clear_buttons(self):
        for widget in self.button_frame.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()