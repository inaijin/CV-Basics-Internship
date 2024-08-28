import cv2
import utils
import pickle
import os.path
import datetime
import tkinter as tk
import face_recognition
from PIL import Image, ImageTk

class App:
    def __init__(self):
        self.mainWindow = tk.Tk()
        self.mainWindow.title("Face Recognition")
        self.mainWindow.geometry("1200x520+350+100")

        # Buttons for login and register
        self.loginButtonMainWindow = utils.get_button(self.mainWindow, 'login', 'green', self.login)
        self.loginButtonMainWindow.place(x=820, y=225)

        self.registerNewUserButtonMainWindow = utils.get_button(self.mainWindow, 'register new user', 'gray', self.registerNewUser, fg='black')
        self.registerNewUserButtonMainWindow.place(x=820, y=275)

        # Webcam label
        self.webcamLabel = utils.get_img_label(self.mainWindow)
        self.webcamLabel.place(x=10, y=0, width=700, height=500)

        self.addWebcam(self.webcamLabel)

        # Directory and log path setup
        self.dbDir = './db'
        if not os.path.exists(self.dbDir):
            os.mkdir(self.dbDir)

        self.logPath = './log.txt'

    def addWebcam(self, label):
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.processWebcam()

    def processWebcam(self):
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            _, frame = self.cap.read()

            if frame is not None:
                self.mostRecentCaptureArr = frame
                img_ = cv2.cvtColor(self.mostRecentCaptureArr, cv2.COLOR_BGR2RGB)
                self.mostRecentCapturePil = Image.fromarray(img_)
                imgtk = ImageTk.PhotoImage(image=self.mostRecentCapturePil)
                self._label.imgtk = imgtk
                self._label.configure(image=imgtk)
            else:
                print("Failed to capture frame from webcam")

            self._label.after(20, self.processWebcam)
        else:
            pass

    def login(self):
        name = utils.recognize(self.mostRecentCaptureArr, self.dbDir)

        if name in ['unknown_person', 'no_persons_found']:
            utils.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            utils.msg_box('Welcome back!', 'Welcome, {}.'.format(name))
            with open(self.logPath, 'a') as f:
                f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                f.close()

            # Stop the webcam feed
            self.stopWebcam()

            # Hide login and register buttons
            self.loginButtonMainWindow.place_forget()
            self.registerNewUserButtonMainWindow.place_forget()

            # Display user's image
            self.display_user_image()

            # Show logout button
            self.logoutButtonMainWindow = utils.get_button(self.mainWindow, 'Logout', 'red', self.logout)
            self.logoutButtonMainWindow.place(x=820, y=250)

    def display_user_image(self):
        imgtk = ImageTk.PhotoImage(image=self.mostRecentCapturePil)
        self.webcamLabel.imgtk = imgtk
        self.webcamLabel.configure(image=imgtk)

    def logout(self):
        name = utils.recognize(self.mostRecentCaptureArr, self.dbDir)

        if name in ['unknown_person', 'no_persons_found']:
            utils.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            utils.msg_box('Goodbye!', 'Goodbye, {}.'.format(name))
            with open(self.logPath, 'a') as f:
                f.write('{},{},out\n'.format(name, datetime.datetime.now()))
                f.close()

            # Reset to initial state: restart the webcam feed
            self.addWebcam(self.webcamLabel)

            # Remove logout button
            self.logoutButtonMainWindow.place_forget()

            # Show login and register buttons again
            self.loginButtonMainWindow.place(x=820, y=225)
            self.registerNewUserButtonMainWindow.place(x=820, y=275)

    def registerNewUser(self):
        self.registerNewUserWindow = tk.Toplevel(self.mainWindow)
        self.registerNewUserWindow.geometry("1200x520+370+120")

        self.acceptButtonRegisterNewUserWindow = utils.get_button(self.registerNewUserWindow, 'Accept', 'green', self.acceptRegisterNewUser)
        self.acceptButtonRegisterNewUserWindow.place(x=750, y=300)

        self.tryAgainButtonRegisterNewUserWindow = utils.get_button(self.registerNewUserWindow, 'Try again', 'red', self.tryAgainRegisterNewUser)
        self.tryAgainButtonRegisterNewUserWindow.place(x=750, y=400)

        self.captureLabel = utils.get_img_label(self.registerNewUserWindow)
        self.captureLabel.place(x=10, y=0, width=700, height=500)

        self.addImgToLabel(self.captureLabel)

        self.entryTextRegisterNewUser = utils.get_entry_text(self.registerNewUserWindow)
        self.entryTextRegisterNewUser.place(x=750, y=150)

        self.textLabelRegisterNewUser = utils.get_text_label(self.registerNewUserWindow, 'Please, \ninput username:')
        self.textLabelRegisterNewUser.place(x=750, y=70)

    def tryAgainRegisterNewUser(self):
        self.registerNewUserWindow.destroy()

    def addImgToLabel(self, label):
        imgtk = ImageTk.PhotoImage(image=self.mostRecentCapturePil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.registerNewUserCapture = self.mostRecentCaptureArr.copy()

    def start(self):
        self.mainWindow.mainloop()

    def stopWebcam(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def acceptRegisterNewUser(self):
        name = self.entryTextRegisterNewUser.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.registerNewUserCapture)[0]

        file = open(os.path.join(self.dbDir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        utils.msg_box('Success!', 'User was registered successfully!')

        self.registerNewUserWindow.destroy()

if __name__ == "__main__":
    app = App()
    app.start()
