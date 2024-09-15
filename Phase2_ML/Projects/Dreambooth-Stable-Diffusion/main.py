import os
import sys
import tkinter as tk
import tkinter.font as font
from genImage import genImg
from PIL import ImageTk, Image
from prompts import DEFAULT_PROMPTS
from tkinter.filedialog import askopenfilename
from util import _from_rgb, generate_random_string, get_custom_prompts

MODEL_DIR = './models/'
os.makedirs(MODEL_DIR, exist_ok=True)

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("{}x{}".format(self.main_window.winfo_screenwidth(),
                                                 self.main_window.winfo_screenheight()))
        self.main_window.configure(bg="gray")

        self.side_bar_label = tk.Label(self.main_window, width=50, height=50,
                                       bg=_from_rgb((100, 100, 100)))
        self.side_bar_label.place(x=150, y=90)
        self.create_side_bar_widgets()

        self.main_section_label = tk.Label(self.main_window, width=150, height=50, bg='gray')
        self.main_section_label.place(x=600, y=90)
        self.main_img_label = tk.Label(self.main_window, width=500, height=500, bg='gray')

    def display_selected_model(self, _=None):
        self.selected_model = self.selected_model_.get()

    def display_selected_style(self, _=None):
        self.selected_style = self.selected_style_.get()
        if self.selected_style == 'CUSTOM PROMPT':
            self.custom_prompt_text_box.config(state="normal")
        else:
            self.custom_prompt_text_box.config(state="disabled")

    def refresh(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def create_main_section_image_view(self):
        self.main_img_label = tk.Label(self.main_section_label, bg='gray')
        self.img_00_label = tk.Label(self.main_section_label, bg='gray')
        self.img_01_label = tk.Label(self.main_section_label, bg='gray')
        self.img_02_label = tk.Label(self.main_section_label, bg='gray')
        self.img_03_label = tk.Label(self.main_section_label, bg='gray')
        self.img_04_label = tk.Label(self.main_section_label, bg='gray')
        self.img_05_label = tk.Label(self.main_section_label, bg='gray')
        self.img_06_label = tk.Label(self.main_section_label, bg='gray')
        self.img_07_label = tk.Label(self.main_section_label, bg='gray')
        self.save_img_button = tk.Button(self.main_section_label, height=1, width=12, text="SAVE IMAGE", bg='gray',
                                         font=font.Font(size=30), command=self.save_main_img)

    def create_side_bar_widgets(self):

        ###########################################################
        ### train new model #######################################
        ###########################################################

        self.train_new_model_button = tk.Button(self.side_bar_label, text='Train new model', height=1, width=15,
                                                bg='gray', command=self.train_new_model)
        self.train_new_model_button.place(x=30, y=140)

        ###########################################################
        ### select model dropdown menu ############################
        ###########################################################

        self.select_model_label = tk.Label(self.side_bar_label, text='Select model:',
                                           font=("Arial", 25),
                                           bg=_from_rgb((100, 100, 100)), fg='white')
        self.select_model_label.place(x=30, y=50)

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
        models = [j for j in os.listdir(MODEL_DIR)]
        models = ['                                         '] + models
        self.selected_model_ = tk.StringVar()
        self.selected_model_.set(models[0])
        self.select_model_dropdown = tk.OptionMenu(
            self.side_bar_label,
            self.selected_model_,
            *models,
            command=self.display_selected_model
        )
        self.select_model_dropdown.place(x=30, y=100)

        ###########################################################
        ### select style dropdown menu ############################
        ###########################################################

        self.select_style_label = tk.Label(self.side_bar_label, text='Select style:',
                                           font=("Arial", 25),
                                           bg=_from_rgb((100, 100, 100)), fg='white')
        self.select_style_label.place(x=30, y=200)

        custom_prompts_ = get_custom_prompts()
        styles = ['                                         '] + list(DEFAULT_PROMPTS.keys()) + \
                 list(custom_prompts_.keys()) + ['CUSTOM PROMPT']

        self.selected_style_ = tk.StringVar()
        self.selected_style_.set(styles[0])
        self.select_style_dropdown = tk.OptionMenu(
            self.side_bar_label,
            self.selected_style_,
            *styles,
            command=self.display_selected_style
        )
        self.select_style_dropdown.place(x=30, y=250)

        ###########################################################
        ### custom prompt text box ################################
        ###########################################################

        self.prompt_label = tk.Label(self.side_bar_label, text='Prompt:',
                                     font=("Arial", 25),
                                     bg=_from_rgb((100, 100, 100)),
                                     fg='white')
        self.prompt_label.place(x=30, y=350)

        self.custom_prompt_text_box = tk.Text(self.side_bar_label, height=6, width=25, font=font.Font(size=15))
        self.custom_prompt_text_box.config(state="disabled", bg=_from_rgb((220, 220, 220)), fg="black")
        self.custom_prompt_text_box.place(x=30, y=400)

        ###########################################################
        ### 'GENERATE' button #####################################
        ###########################################################

        self.generate_button = tk.Button(self.side_bar_label, height=3, width=12, text="GENERATE", bg='gray',
                                         font=font.Font(size=30), command=self.generate_images)
        self.generate_button.place(x=34, y=630)

    def open_file_dialog_box(self):
        self.file_to_upload = askopenfilename()

        self.display_filename = tk.Label(self.main_section_label, text=self.file_to_upload.split(os.sep)[-1],
                                         font=("Arial", 14), bg='gray')
        self.display_filename.place(x=190, y=245)

    def train_new_model(self):

        for att in ['main_img_label', 'img_00_label', 'img_01_label', 'img_02_label', 'img_03_label', 'img_04_label',
                    'img_05_label', 'img_06_label', 'img_07_label', 'save_img_button', 'move_left_button',
                    'move_right_button']:
            if att in self.__dict__.keys():
                self.__getattribute__(att).place_forget()

        self.training_data_label = tk.Label(self.main_section_label, text='Training data', font=("Arial", 24), bg='gray')
        self.training_data_label.place(x=180, y=170)

        self.browse_training_data_button = tk.Button(self.main_section_label, text='Select dataset', height=1, width=12,
                                                     font=font.Font(size=15), bg='gray', command=self.open_file_dialog_box)
        self.browse_training_data_button.place(x=180, y=230)

        self.start_training_button = tk.Button(self.main_section_label, text='Start training', height=1, width=12,
                                               font=font.Font(size=15), bg='gray', command=self.start_training)
        self.start_training_button.place(x=180, y=280)

        self.training_output_label = tk.Label(self.main_section_label, text='', font=("Arial", 12), bg='gray')
        self.training_output_label.place(x=180, y=450)

    def upload_dataset(self):
        dataset_path = self.file_to_upload
        self.training_output_label.config(text=f"Dataset '{dataset_path}' uploaded!")

    def start_training(self):
        dataset_path = self.file_to_upload

        model_name = generate_random_string(5)
        output_dir = os.path.join(MODEL_DIR, model_name)

        # Run DreamBooth training script locally
        training_command = f"""
        accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
        --instance_data_dir="PicsTrain" \
        --output_dir={output_dir} \
        --instance_prompt="a photo of KSJD, a man looking in different directions" \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=1 \
        --learning_rate=2e-6 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=1500
        """
        os.system(training_command)

        self.training_output_label.config(text=f"Model '{model_name}' trained and saved in {output_dir}!")

    def generate_images(self):
        selected_model = self.selected_model_.get().strip()
        selected_style = self.selected_style_.get().strip()

        if selected_model == '' or selected_style == '':
            return

        if selected_style == 'CUSTOM PROMPT':
            prompt = self.custom_prompt_text_box.get("1.0", tk.END).strip()
        else:
            prompt = DEFAULT_PROMPTS.get(selected_style, "a beautiful image")

        output_path = f"./genImages/{selected_model}_{selected_style}.png"
        selectedModel = f"./models/{selected_model}"

        # Generation command using DreamBooth locally
        if(not os.path.exists(output_path)):
            genImg(prompt, output_path, selectedModel)

        # Display the generated images
        self.load_generated_images(output_path)

    def load_generated_images(self, output_path):
        # Load the image from the file path
        image = Image.open(output_path)

        image = image.resize((512, 512), Image.Resampling.LANCZOS)

        self.main_image = ImageTk.PhotoImage(image)

        self.main_img_label.config(image=self.main_image)

        self.main_img_label.place(x=850, y=300)

if __name__ == "__main__":
    app = App()
    app.main_window.mainloop()
