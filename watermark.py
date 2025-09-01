"""
Author: Joel Foster, 2025
"""

import os
import threading
import cv2
import numpy as np
from customtkinter import *
from customtkinter import ctk_tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError

class Watermark(CTk):
    """Watermark Creation, Retrieval, and Tampering Dection App
    """

    NORMAL = 0
    ERROR = 1

    def __init__(self):
        super().__init__()

        # constants
        self.WIDTH, self.HEIGHT = 1300, 800
        self.CANVAS_WIDTH, self.CANVAS_HEIGHT = 300, 300

        # set initial width, title, and grid layout
        self.title("Watermark Creation, Retrieval and Tampering Detection Tool")
        self.geometry(f'{self.WIDTH}x{self.HEIGHT}')
        self.columnconfigure(0, weight=1)
        self.rowconfigure((0,1), weight=1)

        # variables for images as cv2 images
        self.carrier_image = None
        self.watermark_image = None
        self.output_image = None

        # image upload and display frame
        self.images_frame = CTkFrame(self)
        self.images_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nesw")
        self.images_frame.columnconfigure((0,1,2), weight=1)
        self.images_frame.rowconfigure(0, weight=3)
        self.images_frame.rowconfigure(1, weight=1)

        # carrier image upload and display
        self.carrier_canvas = CTkCanvas(self.images_frame, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="gray")
        self.carrier_canvas.grid(row=0, column=0, padx=20, pady=20, sticky="nesw")

        self.carrier_label_frame = CTkFrame(self.images_frame, fg_color="transparent")
        self.carrier_label_frame.grid(row=1, column=0, sticky="nesw")
        self.carrier_label_frame.rowconfigure((0,1), weight=1)
        self.carrier_label_frame.columnconfigure(0, weight=1)

        self.carrier_label = CTkLabel(self.carrier_label_frame, text="Carrier/watermarked image")
        self.carrier_label.grid(row=0, column=0, sticky="n")

        self.upload_carrier_button = CTkButton(self.carrier_label_frame,
                                               text="Upload carrier/watermarked image",
                                               command=lambda: self.import_image(self.carrier_canvas))
        self.upload_carrier_button.grid(row=1, column=0, sticky="n")

        # watermark image upload and display
        self.watermark_canvas = CTkCanvas(self.images_frame, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="gray")
        self.watermark_canvas.grid(row=0, column=1, padx=20, pady=20, sticky="nesw")

        # watermark label and upload button
        self.watermark_label_frame = CTkFrame(self.images_frame, fg_color="transparent")
        self.watermark_label_frame.grid(row=1, column=1, sticky="nesw")
        self.watermark_label_frame.rowconfigure((0,1), weight=1)
        self.watermark_label_frame.columnconfigure(0, weight=1)

        self.watermark_label = CTkLabel(self.watermark_label_frame, text="Watermark image")
        self.watermark_label.grid(row=0, column=0, sticky="n")

        self.upload_watermark_button = CTkButton(self.watermark_label_frame,
                                                 text="Upload watermark image",
                                                 command=lambda: self.import_image(self.watermark_canvas))
        self.upload_watermark_button.grid(row=1, column=0, sticky="n")

        # output image display
        self.output_canvas = CTkCanvas(self.images_frame, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="gray")
        self.output_canvas.grid(row=0, column=2, padx=20, pady=20, sticky="nesw")

        # output label and save button
        self.output_label_frame = CTkFrame(self.images_frame, fg_color="transparent")
        self.output_label_frame.grid(row=1, column=2, sticky="nesw")
        self.output_label_frame.rowconfigure((0,1), weight=1)
        self.output_label_frame.columnconfigure(0, weight=1)

        self.output_label = CTkLabel(self.output_label_frame, text="Output image")
        self.output_label.grid(row=0, column=0, sticky="n")

        self.save_output_button = CTkButton(self.output_label_frame,
                                            text="Save output image",
                                            command=lambda: self.save_image(self.output_image))
        self.save_output_button.grid(row=1, column=0, sticky="n")

        # add event handler to resize image previews on resize of the app
        self.resizing = None
        self.bind('<Configure>', self.on_resize)

        # program controls frame
        self.controls_frame = CTkFrame(self)
        self.controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nesw")
        self.controls_frame.rowconfigure(0, weight=1)
        self.controls_frame.columnconfigure(0, weight=1)
        self.controls_frame.columnconfigure((1,2), weight=2)

        # embed and recover watermark frame
        self.embed_recover_frame = CTkFrame(self.controls_frame)
        self.embed_recover_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nesw")
        # self.embed_recover_frame.rowconfigure((0,1,2), weight=1)
        # self.embed_recover_frame.columnconfigure(0, weight=1)

        # embedding frame
        self.embed_frame = CTkFrame(self.embed_recover_frame)
        self.embed_frame.pack(fill="x", padx=10, pady=(10,5))
        self.embed_frame.rowconfigure((0,1), weight=1)
        self.embed_frame.columnconfigure((0,1), weight=1)

        self.embed_watermark_label = CTkLabel(self.embed_frame, text="Embed watermark:")
        self.embed_watermark_label.grid(row=0, column=0, padx=5, sticky="nw", columnspan=2)

        self.embed_watermark_button = CTkButton(self.embed_frame,
                                                text="Embed watermark",
                                                command=self.embed_watermark)
        self.embed_watermark_button.grid(row=1, column=0, pady=10)

        self.default_watermark_button = CTkButton(self.embed_frame,
                                                  text="Use default watermark",
                                                  command=self.default_watermark)
        self.default_watermark_button.grid(row=1, column=1, pady=10)

        # recovery frame
        self.recovery_frame = CTkFrame(self.embed_recover_frame)
        self.recovery_frame.pack(fill="x", padx=10, pady=5)
        self.recovery_frame.rowconfigure((0,1), weight=1)
        self.recovery_frame.columnconfigure((0,1), weight=1)

        self.recovery_label = CTkLabel(self.recovery_frame, text="Watermark Recovery and Tampering Detection:")
        self.recovery_label.grid(row=0, column=0, padx=5, sticky="nw", columnspan=2)

        self.verify_authenticity_button = CTkButton(self.recovery_frame,
                                                  text="Verify authenticity",
                                                  command=self.recover_watermark)
        self.verify_authenticity_button.grid(row=1, column=0, pady=10)

        self.detect_tampering_button = CTkButton(self.recovery_frame,
                                                 text="Detect tampering",
                                                 command=self.detect_tampering)
        self.detect_tampering_button.grid(row=1, column=1, pady=10)

        # utility frame
        self.utility_frame = CTkFrame(self.embed_recover_frame)
        self.utility_frame.pack(fill="x", padx=10, pady=5)
        self.utility_frame.rowconfigure((0,1), weight=1)
        self.utility_frame.columnconfigure((0,1), weight=1)

        self.utility_label = CTkLabel(self.utility_frame, text="Utilities:")
        self.utility_label.grid(row=0, column=0, padx=5, sticky="nw")

        self.clear_images_button = CTkButton(self.utility_frame, text='Clear images', command=self.clear_images)
        self.clear_images_button.grid(row=1, column=0, pady=10)

        self.clear_output_button = CTkButton(self.utility_frame, text="Clear output console", command=self.clear_output)
        self.clear_output_button.grid(row=1, column=1, pady=10)

        # progress bar
        self.progress_bar = CTkProgressBar(self.embed_recover_frame, orientation="horizontal")
        self.progress_bar.pack(side=BOTTOM, fill=X, padx=20, pady=(5, 10))
        self.progress_bar.set(0)
        self.progress_bar.configure(mode="indeterminate")

        # keypoint visualiser frame
        self.keypoint_frame = CTkFrame(self.controls_frame)
        self.keypoint_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nesw")
        self.keypoint_frame.rowconfigure(0, weight=1)
        self.keypoint_frame.rowconfigure(1, weight=6)
        self.keypoint_frame.columnconfigure(0, weight=1)

        # keypoints label and canvas
        self.keypoint_label = CTkLabel(self.keypoint_frame, text="Keypoints")
        self.keypoint_label.grid(row=0, column=0, padx=10, sticky="sw")

        self.keypoint_canvas = CTkCanvas(self.keypoint_frame, bg="gray")
        self.keypoint_canvas.grid(row=1, column=0, padx=10, pady=10, sticky="nesw")

        # output console frame
        self.output_console_frame = CTkFrame(self.controls_frame)
        self.output_console_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nesw")
        self.output_console_frame.rowconfigure(0, weight=1)
        self.output_console_frame.rowconfigure(1, weight=6)
        self.output_console_frame.columnconfigure(0, weight=1)

        # output console label and text box
        self.output_console_label = CTkLabel(self.output_console_frame, text="Output:")
        self.output_console_label.grid(row=0, column=0, padx=10, sticky="sw")

        self.output_console = CTkTextbox(self.output_console_frame, wrap=WORD)
        self.output_console.configure(state="disabled")
        self.output_console.grid(row=1, column=0, padx=10, pady=10, sticky="nesw")
        self.output_console.tag_config('normal', foreground="white")
        self.output_console.tag_config('error', foreground="red")

        self.output_print('Welcome to the Watermark Creation, Retrieval and Tampering Detection Tool')


    def on_resize(self, e):
        """
        Callback function for dealing with image resizing on program window resize
        """
        # prevents callback resizing images too often, causing program to lag
        if self.resizing:
            self.after_cancel(self.resizing) # cancels previous call to resize if new resize is called within 100ms
        self.resizing = self.after(100, self.on_resize_done) # calls resize function if no interrupts for 100ms

    def on_resize_done(self):
        """
        Resizes preview images after window has finished being resized
        """
        # only attempt to redraw if canvas has image added
        if hasattr(self.carrier_canvas, 'img'):
            self.redraw_image(self.carrier_canvas.img, self.carrier_canvas)
        if hasattr(self.watermark_canvas, 'img'):
            self.redraw_image(self.watermark_canvas.img, self.watermark_canvas)
        if hasattr(self.output_canvas, 'img'):
            self.redraw_image(self.output_canvas.img, self.output_canvas)
        if hasattr(self.keypoint_canvas, 'img'):
            self.redraw_image(self.keypoint_canvas.img, self.keypoint_canvas)
        self.resizing = None

    def output_print(self, message='', type=NORMAL):
        """
        Print message to output box
        """
        self.output_console.configure(state="normal")
        if type == self.NORMAL:
            self.output_console.insert(index=END, text=message + '\n', tags='normal')
        elif type == self.ERROR:
            self.output_console.insert(index=END, text='ERROR: ' + message + '\n', tags='error')
        self.output_console.configure(state="disabled")
        self.output_console.see(END)


    def add_image(self, image, canvas):
        """Takes an uploaded image and displays it in a canvas window.

        Parameters
        ----------
        image: ~PIL.Image.Image
            Image to add to canvas
        canvas: CTkCanvas
            Canvas to add image to
        """
        img_width, img_height = image.size

        # update layout before getting real rendered size of canvas
        self.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # get difference in canvas width and image width to use as scales
        width_scale = canvas_width / img_width
        height_scale = canvas_height / img_height

        img_width *= width_scale
        img_height *= height_scale

        image = image.resize((int(img_width), int(img_height)), Image.LANCZOS)

        img = ImageTk.PhotoImage(image=image)
        canvas.img = img
        canvas.create_image(canvas_width/2, canvas_height/2, image=img, anchor=CENTER)

    def redraw_image(self, image, canvas):
        """Redraws an image on its canvas

        Parameters
        ----------
        image: ~PIL.Image.Image
            Image to redraw on canvas
        canvas: CTkCanvas
            Canvas to redraw image on
        """
        image = ImageTk.getimage(image)
        self.add_image(image, canvas)

    def import_image(self, canvas):
        """Imports an image from a chosen file into the program

        Parameters
        ----------
        canvas: CTkCanvas
            Canvas to load the image onto

        Raises
        ------
        UnidentifiedImageError
            If uploaded image cannot be read into an image
        """
        try:
            file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("Image files", "*.png"), ("All files", "*.*")])
            if file_path:
                valid = False
                image = Image.open(file_path)

                if not file_path.endswith('.png'):
                    self.output_print('Image must be a .png file! Please upload another image', self.ERROR)
                elif canvas == self.watermark_canvas:
                    if image.size[0] != image.size[1]: # check if watermark is a square image
                        self.output_print('Watermark must be a square image! Please upload another image', self.ERROR)
                    elif image.size[0] > 16:
                        self.output_print('Maximum watermark size is 16x16! Please upload another image', self.ERROR)
                    else:
                        valid = True
                else:
                    valid = True

                if valid:
                    self.add_image(image, canvas) # add image to gui

                    _, file_name = os.path.split(file_path)

                    # read uploaded image as cv2 image for processing
                    if canvas == self.carrier_canvas:
                        self.carrier_image = cv2.imread(file_path, cv2.IMREAD_COLOR_RGB)
                        self.output_print(f'Successfully uploaded carrier/watermarked image - ({file_name})', self.NORMAL)
                    elif canvas == self.watermark_canvas:
                        self.watermark_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        self.output_print(f'Successfully uploaded watermark image of size: {image.size} - ({file_name})', self.NORMAL)

        except UnidentifiedImageError:
            messagebox(title='Error uploading image',
                       message='Image could not be read, please make sure a .png image is uploaded')

    def save_image(self, image):
        """
        Saves the output image to a file chosen by the user
        """
        if image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if file_path:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, image)
                self.output_print('Output image saved succesfully!')
        else:
            self.output_print('No output image to save!', self.ERROR)

    def clear_images(self):
        """
        Clears all uploaded or outputted images from the GUI
        """
        # set images back to None
        self.carrier_image = None
        self.watermark_image = None
        self.output_image = None

        # delete img attributes
        if hasattr(self.carrier_canvas, 'img'):
            del self.carrier_canvas.img
        if hasattr(self.watermark_canvas, 'img'):
            del self.watermark_canvas.img
        if hasattr(self.output_canvas, 'img'):
            del self.output_canvas.img
        if hasattr(self.keypoint_canvas, 'img'):
            del self.keypoint_canvas.img

        # clear canvases
        self.carrier_canvas.delete("all")
        self.watermark_canvas.delete("all")
        self.output_canvas.delete("all")
        self.keypoint_canvas.delete("all")

        self.output_print('Images cleared')

    def clear_output(self):
        """
        Clear all the text in the output terminal
        """
        self.output_console.configure(state="normal")
        self.output_console.delete("1.0", "end")
        self.output_console.configure(state="disabled")
        self.output_console.see(END)

    def default_watermark(self):
        """
        Upload the supplied default watermark image for the watermark
        """
        file_path = 'images/default_watermark.png'

        try:
            image = Image.open(file_path)

            self.add_image(image, self.watermark_canvas) # add image to gui

            # read uploaded image as cv2 image for processing
            self.watermark_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.output_print(f'Using default watermark of size: {image.size}')
        except UnidentifiedImageError:
            messagebox(title='Default watermark cannot be found',
                       message='Default watermark could not be read, please make sure it has not been deleted')

    def getKeypoints(self):
        """
        Calculates the keypoints of the carrier image using the SIFT algorithm

        Returns
        -------
        keypoints: list of cv2.KeyPoint
            Keypoints of the carrier image
        """
        grey = cv2.cvtColor(self.carrier_image, cv2.COLOR_RGB2GRAY)

        # compute keypoints using SIFT
        sift = cv2.SIFT.create(nfeatures=100)
        keypoints = sift.detect(grey,None)

        # draw image with keypoints highlighted
        img = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
        self.plot_keypoints(img, keypoints)

        self.add_image(Image.fromarray(img), self.keypoint_canvas)

        keypoints = sorted(keypoints, key=lambda kp: (round(kp.pt[1], 2), round(kp.pt[0], 2))) # sort keypoints by position

        return keypoints

    def plot_keypoints(self, img, keypoints):
        """
        Plot the keypoints on the image with custom styling
        """
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size * (img.shape[0] / kp.size) * 0.01)
            img = cv2.circle(img, (x, y), radius, (255, 0, 0), -1)

    def embed_watermark(self):
        """
        Embed supplied watermark into keypoints of carrier image
        """
        threading.Thread(target=self._embed_watermark_thread, daemon=True).start()

    def _embed_watermark_thread(self):
        """
        Embed supplied watermark into keypoints of carrier image
        """
        if self.carrier_image is not None and self.watermark_image is not None:
            self.after(0, self.progress_bar.start)
            keypoints = self.getKeypoints()

            output = self.carrier_image.copy()
            output = output - output%2 # make every pixel value even so LSB for each is 0

            _, watermark = cv2.threshold(self.watermark_image, 127, 1, cv2.THRESH_BINARY)

            # remove any keypoints on the edge of the image
            keypoints = [kp for kp in keypoints if self.is_keypoint_valid(kp, output, watermark)]

            height, width, _ = output.shape
            blue_mask = np.zeros((height, width), dtype=bool)
            green_mask = np.zeros((height, width), dtype=bool)
            red_mask = np.zeros((height, width), dtype=bool)

            for kp in keypoints:
                area = self.get_area_coords(output, kp, watermark.shape)

                # if none of the coordinates in the area are in a coloured mask, use that mask to embed the watermark
                if not any(blue_mask[y, x] for (y, x) in area):
                    output = self.embed_bits_at_keypoint(output, kp, watermark, 2)

                    for (y, x) in area:
                        blue_mask[y, x] = True
                elif not any(green_mask[y, x] for (y, x) in area):
                    output = self.embed_bits_at_keypoint(output, kp, watermark, 1)

                    for (y, x) in area:
                        green_mask[y, x] = True
                elif not any(red_mask[y, x] for (y, x) in area):
                    output = self.embed_bits_at_keypoint(output, kp, watermark, 0)
                    for (y, x) in area:
                        red_mask[y, x] = True

            self.add_image(Image.fromarray(output), self.output_canvas)
            self.output_image = output

            self.output_print('Succesfully embedded watermark in carrier image, see output image for result')
            self.after(0, self.progress_bar.stop)
        elif self.carrier_image is None and self.watermark_image is None:
            self.output_print('No carrier or watermark image uploaded', type=self.ERROR)
        elif self.carrier_image is None:
            self.output_print('No carrier image uploaded', type=self.ERROR)
        elif self.watermark_image is None:
            self.output_print('No watermark image uploded', type=self.ERROR)

    def recover_watermark(self):
        """
        Recovers the watermarks embedded in the keypoints of the image and determines whether the image is authentic
        """
        threading.Thread(target=self._recover_watermark_thread, daemon=True).start()

    def _recover_watermark_thread(self):
        """
        Recovers the watermarks embedded in the keypoints of the image and determines whether the image is authentic
        """
        if self.carrier_image is not None and self.watermark_image is not None:
            self.after(0, self.progress_bar.start)
            keypoints = self.getKeypoints()

            carrier = self.carrier_image.copy()

            _, watermark = cv2.threshold(self.watermark_image, 127, 1, cv2.THRESH_BINARY)

            # remove any keypoints on the edge of the image
            keypoints = [kp for kp in keypoints if self.is_keypoint_valid(kp, carrier, watermark)]

            has_watermark = False

            for kp in keypoints:
                recovered_watermark_blue = self.recover_bits_at_keypoint(carrier, kp, watermark.shape, 2)
                recovered_watermark_green = self.recover_bits_at_keypoint(carrier, kp, watermark.shape, 1)
                recovered_watermark_red = self.recover_bits_at_keypoint(carrier, kp, watermark.shape, 0)

                if np.array_equal(recovered_watermark_blue, watermark):
                    has_watermark = True
                    break
                elif np.array_equal(recovered_watermark_green, watermark):
                    has_watermark = True
                    break
                elif np.array_equal(recovered_watermark_red, watermark):
                    has_watermark = True
                    break

            self.output_print()
            self.output_print('Verify authenticity:')
            if has_watermark:
                self.output_print('Yes - watermark found in the image')
            else:
                self.output_print('No - watermark not found in the image')

            self.after(0, self.progress_bar.stop)
        elif self.carrier_image is None and self.watermark_image is None:
            self.output_print('No watermarked or watermark image uploaded', type=self.ERROR)
        elif self.carrier_image is None:
            self.output_print('No watermarked image uploaded', type=self.ERROR)
        elif self.watermark_image is None:
            self.output_print('No watermark image uploded', type=self.ERROR)

    def detect_tampering(self):
        """
        Recovers watermark from keypoints and checks the consistency of each to determine if image has been tampered with
        """
        threading.Thread(target=self._detect_tampering_thread, daemon=True).start()

    def _detect_tampering_thread(self):
        """
        Recovers watermark from keypoints and checks the consistency of each to determine if image has been tampered with
        """
        if self.carrier_image is not None and self.watermark_image is not None:
            self.after(0, self.progress_bar.start)
            keypoints = self.getKeypoints()

            carrier = self.carrier_image.copy()

            _, watermark = cv2.threshold(self.watermark_image, 127, 1, cv2.THRESH_BINARY)

            # remove any keypoints on the edge of the image
            keypoints = [kp for kp in keypoints if self.is_keypoint_valid(kp, carrier, watermark)]

            height, width, _ = carrier.shape
            blue_mask = np.zeros((height, width), dtype=bool)
            green_mask = np.zeros((height, width), dtype=bool)
            red_mask = np.zeros((height, width), dtype=bool)

            unsure_keypoints = []
            matches = 0

            for kp in keypoints:
                area = self.get_area_coords(carrier, kp, watermark.shape)

                recovered_watermark_blue = self.recover_bits_at_keypoint(carrier, kp, watermark.shape, 2)
                recovered_watermark_green = self.recover_bits_at_keypoint(carrier, kp, watermark.shape, 1)
                recovered_watermark_red = self.recover_bits_at_keypoint(carrier, kp, watermark.shape, 0)

                if np.array_equal(recovered_watermark_blue, watermark):
                    matches += 1
                    for (y, x) in area:
                        blue_mask[y, x] = True
                elif np.array_equal(recovered_watermark_green, watermark):
                    matches += 1
                    for (y, x) in area:
                        green_mask[y, x] = True
                elif np.array_equal(recovered_watermark_red, watermark):
                    matches += 1
                    for (y, x) in area:
                        red_mask[y, x] = True
                else:
                    unsure_keypoints.append(kp)


            tampered_keypoints = []

            for kp in unsure_keypoints:
                area = self.get_area_coords(carrier, kp, watermark.shape)

                if not any(blue_mask[y, x] for (y, x) in area):
                    tampered_keypoints.append(kp)
                elif not any(green_mask[y, x] for (y, x) in area):
                    tampered_keypoints.append(kp)
                elif not any(red_mask[y, x] for (y, x) in area):
                    tampered_keypoints.append(kp)

            total_watermarked = matches + len(tampered_keypoints)

            self.output_print()
            self.output_print('Tampering detector:')
            self.output_print(f'{matches} out of {total_watermarked} recovered watermarks matched the supplied watermark')

            threshold = 0.8
            if matches / total_watermarked < threshold:
                self.output_print(f'Tampering HAS been detected (based on a threshold of {threshold*100:.0f}% watermark consistency)')
                self.output_print('Keypoints that do not match the watermark are marked in the keypoint image section')

                grey = cv2.cvtColor(self.carrier_image, cv2.COLOR_RGB2GRAY)
                img = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
                self.plot_keypoints(img, unsure_keypoints)
                self.add_image(Image.fromarray(img), self.keypoint_canvas)

            else:
                self.output_print(f'Tampering HAS NOT been detected (based on a threshold of {threshold*100:.0f}% watermark consistency)')

            self.after(0, self.progress_bar.stop)
        elif self.carrier_image is None and self.watermark_image is None:
            self.output_print('No watermarked or watermark image uploaded', type=self.ERROR)
        elif self.carrier_image is None:
            self.output_print('No watermarked image uploaded', type=self.ERROR)
        elif self.watermark_image is None:
            self.output_print('No watermark image uploded', type=self.ERROR)

    def embed_bits_at_keypoint(self, image, kp, watermark, channel):
        """
        Embed the bits of a watermark in the area surrounding the supplied keypoint

        Parameters
        ----------
        image: MatLike
            Image to add the watermark bits to
        kp: cv2.KeyPoint
            Keypoint location to add the watermark at
        watermark: MatLike
            Watermark to embed at the keypoint
        channel: int
            RGB colour channel to embed bits in
        """
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = watermark.shape[0] // 2
        height, width, _ = image.shape

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                px = x + dx
                py = y + dy

                if 0 <= px < width and 0 <= py < height:
                    original_value = image[py, px, channel] # get original pixel value
                    binary_value = bin(original_value)[2:] # convert to binary
                    bit = watermark[dy, dx] # get bit for current section of watermark
                    binary_value = binary_value[:-1] + str(bit) # change LSB of original value to value of the watermark bit
                    image[py, px, channel] = int(binary_value, 2) # update pixel value

        return image

    def recover_bits_at_keypoint(self, image, kp, shape, channel):
        """
        Recover the bits of the watermark at the supplied keypoint

        Parameters
        ----------
        image: MatLike
            Image to recover the watermark bits from
        kp: cv2.KeyPoint
            Keypoint location to recover the watermark from
        shape: MatLike
            Shape of the watermark to recover
        channel: int
            RGB colour channel to recover bits from
        """
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = shape[0] // 2
        height, width, _ = image.shape
        watermark = np.empty(shape)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                px = x + dx
                py = y + dy

                if 0 <= px < width and 0 <= py < height:
                    value = image[py, px, channel] # get pixel value
                    bit = bin(value)[-1:] # convert to binary and get LSB
                    watermark[dy, dx] = int(bit) # add bit to watermark

        return watermark


    def get_area_coords(self, image, kp, shape):
        """
        Calculates the pixels in the area around the keypoint to be watermarked

        Parameters
        ----------
        image: np.ndarray
            Image to get coordinates of
        kp: cv2.KeyPoint
            Keypoint to calculate area around
        shape: int
            Shape of the area around the keypoint to be calculated

        Returns
        -------
        coords: list of tuple of int
            Coordinates of the area around the keypoint
        """
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = shape[0] // 2
        height, width, _ = image.shape

        coords = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                px = x + dx
                py = y + dy

                if 0 <= px < width and 0 <= py < height:
                    coords.append((py, px))

        return coords

    def is_keypoint_valid(self, kp, image, watermark):
        """
        Check if a keypoint is not on the edge of the image
        """
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = watermark.shape[0] // 2
        height, width = image.shape[:2]
        return (radius <= x < width - radius) and (radius <= y < height - radius)



app = Watermark()
app.mainloop()
