# COM31006 Computer Vision Assignment - Watermark Creation and Detection Tool

*Joel Foster, 2025*

## Dependencies

To install the necessary dependencies for the program, run the following command: ```pip install -r requirements.txt```
in your .venv, .conda, or global python environment.

## Running the program

To run the program, navigate to root directory of the program, and execute one of the following commands:
```python watermark.py```, ```python3 watermark.py``` or ```py watermark.py``` based on what your environment uses.

### DPI Scaling

In the event of the GUI not scaling correctly automatically based on your operating system and environment, the argument ```--scale``` can be used when executing the program to manually override the program's DPI scale. For example, if you would like to force 2x scaling, you would run the following command ```python watermark.py --scale 2.0```. If you find an appropriate overriding scale, you can save this with the command ```python watermark.py --scale x --save-scale```, where **x** is the scale you desire. The program will then apply this scale whenever you execute it using just ```python watermark.py```.

## Navigating and using the program

To start using the program, the first step would be to embed a watermark into an image. To achieve this, upload a
carrier image to the carrier image section, and a watermarked image to the watermark image section. Then, select the
'Embed watermark' button. This will produce a version of the carrier image with the watermark embedded, which can be
saved by pressing the 'Save output image' button.

The carrier image section can also be used to upload a watermarked image, which can be authenticated or detected for
tampering against a supplied watermark.

Key points of an image are displayed in the key points section, and when tamper detection is applied, the key points
shown are any key points that are not consistent with the watermark.

## Example images

Example images can be found in [images/](images/), which contain carrier images, watermarks, watermarked carrier images, and tampered watermarked images. These can be used to experiment with the system.
