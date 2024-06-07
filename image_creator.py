import os
from pathlib import Path
import cv2
import numpy as np 
from PIL import Image, ImageDraw
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from typing import List
from typing import List, Tuple
from glob import glob
import utils
from parsers import Parser

class ImageCreator(Parser):
    """"   CONVERTS TXT FILE INTO A GRAYSCALE LABEL-CODED OUTPUT IMAGE (AND, IF SELECTED, ALSO TO AN RGB IMAGE)    """

    def __init__(self, 
                 yaml_fp: str,
                 verbose: bool = True) -> None: 

        super().__init__(yaml_fp, verbose)
        self.colors =   [   (0, 0, 0),       # 0: Black
                            (255, 0, 0),     # 1: Blue
                            (0, 255, 0),     # 2: Green
                            (0, 0, 255),     # 3: Red
                            (255, 255, 0),   # 4: Cyan
                            (255, 0, 255),   # 5: Magenta
                            (0, 255, 255),   # 6: Yellow
                            (128, 128, 128), # 7: Gray
                            (128, 0, 128),   # 8: Purple
                            (0, 128, 128)  ] # 9: Teal
        return
    

    def _get_last_output_dir(self) -> str: 
        """  HELP FUNC: GETS LAST EXP FOLDER WHERE RESULTS WERE SAVED """

        # check that similar fold names exist in same folder like name, name2, name3 etc. 
        output_dir = self.output_dir
        homedir = os.path.dirname(output_dir)
        subfolds = [fold for fold in os.listdir(homedir) if os.path.isdir(os.path.join(homedir, fold))]
        assert len(subfolds)>0, f"No output dir created."
        if len(subfolds) == 1:
            if subfolds[0] == os.path.basename(output_dir):
                last_output_dir = output_dir
            else:
                raise NotImplementedError(f"{subfolds[0]} != {os.path.basename(output_dir)}")
        else:   # if more than one with same name, retrieves the last edited
            last_output_dir = utils.get_last_modified_folder(parent_folder=homedir)
        self.last_output_dir = last_output_dir

        return last_output_dir
    

    def make_all_grayscale_masks(self) -> None:
        """ CREATES GRAYSCALE LABEL CODED IMAGES OUT OF TXT LABEL FILES AND SAVES THEM """

        self._get_last_output_dir()     # get where results are saved
        txt_files = glob(os.path.join(self.last_output_dir, 'labels', "*.txt"))  # get all label txt files
        for txt_file in txt_files:
            self._convert_txtfile_to_grayscaleimg(txt_file=txt_file) # create one grayscale image and saves it

        return
    
    
    def _convert_txtfile_to_grayscaleimg(self, txt_file:str) -> None:
        """ CREATES ONE GRAYSCALE LABEL CODED IMAGE OUT OF A TXT LABEL FILE AND SAVES IT """

        img_basename = os.path.basename(txt_file).split('.')[-2]
        img_path = self._get_img_path(img_basename=img_basename) # gets the input image corresponding to the txt file
        h,w = self._get_img_w_h(img_path=img_path) #  gets shape of the input image to scale coords in the txt file
        objects = self._parse_text_file(txt_file=txt_file, h=h, w=w) # get objects coords from the txt file
        image_array = self.make_grayscale_image(objects, width=w, height=h) # create the grayscale image
        grayscale_img_fp = os.path.join(self.last_output_dir, img_basename + '_mask.png')
        self._save_image(image_array=image_array, output_path= grayscale_img_fp)
        self._make_rgb_image(grayscale_img_fp=grayscale_img_fp)

        return
    
    
    def _make_rgb_image(self, grayscale_img_fp:str) -> None:
        """ CREATES AND SAVES AN RGB COLOR CODED IMAGE """

        # Load the grayscale image
        grayscale_image = cv2.imread( grayscale_img_fp, cv2.IMREAD_GRAYSCALE)
        # Color code the image
        color_coded_image = self._color_code_image(grayscale_image)
        output_fp = os.path.join(self.last_output_dir, os.path.basename(grayscale_img_fp.split('.')[0])+'_RGB.png')
        cv2.imwrite(output_fp, color_coded_image)

        return
    

    def _color_code_image(self, grayscale_image) -> np.ndarray:
        """ MAKES AN RGB COLOR CODED IMAGE OUT OF THE GRAYSCALE LABEL CODED IMAGE """

        # Initialize an empty RGB image
        height, width = grayscale_image.shape
        color_coded_image = np.zeros((height, width, 3), dtype=np.uint8)
        # Apply the color coding
        for value, color in enumerate(self.colors):
            color_coded_image[grayscale_image == value] = color

        return color_coded_image
        
    
    def _parse_text_file(self, txt_file:str, h:int, w:int) -> List[Tuple[int,int]]:
        """ HELPER FUNC TO PARSE TXT FILE AND GET LABELS AND COORDS OF OBJECTS """

        assert self.params['save_txt'] is True, f"Trying to retrieve txt file but 'save_txt' was set to false. "
        objects = []
        with open(txt_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                label = int(parts[0])
                coordinates = list(map(float, parts[1:]))
                scale_coord = lambda i, dim:  min(int(coordinates[i]*dim), dim-1)
                vertices = [(scale_coord(i, w), scale_coord(i+1, h)) for i in range(0, len(coordinates), 2)]
                objects.append((label, vertices))

        return objects
    

    def _get_img_path(self, img_basename:str) -> Path:
        """ HELP FUNC: GETS CORRESPONDING INPUT IMAGE TO THE OUTPUT TXT LABEL FILE """

        # from input dir retrieve image file with same name
        input_dir = self.input_dir if os.path.isdir(self.input_dir) else os.path.dirname(self.input_dir)
        subfiles = os.listdir(input_dir) 
        matching_files = [os.path.join(input_dir, file) for file in subfiles if img_basename == file.split('.')[0]]
        assert len(matching_files)>0, AssertionError(f"{img_basename.split('.')[0]} is not among the images of the input_dir.")
        assert len(matching_files)==1, AssertionError(f"Multiple files in input_dir are named with the same name: {img_basename.split('.')[0]}.")
        img_path = matching_files[0] 
        assert os.path.isfile(img_path), f"'img_path':{img_path} is not a filepath."

        return matching_files[0]
    
    
    def _get_img_w_h(self, img_path:str) -> Tuple[int,int] :
        """ HELP FUNC: GETS W, H OF AN IMAGE """

        image = cv2.imread(img_path)
        w, h  = image.shape[:2]

        return w, h

    
    def make_grayscale_image(self, objects, width, height) -> Image:
        """ HELP FUNC: GETS CORRESPONDING INPUT IMAGE TO THE OUTPUT TXT LABEL FILE """

        # Create a blank image with all pixels set to 0 (background)
        image_array = np.zeros((height, width), dtype=np.uint8)
        # Create an Image object for drawing polygons
        image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(image)
        # Draw each object
        for label, vertices in objects:
            label = label + 1 # otherwise 0 blends with bg
            draw.polygon(vertices, outline=label, fill=label)
        # Convert back to numpy array
        image_array = np.array(image)

        return image_array
    

    def _save_image(self, image_array, output_path) -> None:
        """ SAVES IMAGE """

        image = Image.fromarray(image_array)
        image.save(output_path)    

        return


    def __call__(self) -> None:
        """  RUNS CONVERSION FROM TXT LABELS TO A GRAYSCALE IMAGE WITH LABELS COLOR CODED """

        self.make_all_grayscale_masks()

        return
    



if __name__ == '__main__': 
    
    yaml_fp = '/Users/marco/tg/tg_framework/testing.yaml'
    detector = ImageCreator(yaml_fp=yaml_fp)
    detector()


