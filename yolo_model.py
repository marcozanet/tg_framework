import os
import torch
from parsers import Parser
from image_creator import ImageCreator
import utils 

class Yolo_Model(Parser):
    """"   RUNS YOLOV5 MODEL ON INPUT_DIR IMAGES (OR ON A SINGLE IMAGE IN CASE INPUT_DIR IS A FILEPATH)    """

    def __init__(self, 
                 yaml_fp:str,
                #  img_fp:str,
                 verbose:bool = True) -> None: 

        
        super().__init__(yaml_fp, verbose)
        return
    

    def _select_device(self) -> None:
        """ HELP FUNC: SELECTS WHETHER MODEL RUNS ON GPU OR CPU """

        device = 'cpu' if self.params['device'] == 'cpu' else 'gpu'
        if device == 'gpu':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            if device == 'cpu': 
                self.log.warn(f"Device set to 'gpu' but gpu wasn't found. Running on cpu.")
        self.device = device

        return


    def infere(self) -> None:
        """ PREDICTS BOUNDING BOXES OR SEGMENTATION MASKS ON OBJECTS WITHIN THE IMAGE """

        os.chdir(self.params['yolov5dir'])
        # define command:
        command = "python detect.py" if self.params['task']=="detection" else "python segment/predict.py"
        self.log.info(f"Task is {self.params['task']}")
        command += f" --source {self.input_dir} --weights {self.params['trained_model_weights']}"
        command += f"  --data data/helical.yaml --device {self.device}"
        if self.params['augment'] is True:
            command += " --augment"
        if self.params['save_imgs'] is False:
            command += " --nosave"
        if self.params['save_txt'] is True: 
           command +=" --save-txt"
        if self.params['save_crop'] is True:
            command +=" --save-crop"
        if self.params['conf_thres'] is not None:
            command += f" --conf-thres {self.params['conf_thres']}" 
        command += f" --project {os.path.dirname(self.output_dir)}" 
        command += f" --name {os.path.basename(self.output_dir)}" 
        # infere (e.g. predict):
        self.log.info(f"Start inference YOLO: ⏳")
        os.system(command)
        os.chdir(self.params['repository_dir'])
        self.log.info(f"Inference YOLO done ✅ .")

        return


    def __call__(self) -> None:
        """ RUNS PREDICTION ON INPUT DATA """

        self._select_device()   # selects cpu or gpu
        self.infere()   # runs prediction (detection or segmentation) on input images
        if self.params['create_grayscale_masks'] is True:   # create output grayscale masks
            mask_converter = ImageCreator(yaml_fp=self.yaml_fp)
            mask_converter()

        return



if __name__ == '__main__': 

    _, yaml_fp = utils.setup_paths()
    img_fp = '/Users/marco/tg/tg_framework/testing_data/I_1_S_3_ROI_2_PAS_sample0_1_3.png'
    utils.edit_yaml_fp(yaml_fp=yaml_fp, field='inference', var='input_dir', new_val=img_fp)
    detector = Yolo_Model(yaml_fp=yaml_fp)
    detector()



