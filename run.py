from yolo_model import Yolo_Model
import utils
import os

def tg_run(img_fp:str):

    assert os.path.isfile(img_fp) or os.path.isdir(img_fp), ValueError(f"'img_fp' is not a valid dirpath or filepath.")
    yaml_fp = os.path.join(os.getcwd(), 'config.yaml')
    utils.edit_yaml_fp(yaml_fp=yaml_fp, field='inference', var='input_dir', new_val=img_fp)
    detector = Yolo_Model(yaml_fp=yaml_fp)
    detector()

    return


if __name__ == '__main__':

    img_fp = '/Users/marco/tg/tg_framework/testing_data/I_1_S_3_ROI_2_PAS_sample0_1_3.png'
    tg_run(img_fp=img_fp)