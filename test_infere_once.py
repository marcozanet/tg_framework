from yolo_model import Yolo_Model
import utils
import os

def test_once():

    _, yaml_fp = utils.setup_paths()
    img_fp = os.path.join(os.getcwd(), 'testing_data', 'I_1_S_3_ROI_2_PAS_sample0_1_3.png' )
    utils.edit_yaml_fp(yaml_fp=yaml_fp, field='inference', var='input_dir', new_val=img_fp)
    detector = Yolo_Model(yaml_fp=yaml_fp)
    detector()

    return


if __name__ == '__main__':

    test_once()