import os 
import utils 
from yolo_model import Yolo_Model


class Tester_Specific_Values():
    """ TESTS SOME SPECIFIC PARAM VALUES """

    def _parse(self) -> None:
        """ DEFINES ALL CLASS ARGS """
        
        # restore original .yaml file
        self.n_desired_tests = 5
        safe_copy_yaml, yaml_fp = utils.setup_paths()
        params = utils.get_config_params(yaml_fp=yaml_fp, config_name='inference')
        self.homedir = params['repository_dir']
        self.yaml_fp =  yaml_fp
        self.safe_copy_yaml = safe_copy_yaml
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        self.field = 'inference'
        models = utils.find_pt_files(os.path.join(self.homedir, 'model_weights'))
        detect_models = [model for model in models if 'detect' in model]
        segm_models = [model for model in models if 'segm' in model]
        assert len(detect_models)>0, f"No detection model weights were found in homedir."
        assert len(segm_models)>0, f"No segmentation model weights were found in homedir."
        self.detect_model = detect_models[0]
        self.segm_model = segm_models[0]
        yolov5dir = params['yolov5dir']
        print(yolov5dir)
        self.dictionary =  {'augment': [False, True],
                            'classify': [False, True],
                            'conf_thres': [0.3, 0.8],
                            'create_grayscale_masks': [False, True],
                            'device': ['cpu', 'gpu'],
                            'input_dir': [os.path.join(self.homedir, 'testing_data/normal_size.png'), 
                                          os.path.join(self.homedir, 'testing_data')],
                            'model': ['yolo'],
                            'output_dir': [None] ,
                            'repository_dir': [self.homedir],
                            'save_crop': [False, True],
                            'save_imgs': [False, True],
                            'save_txt': [False, True],
                            'task': ['segmentation', 'detection'],
                            'trained_model_weights': models,
                            'yolov5dir': [params['yolov5dir']]}
        print(self.dictionary['yolov5dir'])
        # raise NotImplementedError()
        return
    
    def _find_pt_files(self, home_dir:str) -> list:
        """ HELP FUNC TO FIND ALL MODEL WEIGHTS IN .pt FORMAT """

        pt_files = []
        for root, _, files in os.walk(home_dir):
            for file in files:
                if file.lower().endswith('.pt'):
                    pt_files.append(os.path.join(root, file))

        return pt_files

    def test_classify_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'CLASSIFY' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'classify'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_conf_thres_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'CONF_THRES' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'conf_thres'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_create_grayscale_masks_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'CREATE_GRAYSCALE_MASKS' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'create_grayscale_masks'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_device_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'DEVICE' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'device'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_input_dir_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'INPUT_DIR' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'input_dir'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_model_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'MODEL' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'model'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_output_dir_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'OUTPUT_DIR' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'output_dir'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_repository_dir_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'REPOSITORY_DIR' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'repository_dir'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_save_crop_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'SAVE_CROP' """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'save_crop'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_save_imgs_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'SAVE_IMGS' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'save_imgs'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_save_txt_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'SAVE_TXT' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'save_txt'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_task_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'TASK' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'task'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_trained_model_weights_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'TRAINED_MODEL_WEIGHTS' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'trained_model_weights'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def test_yolov5dir_values(self) -> None:
        """  TESTS SPECIFIC VALUES FOR PARAM 'YOLOV5DIR' """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
        key = 'yolov5dir'
        for val in self.dictionary[key]:
            utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=key, new_val=val)
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True: 
                utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp)
            detector()
        return

    def __call__(self) -> None:
        """ RUNS ALL TESTS FOR COHERENCE """

        self._parse()
        self.test_classify_values()
        self.test_conf_thres_values()
        self.test_create_grayscale_masks_values()
        self.test_device_values()
        self.test_input_dir_values()
        self.test_model_values()
        self.test_output_dir_values()
        self.test_yolov5dir_values()
        self.test_repository_dir_values()
        self.test_save_crop_values()
        self.test_save_imgs_values()
        self.test_save_txt_values()
        self.test_task_values()
        self.test_trained_model_weights_values()

        return

if __name__ == "__main__": 
    tester = Tester_Specific_Values()
    tester()





