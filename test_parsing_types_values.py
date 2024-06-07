import os 
import yaml
from yaml import SafeLoader
from parsers import Parser
import pytest
import utils


class Tester_Parsing():
    """ TESTS ARGS PARSING TO MAKE SURE WRONG TYPE, FORMAT, VALUES ETC. ARE HANDLED CORRECTLY. """

    def _parse(self) -> None:
        """ DEFINES ALL CLASS ARGS """
        
        # restore original .yaml file
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

        return

    def test_parse_yaml_param_augment(self) -> None: 
        """ TESTING PARSING OF PARAM 'AUGMENT' """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        AUGMENT_NOT_BOOL = "Param 'augment' from .yaml file should be boolean"
        var = 'augment'
        new_val = 'ciao'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=AUGMENT_NOT_BOOL):     #    # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)  
        return
    
    def test_parse_yaml_param_classify(self) -> None: 
        """ TESTING PARSING OF PARAM CLASSIFY """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        CLASSIFY_NOT_BOOL = "Param 'classify' from .yaml file should be boolean"
        var = 'classify'
        new_val = 'ciao'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=CLASSIFY_NOT_BOOL):       # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        return

    def test_parse_yaml_param_create_grayscale_masks(self) -> None: 
        """ TESTING PARSING OF PARAM create_grayscale_masks """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        CREATE_MASKS_NOT_BOOL = "Param 'create_grayscale_masks' from .yaml file should be boolean"
        var = 'create_grayscale_masks'
        new_val = 'ciao'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=CREATE_MASKS_NOT_BOOL):       # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        return

    def test_parse_yaml_param_save_crop(self) -> None: 
        """ TESTING PARSING OF PARAM save_crop """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        SAVE_CROP_NOT_BOOL = "Param 'save_crop' from .yaml file should be boolean"
        var = 'save_crop'
        new_val = 'ciao'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=SAVE_CROP_NOT_BOOL):      # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        return

    def test_parse_yaml_param_save_imgs(self) -> None: 
        """ TESTING PARSING OF PARAM save_imgs """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        SAVE_IMGS_NOT_BOOL = "Param 'save_imgs' from .yaml file should be boolean"
        var = 'save_imgs'
        new_val = 'ciao'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=SAVE_IMGS_NOT_BOOL):      # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        
        return

    def test_parse_yaml_param_save_txt(self) -> None: 
        """ TESTING PARSING OF PARAM save_txt """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        SAVE_TXT_NOT_BOOL = "Param 'save_txt' from .yaml file should be boolean"
        var = 'save_txt'
        new_val = 'ciao'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=SAVE_TXT_NOT_BOOL):       # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        return

    def test_parse_yaml_param_device(self) -> None: 
        """ TESTING PARSING OF PARAM device """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        DEVICE_WRONG_TYPE = "Param 'device' from .yaml file should be type str"
        DEVICE_WRONG_VALUE = "Param 'device' from .yaml file should be either 'gpu' or 'cpu'."
        var = 'device'
        new_val = 37
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=DEVICE_WRONG_TYPE):       # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        new_val = 'GPU'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=DEVICE_WRONG_VALUE):      # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        return

    def test_parse_yaml_input_dir(self) -> None: 
        """ TESTING PARSING OF PARAM device """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        INPUT_DIR_WRONG_TYPE = "Param 'input_dir' from .yaml file should be type str."
        INPUT_DIR_NOT_EXISTS = "'input_dir' does not exist as dirpath or filepath"
        var = 'input_dir'
        new_val = 37
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=INPUT_DIR_WRONG_TYPE):    # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        new_val = os.path.join(os.getcwd(), 'AAAtesting_data')
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=INPUT_DIR_NOT_EXISTS):    # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        return

    def test_parse_yaml_param_model(self) -> None: 
        """ TESTING PARSING OF PARAM show_mask """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        MODEL_NOT_IMPLEMENTED = "'model' should be set to 'yolo'."
        var = 'model'
        new_val = 'ciao'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=MODEL_NOT_IMPLEMENTED):       # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        return

    def test_parse_yaml_output_dir(self) -> None: 
        """ TESTING PARSING OF PARAM device """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        OUTPUT_DIR_WRONG_TYPE = "Param 'output_dir' from .yaml file should be type str."
        var = 'output_dir'
        new_val = 37
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=OUTPUT_DIR_WRONG_TYPE):       # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        return

    def test_parse_yaml_param_task(self) -> None: 
        """ TESTING PARSING OF PARAM task """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        TASK_WRONG_VALUE = "Param 'task' from .yaml file should be either 'segmentation' or 'detection'."
        var = 'task'
        new_val = 'segment'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=TASK_WRONG_VALUE):    # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        return

    def test_parse_trained_model_weights(self) -> None: 
        """ TESTING WEIGHTS_NOT_EXISTS AND WEIGHTS_WRONG_FMT """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        WEIGHTS_WRONG_TYPE = "Param 'trained_model_weights' from .yaml file should be type str."
        WEIGHTS_WRONG_FMT = "Param 'trained_model_weights' from .yaml file should be format .pt."
        WEIGHTS_NOT_EXISTS = f"'trained_model_weights' does not exist or is not a file"
        var = 'trained_model_weights'
        # 1) file doesn't exist
        new_val = os.path.join(os.getcwd(), 'model_weights', 'segm', 'bestAAA.pt')
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=WEIGHTS_NOT_EXISTS):      # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)
        # 2) wrong type
        new_val = 8549364
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=WEIGHTS_WRONG_TYPE):      # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)
        # 3) wrong format
        new_val = os.path.join(os.getcwd(), 'testing_data', 'normal_size.png')
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=WEIGHTS_WRONG_FMT):       # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)
        return

    def test_parse_conf_thr(self) -> None: 
        """ TESTING CONF_THR_WRONG_TYPE  """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        CONF_THR_WRONG_TYPE = f"'conf_thres' should be type float"
        CONF_THR_WRONG_VALUE = f"'conf_thres' should be in the range 0.0<=conf_thr<=1.0"
        var = 'conf_thres'
        # 1) conf_thres str
        new_val = '0.3'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=CONF_THR_WRONG_TYPE):     # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)
        # 2) conf_thres int
        new_val = 2
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=CONF_THR_WRONG_TYPE):     # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)
        # 3) conf_thres not in 0.0<x<1.0
        new_val = 1.2
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=CONF_THR_WRONG_VALUE):    # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)
        return

    def test_parse_yaml_fp(self) -> None: 
        """ TESTING YAML_NOT_EXISTS AND YAML_WRONG_FMT """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        YAML_NOT_EXISTS = f"'yaml_fp' does not exist"
        YAML_WRONG_FMT = f"'yaml_fp' should be format .yaml"
        YAML_WRONG_TYPE = f"'yaml_fp' should be type str"
        # 1) wrong format
        yaml_fp = os.path.join(os.getcwd(), 'loggers.py') 
        with pytest.raises(AssertionError, match=YAML_WRONG_FMT):      # check that code manages issue as expected
            Parser(yaml_fp=yaml_fp)
        # 1) wrong type
        yaml_fp = 7362
        with pytest.raises(AssertionError, match=YAML_WRONG_TYPE):     # check that code manages issue as expected
            Parser(yaml_fp=yaml_fp)
        # 2) file doesn't exist
        yaml_fp = os.path.join(os.getcwd(), 'AAA.yaml')    
        with pytest.raises(AssertionError, match=YAML_NOT_EXISTS):     # check that code manages issue as expected
            Parser(yaml_fp=yaml_fp)
        return

    def test_params_present(self) -> None: 
        """   TEST THAT ALL PARAMS ARE PRESENT IN THE .YAML FILE    """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)     # restore original testing.yaml file before starting test
        with open(self.yaml_fp, 'r') as file:    
            yaml.safe_load(file)
        rem_item = utils.remove_last_item_yaml_fp(yaml_fp=self.yaml_fp, verbose=False)   # remove one (the last) item from yaml
        key = list(rem_item.keys())[0]
        with pytest.raises(AssertionError, match=f"'{key}' param is missing from the .yaml config file."):     # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)
        return 

    def __call__(self) -> None:
        """ RUNS ALL TESTS FOR PARSING """

        self._parse()
        self.test_params_present()
        self.test_parse_conf_thr()
        self.test_parse_trained_model_weights()
        self.test_parse_yaml_fp()
        self.test_parse_yaml_input_dir()
        self.test_parse_yaml_output_dir()
        self.test_parse_yaml_param_augment()
        self.test_parse_yaml_param_classify()
        self.test_parse_yaml_param_create_grayscale_masks()
        self.test_parse_yaml_param_device()
        self.test_parse_yaml_param_model()
        self.test_parse_yaml_param_save_crop()
        self.test_parse_yaml_param_save_imgs()
        self.test_parse_yaml_param_save_txt()
        self.test_parse_yaml_param_task()
        return


if __name__ == "__main__": 

    Tester = Tester_Parsing()
    Tester()