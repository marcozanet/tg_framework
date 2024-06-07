import os
import yaml
from yaml import SafeLoader
import loggers
import utils


class Parser():
    """"   RUNS YOLOV5 MODEL ON INPUT_DIR IMAGES (OR ON A SINGLE IMAGE IN CASE INPUT_DIR IS A FILEPATH)    """

    def __init__(self, 
                 yaml_fp: str,
                 verbose: bool = True) -> None: 

        self.yaml_fp = yaml_fp
        self.log = loggers.get_logger(verbose=verbose)
        loggers.set_logging_level('CRITICAL')
        self._parse()

        return
    

    def _get_output_dir(self) -> str: 
        """ GETS OUTPUT DIRECTORY """

        task_fold = "out_segm/exp" if self.params['task'] == 'segmentation' else "out_detect/exp"
        output_dir = os.path.join(self.params['repository_dir'], 'results', task_fold) if self.params['output_dir'] is None else os.path.join(self.params['output_dir'], task_fold)
        self.log.info(f"OUPUT: {output_dir}")

        return output_dir
    

    def _parse(self) -> None:
        """ PARSES ARGS, PARAMS AND DEFINES CLASS ARGS """

        self._parse_args()
        self._parse_params()
        self.input_dir = self.params['input_dir']
        self.output_dir = self._get_output_dir()

        return
    
    
    def _parse_params(self) -> None: 
        """ PARSES PARAMS FROM THE .YAML FILE """

        self.params = utils.get_config_params(yaml_fp=self.yaml_fp, config_name='inference')
        self._parse_params_check_all_present()
        self._parse_params_check_types_and_values()
        self._parse_parms_coherence()

        return
    

    def _parse_args(self) -> None:
        """ PARSES CLASS ARGS """

        # DEFINE ASSERTIONERRORS TO BE MATCHED WITH THE ASSERTION THAT GETS RAISED: 
        YAML_NOT_EXISTS = f"'yaml_fp' does not exist"
        YAML_WRONG_FMT = f"'yaml_fp' should be format .yaml"
        YAML_WRONG_TYPE = f"'yaml_fp' should be type str"

        # PARSING PARAMS TYPE AND VALUES:
        assert isinstance(self.yaml_fp, str), TypeError(YAML_WRONG_TYPE)
        assert os.path.isfile(self.yaml_fp), FileNotFoundError(YAML_NOT_EXISTS)
        yaml_fp_fmt = os.path.basename(self.yaml_fp).split('.')[-1]     
        assert yaml_fp_fmt == 'yaml', TypeError(YAML_WRONG_FMT)

        return
    
    
    def _parse_params_check_types_and_values(self) -> None:
        """ PARSES PARAMS TYPES, FORMATS, VALUES ETC. """

        # DEFINE ASSERTIONERRORS TO BE MATCHED WITH THE ASSERTION THAT GETS RAISED: 
        AUGMENT_NOT_BOOL = "Param 'augment' from .yaml file should be boolean"
        CLASSIFY_NOT_BOOL = "Param 'classify' from .yaml file should be boolean"
        CREATE_MASKS_NOT_BOOL = "Param 'create_grayscale_masks' from .yaml file should be boolean"
        SAVE_CROP_NOT_BOOL = "Param 'save_crop' from .yaml file should be boolean"
        SAVE_IMGS_NOT_BOOL = "Param 'save_imgs' from .yaml file should be boolean"
        SAVE_TXT_NOT_BOOL = "Param 'save_txt' from .yaml file should be boolean"
        DEVICE_WRONG_TYPE = "Param 'device' from .yaml file should be type str"
        DEVICE_WRONG_VALUE = "Param 'device' from .yaml file should be either 'gpu' or 'cpu'."
        INPUT_DIR_WRONG_TYPE = "Param 'input_dir' from .yaml file should be type str."
        INPUT_DIR_NOT_EXISTS = "'input_dir' does not exist as dirpath or filepath"
        MODEL_NOT_IMPLEMENTED = "'model' should be set to 'yolo'."
        OUTPUT_DIR_WRONG_TYPE = "Param 'output_dir' from .yaml file should be type str."
        TASK_WRONG_VALUE = "Param 'task' from .yaml file should be either 'segmentation' or 'detection'."
        YOLOV5DIR_WRONG_TYPE = "Param 'yolov5dir' from .yaml file should be type str."
        WEIGHTS_WRONG_TYPE = "Param 'trained_model_weights' from .yaml file should be type str."
        WEIGHTS_WRONG_FMT = "Param 'trained_model_weights' from .yaml file should be format .pt."
        WEIGHTS_NOT_EXISTS = f"'trained_model_weights' does not exist or is not a file"
        CONF_THR_WRONG_TYPE = f"'conf_thres' should be type float"
        CONF_THR_WRONG_VALUE = f"'conf_thres' should be in the range 0.0<=conf_thr<=1.0"

        # PARSING PARAMS TYPE AND VALUES:
        assert isinstance(self.params['augment'], bool), TypeError(AUGMENT_NOT_BOOL)
        assert isinstance(self.params['classify'], bool), TypeError(CLASSIFY_NOT_BOOL)
        assert isinstance(self.params['create_grayscale_masks'], bool), TypeError(CREATE_MASKS_NOT_BOOL)
        assert isinstance(self.params['save_crop'], bool), TypeError(SAVE_CROP_NOT_BOOL)
        assert isinstance(self.params['save_imgs'], bool), TypeError(SAVE_IMGS_NOT_BOOL)
        assert isinstance(self.params['save_txt'], bool), TypeError(SAVE_TXT_NOT_BOOL)
        assert isinstance(self.params['device'], str), TypeError(DEVICE_WRONG_TYPE)
        assert self.params['device'] in ['gpu', 'cpu'], ValueError(DEVICE_WRONG_VALUE)
        assert isinstance(self.params['input_dir'], str), TypeError(INPUT_DIR_WRONG_TYPE)
        assert os.path.isdir(self.params['input_dir']) or os.path.isfile(self.params['input_dir']), ValueError(INPUT_DIR_NOT_EXISTS)
        assert self.params['model'] == 'yolo', NotImplementedError(MODEL_NOT_IMPLEMENTED)
        assert isinstance(self.params['output_dir'], str) or self.params['output_dir'] is None, TypeError(OUTPUT_DIR_WRONG_TYPE)
        assert self.params['task'] in ['segmentation', 'detection'], ValueError(TASK_WRONG_VALUE)
        assert isinstance(self.params['yolov5dir'], str), TypeError(YOLOV5DIR_WRONG_TYPE)
        assert isinstance(self.params['trained_model_weights'], str), TypeError(WEIGHTS_WRONG_TYPE)
        assert os.path.isfile(self.params['trained_model_weights']), FileNotFoundError(WEIGHTS_NOT_EXISTS)
        trained_model_weights_fmt = os.path.basename(self.params['trained_model_weights']).split('.')[-1]  
        assert trained_model_weights_fmt == 'pt', TypeError(WEIGHTS_WRONG_FMT)
        assert isinstance(self.params['conf_thres'], float), TypeError(CONF_THR_WRONG_TYPE)
        assert 0.<=self.params['conf_thres']<=1., ValueError(CONF_THR_WRONG_VALUE)

        return
    
    
    def _parse_parms_coherence(self) -> None: 
        """ PARSES COHERENCE AMONG PARAMS """

        # DEFINE ASSERTIONERRORS TO BE MATCHED WITH THE ASSERTION THAT GETS RAISED: 
        COHERENCE_SEGM_WEIGHTS = "'segm' not in 'trained_model_weights' although task is 'segmentation'."
        COHERENCE_DETECT_WEIGHTS = "'detect' not in 'trained_model_weights' although task is 'detection'."
        COHERENCE_TXTFILES_GRAYSCALEIMG = "'save_txt' is set to False although create_grayscale_masks is True. In order to create masks, output txt labels are necessary."
        
        # PARSING COHERENCE AMONG PARAMS:
        if self.params['task'] == 'segmentation': 
            assert 'segm' in self.params['trained_model_weights'], ValueError(COHERENCE_SEGM_WEIGHTS)
        elif self.params['task'] == 'detection': 
            assert 'detect' in self.params['trained_model_weights'], ValueError(COHERENCE_DETECT_WEIGHTS)
        if self.params['create_grayscale_masks'] is True:
            assert self.params['save_txt'] is True, ValueError(COHERENCE_TXTFILES_GRAYSCALEIMG)

        return

    
    def _parse_params_check_all_present(self) -> None: 
        """ PARSES PARAMS: CHECKS THAT ALL PARAMS ARE PRESENT IN THE .YAML FILE """

        all_keys = ['augment', 'classify', 'conf_thres', 'create_grayscale_masks', 'device', 'input_dir', 'model', 'output_dir', 
                    'repository_dir', 'save_crop', 'save_imgs', 'save_txt', 'task', 'trained_model_weights', 'yolov5dir' ]
        with open(self.yaml_fp, 'r') as f: 
            all_params = yaml.load(f, Loader=SafeLoader)
        params = all_params['inference']
        yaml_keys = list(params.keys())
        for key in all_keys: 
            assert key in yaml_keys, TypeError(f"'{key}' param is missing from the .yaml config file.")

        return     
    
    
    def __call__(self) -> None: 
        """ RUNS ALL THE PARSING FOR ARGS AND PARAMS """

        self._parse()

        return
    
    

if __name__ == '__main__': 
    
    yaml_fp = '/Users/marco/tg/tg_framework/testing.yaml'
    detector = Parser(yaml_fp=yaml_fp)
    detector()


