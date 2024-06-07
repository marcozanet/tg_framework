import os 
import utils 
import yaml
import random
from tqdm import tqdm
from yolo_model import Yolo_Model
from itertools import product


class Tester_Grid_Combos():

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
        self.n_desired_tests = 5
        params = utils.get_config_params(self.yaml_fp, config_name='inference')
        self.homedir = params['repository_dir']
        self.safe_copy_yaml = safe_copy_yaml
        self.yaml_fp =  yaml_fp
        self.field = 'inference'
        models = utils.find_pt_files(os.path.join(self.homedir, 'model_weights'))
        detect_models = [model for model in models if 'detect' in model]
        segm_models = [model for model in models if 'segm' in model]
        assert len(detect_models)>0, f"No detection model weights were found in homedir."
        assert len(segm_models)>0, f"No segmentation model weights were found in homedir."
        self.detect_model = detect_models[0]
        self.segm_model = segm_models[0]
        self.dictionary =  {'augment': [False, True],
                            'classify': [False, True],
                            'conf_thres': [0.3, 0.8],
                            'create_grayscale_masks': [False, True],
                            'device': ['cpu', 'gpu'],
                            'input_dir': [os.path.join(self.homedir, 'testing_data/normal_size.png'), 
                                          os.path.join(self.homedir, 'testing_data')],
                            'model': ['yolo'],
                            'output_dir': [None, os.path.join(self.homedir, 'results/testing')] ,
                            'repository_dir': [self.homedir],
                            'save_crop': [False, True],
                            'save_imgs': [False, True],
                            'save_txt': [False, True],
                            'task': ['segmentation', 'detection'],
                            'trained_model_weights': models,
                            'yolov5dir': [params['yolov5dir']]}
        return
    
    def _find_pt_files(self, home_dir:str) -> list:
        """ HELP FUNC TO FIND ALL MODEL WEIGHTS IN .pt FORMAT """

        pt_files = []
        for root, _, files in os.walk(home_dir):
            for file in files:
                if file.lower().endswith('.pt'):
                    pt_files.append(os.path.join(root, file))

        return pt_files

    def test_all_combos(self) -> None:
        """ TEST A NUMBER OF COMBINATIONS FOR ALL PARAMS AND CHECK THAT THE MODEL WORKS SMOOTHLY. """

        self._parse()   # sets all class necessary args 
        all_keys, all_values = list(self.dictionary.keys()), list(self.dictionary.values())
        combos_values = list(product(*all_values))  # generate all possible combinations
        random.shuffle(combos_values)   # shuffle combinations 
        combos_values = combos_values[:self.n_desired_tests] # limit number of combo to 'n_desired_tests'
        # Run all test combinations:
        for i, combo in enumerate(tqdm(combos_values, desc=f"Testing combo: ")): 
            print(f" ##########     ⏳ START TEST {i}     ##########")
            utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)   # restore original testing.yaml file before starting test 
            pairs = zip(all_keys, combo)
            dictionary = {k:v for k,v in pairs }
            new_data = {'inference': dictionary}
            with open(self.yaml_fp, 'w') as file:
                yaml.safe_dump(new_data, file, default_flow_style=False)    # edit .yaml file with a new combination of params to be tested
            if utils.is_there_incoherence_in_params(yaml_fp=self.yaml_fp) is True:  # if params are incoherent, skip it (issue management already being tested separately)
                continue
            detector = Yolo_Model(yaml_fp=self.yaml_fp, verbose = False)
            detector()        
            print(f" ##########     🎉 DONE TEST {i}      ##########")


        return 


if __name__ == "__main__":
    Tester = Tester_Grid_Combos()
    Tester.test_all_combos()






