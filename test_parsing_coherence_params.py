import os
import utils
import pytest
from parsers import Parser


class Tester_Coherence():
    """  TESTS THAT PARSING CORRECTLY HANDLES INCOHERENCE AMONG PARAMS  """

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
    
    
    def test_parse_coherence_segm_weights(self) -> None: 
        """ TESTS THAT PARSING CORRECTLY HANDLES INCOHERENCE IN PARAMS 
            BETWEEN SEGMENTATION AND WRONG CHOICE OF MODEL WEIGHTS (DETECTION).
            E.g., Weights of segmentation models should be used only in 
            segmentation tasks (not in detection) and viceversa.   """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)   # restore original testing.yaml file before starting test 
        COHERENCE_SEGM_WEIGHTS = "'segm' not in 'trained_model_weights' although task is 'segmentation'."
        var1 = 'task'
        new_val = 'segmentation'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var1, new_val=new_val)
        var2 = 'trained_model_weights'
        new_val = self.detect_model
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var2, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=COHERENCE_SEGM_WEIGHTS):   # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    
        
        return

    def test_parse_coherence_detect_weights(self) -> None: 
        """ TESTS THAT PARSING CORRECTLY HANDLES INCOHERENCE IN PARAMS 
            BETWEEN DETECTION AND WRONG CHOICE OF MODEL WEIGHTS (SEGMENTATION).
            E.g., Weights of segmentation models should be used only in 
            segmentation tasks (not in detection) and viceversa.   """
        
        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)   # restore original testing.yaml file before starting test
        COHERENCE_DETECT_WEIGHTS = "'detect' not in 'trained_model_weights' although task is 'detection'."
        var1 = 'task'
        new_val = 'detection'
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var1, new_val=new_val)
        var2 = 'trained_model_weights'
        new_val = self.segm_model
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var2, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=COHERENCE_DETECT_WEIGHTS):   # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    

        return

    def test_parse_coherence_grayscale_txt(self) -> None: 
        """ TESTS THAT PARSING CORRECTLY HANDLES INCOHERENCE IN PARAMS 
            BETWEEN 'CREATE_GRAYSCALE_IMAGE' and 'SAVE_TXT'.
            E.g., create_grayscale_masks=True, save_txt should be also 
            True since it's used to create the grayscale masks   """

        self._parse()
        utils.restore_yaml(original_yaml=self.safe_copy_yaml, modified_yaml=self.yaml_fp)   # restore original testing.yaml file before starting test
        COHERENCE_TXTFILES_GRAYSCALEIMG = "'save_txt' is set to False although create_grayscale_masks is True. In order to create masks, output txt labels are necessary."
        var1 = 'create_grayscale_masks'
        new_val = True
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var1, new_val=new_val)
        var2 = 'save_txt'
        new_val = False
        utils.edit_yaml_fp(yaml_fp=self.yaml_fp, field=self.field, var=var2, new_val=new_val)  # edit .yaml file with wrong params
        with pytest.raises(AssertionError, match=COHERENCE_TXTFILES_GRAYSCALEIMG):   # check that code manages issue as expected
            Parser(yaml_fp=self.yaml_fp)    

        return

    def __call__(self, ) -> None:
        """ RUNS ALL TESTS FOR COHERENCE """

        self.test_parse_coherence_detect_weights()
        self.test_parse_coherence_grayscale_txt()
        self.test_parse_coherence_segm_weights()

        return



if __name__ == "__main__":
    tester = Tester_Coherence()
    tester.test_parse_coherence_detect_weights()