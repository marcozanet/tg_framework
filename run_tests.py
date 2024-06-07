import pytest 
import shutil
import os
import utils

def run_pytest():
    """ RUNS A SERIES OF TESTS OT MAKE SURE PARSING WORKS ON DIFFERENT ARGS AND 
        MODEL RUNS ACROSS A VARIETY OF INPUTS AND PARAMS. """
    
    HOMEDIR = os.getcwd()
    safe_copy_yaml, yaml_fp = utils.setup_paths()
    shutil.rmtree(os.path.join(HOMEDIR, 'results'), ignore_errors=True)  # clean test_output before running new tests
    result = pytest.main(["-v"])    # You can specify additional arguments for pytest in the list, e.g., ['-v', '--tb=short']
    utils.restore_yaml(original_yaml=safe_copy_yaml, modified_yaml=yaml_fp)     # restores original testing .yaml file after all tests are run 

    return result



if __name__ == "__main__":
    exit_code = run_pytest()
    print(f"pytest finished with exit code: {exit_code}")